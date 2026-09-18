"""Проверка наличия и загрузка локальных моделей.

Пользователь не должен сам искать названия моделей и выполнять команды для
каждой из них — приложение показывает, чего не хватает, во сколько обойдётся
загрузка, и скачивает только после явного запроса. Молча тянуть гигабайты
весов приложение не имеет права, поэтому загрузка запускается исключительно
через явный вызов, а не автоматически при старте.
"""

from __future__ import annotations

import enum
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.errors import InferenceError, NotFoundError, ValidationError
from app.hardware.profiles import (
    HardwareProfile,
    ModelRingEntry,
    get_profile_definition,
)
from app.llm.base import LocalLLMProvider

logger = logging.getLogger("rag.provisioning")

_OLLAMA_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class DownloadState(enum.StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ModelStatus:
    """Состояние одной модели кольца в терминах «можно ли ей отвечать»."""

    family: str
    model: str
    installed: bool
    download_size_gb: float
    min_ram_gb: int


@dataclass(slots=True)
class ProvisioningPlan:
    """Что именно предстоит скачать и хватит ли под это места."""

    profile: HardwareProfile
    models: list[ModelStatus]
    free_disk_gb: float

    @property
    def missing(self) -> list[ModelStatus]:
        return [m for m in self.models if not m.installed]

    @property
    def required_disk_gb(self) -> float:
        return round(sum(m.download_size_gb for m in self.missing), 1)

    @property
    def enough_disk_space(self) -> bool:
        return self.free_disk_gb >= self.required_disk_gb

    @property
    def ready(self) -> bool:
        """Готовность = хотя бы одна модель кольца установлена.

        Кольцо переживает отсутствие части моделей — оно просто обойдёт
        недоступные. Полная неготовность — это когда нет ни одной.
        """
        return any(m.installed for m in self.models)


@dataclass(slots=True)
class DownloadProgress:
    model: str
    state: DownloadState = DownloadState.PENDING
    completed_bytes: int = 0
    total_bytes: int = 0
    status: str | None = None
    error: str | None = None
    # Флаг проверяется между строками потока: отмена не убивает запрос
    # силой, а даёт загрузке остановиться на ближайшей границе.
    cancel_requested: bool = False
    stream_active: bool = False
    digests: set[str] = field(default_factory=set)

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return round(self.completed_bytes / self.total_bytes * 100, 1)


@dataclass(slots=True)
class ModelRemoval:
    """Результат удаления весов.

    deleted=False — модель не была зарегистрирована в runtime: либо её уже
    удалили, либо загрузку прервали на середине и до регистрации не дошло.
    """

    model: str
    deleted: bool
    partials_deleted: int = 0


class ModelProvisioner:
    """Проверяет наличие моделей профиля и скачивает недостающие.

    Прогресс держится в памяти процесса: перевод длительных загрузок в
    Celery — задача Stage 4, здесь важнее сам явный контракт «показать цену →
    получить согласие → скачать».
    """

    def __init__(
        self,
        provider: LocalLLMProvider,
        base_url: str | None = None,
        timeout_s: float = 3600.0,
        blobs_path: Path | None = None,
    ) -> None:
        self._provider = provider
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._timeout_s = timeout_s
        self._blobs_path = (blobs_path or settings.ollama_blobs_path).expanduser()
        self._downloads: dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()

    def build_plan(
        self,
        profile: HardwareProfile,
        free_disk_gb: float,
        entries: list[ModelRingEntry] | None = None,
    ) -> ProvisioningPlan:
        """План загрузки для кольца.

        По умолчанию — кольцо профиля; entries передаётся, когда состав
        кольца переопределён пользователем и план должен считаться по
        фактическим моделям, а не по рекомендованным.
        """
        ring = entries if entries is not None else get_profile_definition(profile).ring
        installed = {info.name for info in self._provider.list_models()}
        installed_families = {name.split(":")[0] for name in installed}

        models = [
            ModelStatus(
                family=entry.family,
                model=entry.model,
                # Точное совпадение с тегом либо наличие семейства: тег
                # мог отличаться при ручной установке пользователем.
                installed=entry.model in installed
                or entry.model.split(":")[0] in installed_families,
                download_size_gb=entry.download_size_gb,
                min_ram_gb=entry.min_ram_gb,
            )
            for entry in ring
        ]
        return ProvisioningPlan(
            profile=profile, models=models, free_disk_gb=round(free_disk_gb, 1)
        )

    def progress(self, model: str) -> DownloadProgress:
        with self._lock:
            progress = self._downloads.get(model)
        if progress is None:
            raise NotFoundError(f"Загрузка модели '{model}' не запускалась")
        return progress

    def all_progress(self) -> list[DownloadProgress]:
        with self._lock:
            return list(self._downloads.values())

    def cancel_download(self, model: str) -> DownloadProgress:
        """Просит остановить загрузку. Идемпотентно: повтор ничего не ломает.

        Отменяется ожидание в интерфейсе, но поток Ollama остаётся открытым и
        дочитывается фоновым заданием. Закрытие потока здесь обрывало pull и
        оставляло `*-partial` blobs, которые `/api/delete` не видит.
        """
        progress = self.progress(model)
        with self._lock:
            if progress.state in (DownloadState.PENDING, DownloadState.DOWNLOADING):
                progress.cancel_requested = True
                if progress.state == DownloadState.PENDING:
                    # Фоновая задача ещё не стартовала — останавливаем сразу,
                    # чтобы состояние не висело в «в очереди» до её запуска.
                    progress.state = DownloadState.CANCELLED
                logger.info("model_download_cancel_requested", extra={"model": model})
        return progress

    def delete_model(self, model: str) -> ModelRemoval:
        """Удаляет веса модели из runtime по явному запросу пользователя.

        Для незарегистрированной модели удаляет только partial-файлы digest'ов,
        которые были получены из потока именно этой загрузки.
        """
        try:
            response = httpx.request(
                "DELETE",
                f"{self._base_url}/api/delete",
                json={"model": model},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            raise InferenceError(
                f"Не удалось удалить модель '{model}': {exc}"
            ) from exc

        deleted = response.status_code != 404
        if deleted:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise InferenceError(
                    f"Не удалось удалить модель '{model}': {exc}"
                ) from exc

        with self._lock:
            progress = self._downloads.get(model)
            digests = set(progress.digests) if progress else set()
            if progress:
                progress.cancel_requested = True

        partials_deleted = self._delete_partial_blobs(digests) if not deleted else 0

        with self._lock:
            self._downloads.pop(model, None)

        logger.info(
            "model_deleted",
            extra={
                "model": model,
                "registered": deleted,
                "partials_deleted": partials_deleted,
            },
        )
        return ModelRemoval(
            model=model,
            deleted=deleted,
            partials_deleted=partials_deleted,
        )

    def _delete_partial_blobs(self, digests: set[str]) -> int:
        """Удаляет только незавершённые файлы известных digest'ов Ollama."""
        deleted = 0
        for digest in digests:
            if not _OLLAMA_DIGEST.fullmatch(digest):
                continue
            prefix = digest.replace(":", "-") + "-partial"
            for path in self._blobs_path.glob(prefix + "*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
                    deleted += 1
        return deleted

    def start_download(
        self,
        model: str,
        profile: HardwareProfile,
        allowed: set[str] | None = None,
    ) -> DownloadProgress:
        """Регистрирует загрузку. Вызывается только по явному запросу.

        Скачать можно лишь модель кольца: allowed передаётся, когда состав
        переопределён пользователем, иначе кольцо берётся из профиля. Так
        эндпоинт остаётся закрытым списком, а не произвольным `pull` в
        реестр моделей по имени из запроса.
        """
        known = allowed or {entry.model for entry in get_profile_definition(profile).ring}
        if model not in known:
            raise ValidationError(
                f"Модель '{model}' не входит в кольцо профиля {profile}. "
                f"Доступны: {', '.join(sorted(known))}"
            )

        with self._lock:
            existing = self._downloads.get(model)
            if existing and existing.state in (
                DownloadState.PENDING,
                DownloadState.DOWNLOADING,
            ):
                # Повторный запрос не плодит вторую загрузку тех же весов.
                return existing
            progress = DownloadProgress(model=model, state=DownloadState.PENDING)
            self._downloads[model] = progress
        return progress

    def run_download(self, model: str) -> DownloadProgress:
        """Тянет веса из реестра runtime'а, обновляя прогресс по ходу."""
        progress = self.progress(model)
        if progress.cancel_requested:
            progress.state = DownloadState.CANCELLED
            return progress
        progress.state = DownloadState.DOWNLOADING
        progress.stream_active = True
        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json={"model": model, "stream": True},
                timeout=self._timeout_s,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if progress.cancel_requested:
                        progress.stream_active = False
                        progress.state = DownloadState.CANCELLED
                        logger.info(
                            "model_download_cancelled", extra={"model": model}
                        )
                        return progress
                    if line:
                        _apply_progress_line(progress, line)
        except httpx.HTTPError as exc:
            progress.stream_active = False
            progress.state = DownloadState.FAILED
            progress.error = str(exc)
            logger.warning(
                "model_download_failed", extra={"model": model, "error": str(exc)}
            )
            raise InferenceError(
                f"Не удалось загрузить модель '{model}': {exc}"
            ) from exc

        progress.stream_active = False
        progress.state = DownloadState.COMPLETED
        if progress.total_bytes:
            progress.completed_bytes = progress.total_bytes
        logger.info("model_download_completed", extra={"model": model})
        return progress

def _apply_progress_line(progress: DownloadProgress, line: str) -> None:
    """Разбирает строку NDJSON-потока загрузки.

    Битую строку молча пропускаем: поток прогресса не должен ронять
    загрузку, которая на самом деле идёт нормально.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return

    if error := payload.get("error"):
        progress.state = DownloadState.FAILED
        progress.error = str(error)
        return

    progress.status = payload.get("status") or progress.status
    if (digest := payload.get("digest")) and _OLLAMA_DIGEST.fullmatch(str(digest)):
        progress.digests.add(str(digest))
    if (total := payload.get("total")) is not None:
        progress.total_bytes = int(total)
    if (completed := payload.get("completed")) is not None:
        progress.completed_bytes = int(completed)
