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
import threading
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.core.errors import InferenceError, NotFoundError, ValidationError
from app.hardware.profiles import HardwareProfile, get_profile_definition
from app.llm.base import LocalLLMProvider

logger = logging.getLogger("rag.provisioning")


class DownloadState(enum.StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


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

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return round(self.completed_bytes / self.total_bytes * 100, 1)


class ModelProvisioner:
    """Проверяет наличие моделей профиля и скачивает недостающие.

    Прогресс держится в памяти процесса: перевод длительных загрузок в
    Celery — задача V4, здесь важнее сам явный контракт «показать цену →
    получить согласие → скачать».
    """

    def __init__(
        self,
        provider: LocalLLMProvider,
        base_url: str | None = None,
        timeout_s: float = 3600.0,
    ) -> None:
        self._provider = provider
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._timeout_s = timeout_s
        self._downloads: dict[str, DownloadProgress] = {}
        self._lock = threading.Lock()

    def build_plan(self, profile: HardwareProfile, free_disk_gb: float) -> ProvisioningPlan:
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
            for entry in get_profile_definition(profile).ring
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

    def start_download(self, model: str, profile: HardwareProfile) -> DownloadProgress:
        """Регистрирует загрузку. Вызывается только по явному запросу."""
        known = {entry.model for entry in get_profile_definition(profile).ring}
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
        progress.state = DownloadState.DOWNLOADING
        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json={"model": model, "stream": True},
                timeout=self._timeout_s,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        _apply_progress_line(progress, line)
        except httpx.HTTPError as exc:
            progress.state = DownloadState.FAILED
            progress.error = str(exc)
            logger.warning(
                "model_download_failed", extra={"model": model, "error": str(exc)}
            )
            raise InferenceError(
                f"Не удалось загрузить модель '{model}': {exc}"
            ) from exc

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
    if (total := payload.get("total")) is not None:
        progress.total_bytes = int(total)
    if (completed := payload.get("completed")) is not None:
        progress.completed_bytes = int(completed)
