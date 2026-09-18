"""Пользовательский состав и порядок кольца моделей.

Профиль железа задаёт кольцо по умолчанию, но выбор моделей — решение
пользователя: на конкретной машине одно семейство может отвечать заметно
хуже других, а какие-то веса просто не хочется держать на диске. Поэтому
переопределение хранится отдельно от профиля и рядом с данными приложения,
а не в переменных окружения: его меняют из интерфейса, без перезапуска.

Хранится список имён моделей — порядок в нём и есть порядок обхода кольца.
Пустого состава не бывает: сброс возвращает кольцо профиля, а не оставляет
приложение без моделей.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.hardware.profiles import ModelRingEntry, known_ring_entries

logger = logging.getLogger("rag.ring_config")


class RingConfigStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[str] | None:
        """Сохранённый состав кольца либо None, если его не переопределяли."""
        if not self._path.is_file():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            models = [str(model) for model in payload["models"]]
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            # Битый файл не должен ронять генерацию: кольцо просто вернётся
            # к составу профиля, а расхождение видно в логах.
            logger.warning("ring_config_unreadable", extra={"path": str(self._path)})
            return None
        return models or None

    def save(self, models: list[str]) -> list[str]:
        normalized = self._validate(models)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"models": normalized}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return normalized

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)

    def resolve(
        self,
        profile_ring: list[ModelRingEntry],
        installed: set[str] | None = None,
    ) -> list[ModelRingEntry]:
        """Состав кольца с учётом переопределения.

        Данные о размере весов и требованиях к памяти берутся из каталога
        профилей; для модели, которой там нет (пользователь добавил свою),
        они неизвестны — тогда 0, а семейство выводится из имени.

        Кольцо — это модели, которые реально могут ответить, поэтому
        отсутствующие в runtime'е из него выпадают сами: держать такую
        значило бы тратить на неё попытку обхода при каждом запросе.
        Файл состава при этом не трогаем — модель, установленную заново,
        кольцо примет обратно на её прежнее место. Если не установлено
        вообще ничего, возвращаем полный состав: пустое кольцо непригодно,
        а список моделей профиля хотя бы показывает, что предстоит скачать.
        """
        models = self.load()
        if models is None:
            return _installed_first(profile_ring, installed)
        # Сохранённый состав мог попасть сюда до того, как embedding-модели
        # перестали приниматься: отбрасываем их и здесь, а если после этого
        # кольцо опустело — возвращаемся к профилю, а не к пустому кольцу.
        models = [model for model in models if not is_embedding_model(model)]
        if installed is not None:
            models = [model for model in models if model in installed]
        if not models:
            return list(profile_ring)
        catalog = known_ring_entries()
        return [catalog.get(model) or _foreign_entry(model) for model in models]




    def _validate(self, models: list[str]) -> list[str]:
        normalized = [model.strip() for model in models if model and model.strip()]
        if not normalized:
            raise ValidationError("В кольце должна остаться хотя бы одна модель")
        if len(set(normalized)) != len(normalized):
            raise ValidationError("Модель не может входить в кольцо дважды")
        return normalized


# Кольцо — это модели генерации. Embedding-модели тоже стоят в runtime и
# попадают в его список, но текста не порождают: в кольце такая модель
# отвечала бы отказом на каждой попытке. Имя модели — единственное, что о
# ней известно до вызова, поэтому отбор идёт по общепринятым маркерам.
_EMBEDDING_MARKERS = (
    "embed",
    "bge-",
    "gte-",
    "e5-",
    "all-minilm",
)


def is_embedding_model(model: str) -> bool:
    name = model.split(":", 1)[0].lower()
    if name == get_settings().embedding_model.split(":", 1)[0].lower():
        return True
    return any(marker in name for marker in _EMBEDDING_MARKERS)


def _installed_first(
    ring: list[ModelRingEntry], installed: set[str] | None
) -> list[ModelRingEntry]:
    """Состав профиля без моделей, которых нет в runtime'е.

    На чистой машине не установлено ещё ничего — тогда возвращаем состав
    целиком: он читается как план загрузки, а не как рабочее кольцо.
    """
    if installed is None:
        return list(ring)
    present = [entry for entry in ring if entry.model in installed]
    return present or list(ring)


def _foreign_entry(model: str) -> ModelRingEntry:
    family = model.split(":", 1)[0]
    return ModelRingEntry(family=family, model=model)
