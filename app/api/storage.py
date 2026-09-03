"""Хранилище загруженных файлов.

Имя файла из запроса не используется как путь: оно приходит от пользователя и
может содержать переходы по каталогам. На диск файл кладётся под собственным
идентификатором, исходное имя остаётся только в метаданных.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from app.core.errors import ValidationError

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
_UNSAFE = re.compile(r"[^\w.\- ]", re.UNICODE)


def safe_filename(filename: str) -> str:
    name = Path(filename or "").name
    name = _UNSAFE.sub("_", name).strip(". ")
    if not name:
        raise ValidationError("Некорректное имя файла")
    return name[:255]


def validate_upload(filename: str, size_bytes: int, allowed: tuple[str, ...]) -> str:
    """Проверяет расширение и размер, возвращает безопасное имя."""
    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise ValidationError(
            f"Формат {suffix or '<без расширения>'} не поддерживается. "
            f"Доступны: {', '.join(allowed)}"
        )
    if size_bytes <= 0:
        raise ValidationError("Пустой файл")
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"Файл больше допустимых {MAX_FILE_SIZE_BYTES // (1024 * 1024)} МБ"
        )
    return name


class DocumentStorage:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, content: bytes, original_name: str) -> tuple[Path, str]:
        """Кладёт файл на диск, возвращает путь и sha256."""
        suffix = Path(safe_filename(original_name)).suffix.lower()
        path = self._root / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(content)
        return path, hashlib.sha256(content).hexdigest()

    def delete(self, path: Path) -> None:
        path.unlink(missing_ok=True)
