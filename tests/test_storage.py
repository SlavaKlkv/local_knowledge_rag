import pytest

from app.api.storage import (
    MAX_FILE_SIZE_BYTES,
    DocumentStorage,
    safe_filename,
    validate_upload,
)
from app.core.errors import ValidationError

ALLOWED = (".pdf", ".txt", ".md")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("отчёт 2026.pdf", "отчёт 2026.pdf"),
        ("bad;name|rm -rf.txt", "bad_name_rm -rf.txt"),
        ("/absolute/path/doc.md", "doc.md"),
    ],
)
def test_filenames_are_sanitized(raw, expected):
    assert safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "...", "/"])
def test_empty_filename_is_rejected(raw):
    with pytest.raises(ValidationError):
        safe_filename(raw)


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValidationError, match="не поддерживается"):
        validate_upload("archive.zip", 100, ALLOWED)


def test_oversized_file_is_rejected():
    with pytest.raises(ValidationError, match="больше допустимых"):
        validate_upload("doc.pdf", MAX_FILE_SIZE_BYTES + 1, ALLOWED)


def test_empty_file_is_rejected():
    with pytest.raises(ValidationError, match="Пустой файл"):
        validate_upload("doc.pdf", 0, ALLOWED)


def test_stored_file_does_not_reuse_user_supplied_path(tmp_path):
    storage = DocumentStorage(tmp_path)

    path, checksum = storage.save(b"content", "../../evil.txt")

    assert path.parent == tmp_path
    assert path.suffix == ".txt"
    assert path.name != "evil.txt"
    assert len(checksum) == 64


def test_identical_content_has_identical_checksum(tmp_path):
    storage = DocumentStorage(tmp_path)

    _, first = storage.save(b"same", "a.txt")
    _, second = storage.save(b"same", "b.txt")

    assert first == second
