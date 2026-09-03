import pytest

from app.core.errors import ValidationError
from app.ingestion.parsers import ParserRegistry


@pytest.fixture
def registry() -> ParserRegistry:
    return ParserRegistry()


def test_parses_plain_text_into_paragraph_blocks(tmp_path, registry):
    path = tmp_path / "note.txt"
    path.write_text("первый абзац\n\nвторой абзац", encoding="utf-8")

    document = registry.parse(path)

    assert [block.text for block in document.blocks] == ["первый абзац", "второй абзац"]


def test_markdown_headings_become_sections(tmp_path, registry):
    path = tmp_path / "policy.md"
    path.write_text("# Отпуска\n\nПравила отпуска.\n\n## Перенос\n\nПравила переноса.",
                    encoding="utf-8")

    blocks = registry.parse(path).blocks

    assert [(b.section, b.text) for b in blocks] == [
        ("Отпуска", "Правила отпуска."),
        ("Перенос", "Правила переноса."),
    ]


def test_unsupported_format_is_rejected(tmp_path, registry):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK")

    with pytest.raises(ValidationError, match="не поддерживается"):
        registry.parse(path)


def test_pdf_pages_are_preserved_for_citations(tmp_path, registry):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "empty.pdf"
    with path.open("wb") as handle:
        writer.write(handle)

    document = registry.parse(path)

    assert document.meta["format"] == "pdf"
    assert document.meta["pages"] == "1"
