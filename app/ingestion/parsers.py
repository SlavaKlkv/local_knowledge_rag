"""Парсеры документов: из файла в набор блоков с привязкой к странице/секции."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from app.core.errors import ValidationError
from app.ingestion.models import ParsedBlock, ParsedDocument
from app.ingestion.normalization import normalize_text


class DocumentParser(Protocol):
    """Единый контракт парсера: подмена формата не влияет на остальной pipeline."""

    extensions: tuple[str, ...]

    def parse(self, path: Path) -> ParsedDocument: ...


class TextParser:
    extensions = (".txt",)

    def parse(self, path: Path) -> ParsedDocument:
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        blocks = [
            ParsedBlock(text=paragraph)
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]
        return ParsedDocument(blocks=blocks, meta={"format": "txt"})


class MarkdownParser:
    """Markdown: заголовки становятся секциями, чтобы citations были адресными."""

    extensions = (".md", ".markdown")
    _HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

    def parse(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks: list[ParsedBlock] = []
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                text = normalize_text("\n".join(buffer))
                if text:
                    blocks.append(ParsedBlock(text=text, section=section))
                buffer.clear()

        for line in raw.replace("\r\n", "\n").split("\n"):
            heading = self._HEADING.match(line)
            if heading:
                flush()
                section = heading.group(2).strip()
                continue
            if not line.strip():
                flush()
                continue
            buffer.append(line)
        flush()
        return ParsedDocument(blocks=blocks, meta={"format": "markdown"})


class PdfParser:
    """PDF: одна страница — один блок, номер страницы сохраняется для цитат."""

    extensions = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        blocks: list[ParsedBlock] = []
        for number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                blocks.append(ParsedBlock(text=text, page=number))
        return ParsedDocument(
            blocks=blocks, meta={"format": "pdf", "pages": str(len(reader.pages))}
        )


class ParserRegistry:
    """Выбор парсера по расширению файла."""

    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        for parser in parsers if parsers is not None else default_parsers():
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        for extension in parser.extensions:
            self._by_extension[extension] = parser

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_extension))

    def parse(self, path: Path) -> ParsedDocument:
        parser = self._by_extension.get(path.suffix.lower())
        if parser is None:
            raise ValidationError(
                f"Формат {path.suffix or '<без расширения>'} не поддерживается. "
                f"Доступны: {', '.join(self.supported_extensions)}"
            )
        return parser.parse(path)


def default_parsers() -> list[DocumentParser]:
    """V1 поддерживает PDF и TXT; Markdown добавлен как бесплатный случай."""
    return [TextParser(), MarkdownParser(), PdfParser()]
