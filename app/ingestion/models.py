"""Промежуточные структуры конвейера ingestion."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedBlock:
    """Фрагмент исходного документа с сохранённой структурной привязкой."""

    text: str
    page: int | None = None
    section: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    blocks: list[ParsedBlock]
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


@dataclass(slots=True)
class Chunk:
    """Единица индексации: текст плюс метаданные для цитирования."""

    index: int
    text: str
    page: int | None = None
    section: str | None = None
    token_count: int = 0
