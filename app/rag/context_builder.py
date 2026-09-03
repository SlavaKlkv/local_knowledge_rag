"""ContextBuilder: сборка контекста для LLM из ранжированных чанков.

Отвечает за бюджет токенов, дедупликацию и сохранение метаданных источника —
без них ответ невозможно проверить по citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.chunking import count_tokens
from app.rag.vector_store import RetrievedChunk


@dataclass(slots=True)
class ContextItem:
    """Чанк, попавший в контекст, с номером ссылки для цитирования."""

    ref: int
    chunk: RetrievedChunk


@dataclass(slots=True)
class BuiltContext:
    items: list[ContextItem]
    text: str
    token_count: int

    @property
    def is_empty(self) -> bool:
        return not self.items


_WHITESPACE = re.compile(r"\s+")


class ContextBuilder:
    def __init__(self, token_budget: int = 2000, max_chunks: int = 8) -> None:
        self.token_budget = token_budget
        self.max_chunks = max_chunks

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        items: list[ContextItem] = []
        seen: set[str] = set()
        used_tokens = 0

        for chunk in chunks:
            if len(items) >= self.max_chunks:
                break
            fingerprint = _fingerprint(chunk.text)
            if not fingerprint or fingerprint in seen:
                # Перекрытие чанков и повторная индексация дают дубликаты,
                # которые впустую расходуют бюджет контекста.
                continue
            tokens = count_tokens(chunk.text)
            if used_tokens + tokens > self.token_budget and items:
                continue
            seen.add(fingerprint)
            used_tokens += tokens
            items.append(ContextItem(ref=len(items) + 1, chunk=chunk))

        return BuiltContext(
            items=items, text=self._render(items), token_count=used_tokens
        )

    @staticmethod
    def _render(items: list[ContextItem]) -> str:
        blocks = []
        for item in items:
            chunk = item.chunk
            location = []
            if chunk.document_name:
                location.append(chunk.document_name)
            if chunk.page is not None:
                location.append(f"стр. {chunk.page}")
            if chunk.section:
                location.append(chunk.section)
            header = f"[{item.ref}] " + (" | ".join(location) if location else "источник")
            blocks.append(f"{header}\n{chunk.text}")
        return "\n\n".join(blocks)


def _fingerprint(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()
