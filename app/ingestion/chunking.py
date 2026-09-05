"""Стратегии чанкинга.

Реализованы самостоятельно и взаимозаменяемы: выбор стратегии и её параметров
напрямую влияет на качество retrieval, поэтому их сравнение — часть evaluation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.errors import ValidationError
from app.ingestion.models import Chunk, ParsedBlock, ParsedDocument

_WORD = re.compile(r"\S+")
# Границы предложений: точка/!/?/… с последующим пробелом и заглавной буквой.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[]?[A-ZА-ЯЁ0-9])")


def count_tokens(text: str) -> int:
    """Приближение числа токенов по числу слов.

    Точный токенизатор зависит от модели; для бюджета контекста и размеров
    чанков достаточно устойчивой оценки, одинаковой для всех стратегий.
    """
    return len(_WORD.findall(text))


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


class ChunkingStrategy(ABC):
    """Общий контракт: документ на входе, готовые к индексации чанки на выходе."""

    name: str

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60) -> None:
        if chunk_size <= 0:
            raise ValidationError("chunk_size должен быть положительным")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValidationError("chunk_overlap должен быть в диапазоне [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, document: ParsedDocument) -> list[Chunk]: ...

    @staticmethod
    def _chunk(index: int, words: list[str], block: ParsedBlock) -> Chunk:
        text = " ".join(words)
        return Chunk(
            index=index,
            text=text,
            page=block.page,
            section=block.section,
            token_count=len(words),
        )


class FixedTokenChunking(ChunkingStrategy):
    """Скользящее окно фиксированного размера с перекрытием.

    Структуру документа игнорирует: базовая стратегия для сравнения с остальными.
    """

    name = "fixed"

    def split(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        for block in document.blocks:
            words = _WORD.findall(block.text)
            for start in range(0, max(len(words), 1), step):
                window = words[start : start + self.chunk_size]
                if not window:
                    break
                chunks.append(self._chunk(len(chunks), window, block))
                if start + self.chunk_size >= len(words):
                    break
        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Structure-aware стратегия.

    Режет по естественным границам — блок, затем предложения — и склеивает
    короткие фрагменты до целевого размера, не разрывая предложение посередине.
    Перекрытие добирается последними предложениями предыдущего чанка.
    """

    name = "recursive"

    def split(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for block in document.blocks:
            for words in self._split_block(block.text):
                chunks.append(self._chunk(len(chunks), words, block))
        return chunks

    def _split_block(self, text: str) -> list[list[str]]:
        result: list[list[str]] = []
        current: list[str] = []

        for sentence in split_sentences(text) or [text]:
            words = _WORD.findall(sentence)
            if not words:
                continue
            if len(words) > self.chunk_size:
                # Предложение длиннее окна — режем его фиксированными кусками.
                if current:
                    result.append(current)
                    current = []
                for start in range(0, len(words), self.chunk_size):
                    result.append(words[start : start + self.chunk_size])
                continue
            if current and len(current) + len(words) > self.chunk_size:
                result.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap else []
                current = [*overlap, *words]
            else:
                current = [*current, *words]
        if current:
            result.append(current)
        return result


_STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    FixedTokenChunking.name: FixedTokenChunking,
    RecursiveChunking.name: RecursiveChunking,
}


def get_chunking_strategy(
    name: str, chunk_size: int = 400, chunk_overlap: int = 60
) -> ChunkingStrategy:
    strategy = _STRATEGIES.get(name)
    if strategy is None:
        raise ValidationError(
            f"Неизвестная стратегия чанкинга '{name}'. "
            f"Доступны: {', '.join(sorted(_STRATEGIES))}"
        )
    return strategy(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
