"""Локальный reranker: сужает 20-30 кандидатов до 5-10 лучших чанков.

Независим от embedding-модели, LLM и Retriever — принимает на вход только
текст запроса и тексты кандидатов, поэтому его можно включать/выключать
и сравнивать в evaluation без изменения остального pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.errors import InferenceError
from app.rag.vector_store import RetrievedChunk


@dataclass(slots=True)
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 8
    ) -> list[RerankedChunk]: ...

    @abstractmethod
    def health_check(self) -> bool: ...


class CrossEncoderReranker(Reranker):
    """Локальный cross-encoder через sentence-transformers.

    Модель грузится лениво и один раз на процесс: инициализация cross-encoder
    заметно дороже, чем сам forward pass на паре запрос-кандидат.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise InferenceError(
                    "sentence-transformers не установлен: локальный reranker недоступен"
                ) from exc
            try:
                self._model = CrossEncoder(self._model_name)
            except Exception as exc:  # noqa: BLE001 - любая ошибка загрузки модели
                raise InferenceError(
                    f"Не удалось загрузить локальный reranker '{self._model_name}': {exc}"
                ) from exc
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 8
    ) -> list[RerankedChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, candidate.text) for candidate in candidates]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"Reranker '{self._model_name}' упал при инференсе: {exc}"
            ) from exc

        ranked = sorted(
            (
                RerankedChunk(chunk=chunk, rerank_score=float(score))
                for chunk, score in zip(candidates, scores, strict=True)
            ),
            key=lambda item: item.rerank_score,
            reverse=True,
        )
        return ranked[:top_k]

    def health_check(self) -> bool:
        try:
            self._load()
        except InferenceError:
            return False
        return True


class NoOpReranker(Reranker):
    """Reranker выключен: сохраняет порядок и скор retrieval как есть.

    Позволяет включать/выключать reranking флагом конфигурации без ветвлений
    в вызывающем коде — retriever и generation всегда работают через один
    и тот же интерфейс Reranker.
    """

    name = "noop"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int = 8
    ) -> list[RerankedChunk]:
        return [
            RerankedChunk(chunk=c, rerank_score=c.score) for c in candidates[:top_k]
        ]

    def health_check(self) -> bool:
        return True
