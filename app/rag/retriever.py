"""Retriever: превращает вопрос пользователя в набор релевантных чанков.

V1 — dense semantic retrieval. Слой намеренно отделён от векторного хранилища
и от embeddings, чтобы hybrid retrieval и reranking добавлялись поверх
без изменения вызывающего кода.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import ValidationError
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import QdrantVectorStore, RetrievedChunk


@dataclass(slots=True)
class RetrievalQuery:
    text: str
    knowledge_base_id: str
    top_k: int = 10
    score_threshold: float | None = None
    document_ids: list[str] | None = None


class DenseRetriever:
    def __init__(
        self, embeddings: EmbeddingProvider, vector_store: QdrantVectorStore
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        if not query.text.strip():
            raise ValidationError("Пустой поисковый запрос")
        if query.top_k <= 0:
            raise ValidationError("top_k должен быть положительным")

        vector = self._embeddings.embed_query(query.text)
        # Изоляция базы знаний применяется в самом Qdrant-фильтре, а не после
        # выдачи: иначе top_k заполнялся бы чужими документами.
        return self._vector_store.search(
            vector=vector,
            knowledge_base_id=query.knowledge_base_id,
            top_k=query.top_k,
            score_threshold=query.score_threshold,
            document_ids=query.document_ids,
        )
