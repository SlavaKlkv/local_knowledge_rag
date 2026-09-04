"""Retriever: превращает вопрос пользователя в набор релевантных чанков.

Dense semantic retrieval ловит смысловую близость, но плохо — точные
лексические совпадения (номера статей, коды). Sparse retrieval — наоборот.
HybridRetriever объединяет оба через Reciprocal Rank Fusion (RRF), сохраняя
общий интерфейс Retriever, поэтому вызывающий код (ContextBuilder, /chat)
не меняется от выбора стратегии.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.errors import ValidationError
from app.rag.embeddings import EmbeddingProvider
from app.rag.sparse import HashedSparseVectorizer
from app.rag.vector_store import QdrantVectorStore, RetrievedChunk


@dataclass(slots=True)
class RetrievalQuery:
    text: str
    knowledge_base_id: str
    top_k: int = 10
    score_threshold: float | None = None
    document_ids: list[str] | None = None


class Retriever(Protocol):
    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


def _validate(query: RetrievalQuery) -> None:
    if not query.text.strip():
        raise ValidationError("Пустой поисковый запрос")
    if query.top_k <= 0:
        raise ValidationError("top_k должен быть положительным")


class DenseRetriever:
    def __init__(
        self, embeddings: EmbeddingProvider, vector_store: QdrantVectorStore
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        _validate(query)
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


class SparseRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        vectorizer: HashedSparseVectorizer | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._vectorizer = vectorizer or HashedSparseVectorizer()

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        _validate(query)
        sparse_vector = self._vectorizer.vectorize(query.text)
        return self._vector_store.sparse_search(
            sparse_vector=sparse_vector,
            knowledge_base_id=query.knowledge_base_id,
            top_k=query.top_k,
            document_ids=query.document_ids,
        )


class HybridRetriever:
    """Dense + sparse retrieval, объединённые Reciprocal Rank Fusion.

    RRF не требует сопоставимости шкал скора dense/sparse (косинус против
    произвольного лексического веса) — ранг в каждом списке значит больше,
    чем абсолютное значение скора, что и делает фьюжн устойчивым.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        _validate(query)
        # Каждый список берётся с запасом: после фьюжна порядок может
        # измениться, и не должно оказаться, что в top_k текущего списка
        # не попал чанк, который в объединении был бы лучшим.
        candidate_k = max(query.top_k * 3, query.top_k)
        # Порог применяется только к dense-ноге: он задан в шкале косинусной
        # близости, тогда как лексический вес sparse и итоговый скор RRF
        # живут в других шкалах, и один и тот же порог значил бы там другое.
        dense_hits = self._dense.retrieve(
            RetrievalQuery(
                text=query.text,
                knowledge_base_id=query.knowledge_base_id,
                top_k=candidate_k,
                score_threshold=query.score_threshold,
                document_ids=query.document_ids,
            )
        )
        sparse_hits = self._sparse.retrieve(
            RetrievalQuery(
                text=query.text,
                knowledge_base_id=query.knowledge_base_id,
                top_k=candidate_k,
                document_ids=query.document_ids,
            )
        )

        fused = self._fuse(dense_hits, sparse_hits)
        return fused[: query.top_k]

    def _fuse(
        self, dense_hits: list[RetrievedChunk], sparse_hits: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, chunk in enumerate(hits, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                    self._rrf_k + rank
                )
                chunks.setdefault(chunk.chunk_id, chunk)

        ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunks[chunk_id].chunk_id,
                document_id=chunks[chunk_id].document_id,
                document_name=chunks[chunk_id].document_name,
                text=chunks[chunk_id].text,
                score=scores[chunk_id],
                chunk_index=chunks[chunk_id].chunk_index,
                page=chunks[chunk_id].page,
                section=chunks[chunk_id].section,
            )
            for chunk_id in ranked_ids
        ]
