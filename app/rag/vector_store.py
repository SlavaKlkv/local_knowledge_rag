"""Работа с Qdrant: индексация чанков и поиск по векторам.

Qdrant хранит вектор и payload, достаточный для фильтрации и цитирования,
без обращения к PostgreSQL на горячем пути поиска. Каждая точка несёт два
именованных вектора — "dense" (embedding-модель) и "sparse" (лексический,
см. app/rag/sparse.py) — это то, что делает возможным hybrid retrieval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.rag.sparse import SparseVector

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


@dataclass(slots=True)
class ChunkPoint:
    """Чанк, готовый к записи в коллекцию."""

    chunk_id: str
    document_id: str
    knowledge_base_id: str
    chunk_index: int
    text: str
    vector: list[float]
    sparse_vector: SparseVector = field(default_factory=lambda: SparseVector([], []))
    version: int = 1
    page: int | None = None
    section: str | None = None
    document_name: str | None = None
    embedding_version: str | None = None


@dataclass(slots=True)
class RetrievedChunk:
    """Результат поиска с метаданными, нужными для citations."""

    chunk_id: str
    document_id: str
    document_name: str | None
    text: str
    score: float
    chunk_index: int
    page: int | None = None
    section: str | None = None


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client or QdrantClient(url=settings.qdrant_url)
        self._collection = collection or settings.qdrant_collection
        self._dimension = dimension or settings.embedding_dim

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Создаёт коллекцию и индексы под фильтрацию по payload.

        Без индексов фильтры по knowledge_base_id работают полным перебором,
        а изоляция баз знаний применяется к каждому запросу.
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    _DENSE_VECTOR_NAME: qm.VectorParams(
                        size=self._dimension, distance=qm.Distance.COSINE
                    ),
                },
                sparse_vectors_config={_SPARSE_VECTOR_NAME: qm.SparseVectorParams()},
            )
        for field_name in ("knowledge_base_id", "document_id"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=qm.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def upsert_chunks(self, chunks: list[ChunkPoint]) -> int:
        if not chunks:
            return 0
        points = [
            qm.PointStruct(
                id=chunk.chunk_id,
                vector={
                    _DENSE_VECTOR_NAME: chunk.vector,
                    _SPARSE_VECTOR_NAME: qm.SparseVector(
                        indices=chunk.sparse_vector.indices,
                        values=chunk.sparse_vector.values,
                    ),
                },
                payload={
                    "document_id": chunk.document_id,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "document_name": chunk.document_name,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "page": chunk.page,
                    "section": chunk.section,
                    "version": chunk.version,
                    "embedding_version": chunk.embedding_version,
                },
            )
            for chunk in chunks
        ]
        self._client.upsert(self._collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: list[float],
        knowledge_base_id: str,
        top_k: int = 10,
        score_threshold: float | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=_DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=_build_filter(knowledge_base_id, document_ids),
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [_to_retrieved(point) for point in response.points]

    def sparse_search(
        self,
        sparse_vector: SparseVector,
        knowledge_base_id: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not sparse_vector.indices:
            # Пустой sparse-вектор (например, вопрос из одних стоп-слов
            # в хешированном пространстве) не даёт Qdrant искать что-либо.
            return []
        response = self._client.query_points(
            collection_name=self._collection,
            query=qm.SparseVector(
                indices=sparse_vector.indices, values=sparse_vector.values
            ),
            using=_SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=_build_filter(knowledge_base_id, document_ids),
            with_payload=True,
        )
        return [_to_retrieved(point) for point in response.points]

    def delete_document(self, document_id: str, version: int | None = None) -> None:
        """Удаляет векторы документа.

        При переиндексации вызывается со старой версией: иначе в коллекции
        останутся stale-векторы, и ответы будут ссылаться на неактуальный текст.
        """
        must: list[qm.Condition] = [
            qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))
        ]
        if version is not None:
            must.append(
                qm.FieldCondition(key="version", match=qm.MatchValue(value=version))
            )
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=must)),
            wait=True,
        )

    def count(self, knowledge_base_id: str | None = None) -> int:
        query_filter = None
        if knowledge_base_id:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="knowledge_base_id",
                        match=qm.MatchValue(value=knowledge_base_id),
                    )
                ]
            )
        return self._client.count(
            self._collection, count_filter=query_filter, exact=True
        ).count


def _build_filter(
    knowledge_base_id: str, document_ids: list[str] | None
) -> qm.Filter:
    must: list[qm.Condition] = [
        qm.FieldCondition(
            key="knowledge_base_id", match=qm.MatchValue(value=knowledge_base_id)
        )
    ]
    if document_ids:
        must.append(
            qm.FieldCondition(key="document_id", match=qm.MatchAny(any=document_ids))
        )
    return qm.Filter(must=must)


def build_chunk_id(document_id: str, version: int, chunk_index: int) -> str:
    """Детерминированный id чанка: повторная индексация той же версии
    перезаписывает точку, а не плодит дубликаты."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}/{version}/{chunk_index}"))


def _to_retrieved(point: qm.ScoredPoint) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        chunk_id=str(point.id),
        document_id=payload.get("document_id", ""),
        document_name=payload.get("document_name"),
        text=payload.get("text", ""),
        score=point.score,
        chunk_index=payload.get("chunk_index", 0),
        page=payload.get("page"),
        section=payload.get("section"),
    )
