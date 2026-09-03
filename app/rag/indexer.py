"""Сервис индексации: документ из хранилища — в векторы Qdrant.

Собирает уже готовые части (парсинг → нормализация → чанкинг → эмбеддинги →
запись) и отвечает за то, чтобы после переиндексации не оставалось векторов
предыдущей версии.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunking import ChunkingStrategy, get_chunking_strategy
from app.ingestion.parsers import ParserRegistry
from app.rag.embeddings import EmbeddingProvider
from app.rag.sparse import HashedSparseVectorizer
from app.rag.vector_store import ChunkPoint, QdrantVectorStore, build_chunk_id


@dataclass(slots=True)
class IndexingResult:
    document_id: str
    version: int
    chunk_count: int
    embedding_version: str
    chunking_strategy: str


class DocumentIndexer:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        parsers: ParserRegistry | None = None,
        strategy: ChunkingStrategy | None = None,
        sparse_vectorizer: HashedSparseVectorizer | None = None,
        batch_size: int = 32,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._parsers = parsers or ParserRegistry()
        self._strategy = strategy or get_chunking_strategy("recursive")
        self._sparse_vectorizer = sparse_vectorizer or HashedSparseVectorizer()
        self._batch_size = batch_size

    def index(
        self,
        path: Path,
        document_id: str,
        knowledge_base_id: str,
        version: int = 1,
        document_name: str | None = None,
        previous_version: int | None = None,
    ) -> IndexingResult:
        document = self._parsers.parse(path)
        chunks = self._strategy.split(document)

        points: list[ChunkPoint] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embeddings.embed_texts([chunk.text for chunk in batch])
            points.extend(
                ChunkPoint(
                    chunk_id=build_chunk_id(document_id, version, chunk.index),
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    vector=vector,
                    sparse_vector=self._sparse_vectorizer.vectorize(chunk.text),
                    version=version,
                    page=chunk.page,
                    section=chunk.section,
                    document_name=document_name or path.name,
                    embedding_version=self._embeddings.version,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            )

        self._vector_store.ensure_collection()
        self._vector_store.upsert_chunks(points)
        if previous_version is not None:
            # Порядок важен: старые векторы удаляются только после успешной
            # записи новых, иначе сбой оставил бы документ без индекса.
            self._vector_store.delete_document(document_id, version=previous_version)

        return IndexingResult(
            document_id=document_id,
            version=version,
            chunk_count=len(points),
            embedding_version=self._embeddings.version,
            chunking_strategy=self._strategy.name,
        )

    def remove(self, document_id: str) -> None:
        self._vector_store.delete_document(document_id)
