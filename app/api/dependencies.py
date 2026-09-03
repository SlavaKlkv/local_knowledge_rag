"""Зависимости FastAPI: сборка компонентов RAG-конвейера."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.api.storage import DocumentStorage
from app.core.config import get_settings
from app.llm.base import LocalLLMProvider
from app.llm.ollama import OllamaProvider
from app.rag.context_builder import ContextBuilder
from app.rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from app.rag.generation import AnswerGenerator
from app.rag.indexer import DocumentIndexer
from app.rag.query_rewriting import QueryRewriter
from app.rag.reranker import CrossEncoderReranker, NoOpReranker, Reranker
from app.rag.retriever import (
    DenseRetriever,
    HybridRetriever,
    Retriever,
    SparseRetriever,
)
from app.rag.vector_store import QdrantVectorStore

ALLOWED_UPLOAD_EXTENSIONS = (
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".docx",
    ".html",
    ".htm",
)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return OllamaEmbeddingProvider()


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


@lru_cache
def get_llm_provider() -> LocalLLMProvider:
    return OllamaProvider()


@lru_cache
def get_document_storage() -> DocumentStorage:
    return DocumentStorage(Path("storage/documents"))


def get_dense_retriever() -> DenseRetriever:
    return DenseRetriever(get_embedding_provider(), get_vector_store())


def get_sparse_retriever() -> SparseRetriever:
    return SparseRetriever(get_vector_store())


def get_retriever() -> Retriever:
    # Флаг конфигурации переключает стратегию, а не ветвление в вызывающем
    # коде: ContextBuilder и /chat всегда работают через единый интерфейс.
    if not get_settings().hybrid_retrieval_enabled:
        return get_dense_retriever()
    return HybridRetriever(get_dense_retriever(), get_sparse_retriever())


def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(get_embedding_provider(), get_vector_store())


@lru_cache
def get_reranker() -> Reranker:
    # Флаг конфигурации переключает реализацию, а не ветвление в вызывающем
    # коде: retriever/generation всегда работают через единый интерфейс.
    settings = get_settings()
    if not settings.rerank_enabled:
        return NoOpReranker()
    return CrossEncoderReranker(model_name=settings.reranker_model)


def get_context_builder() -> ContextBuilder:
    return ContextBuilder()


def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator(get_llm_provider(), get_settings().llm_model)


def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter(get_llm_provider(), get_settings().llm_model)
