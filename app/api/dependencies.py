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
from app.rag.retriever import DenseRetriever
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


def get_retriever() -> DenseRetriever:
    return DenseRetriever(get_embedding_provider(), get_vector_store())


def get_indexer() -> DocumentIndexer:
    return DocumentIndexer(get_embedding_provider(), get_vector_store())


def get_context_builder() -> ContextBuilder:
    return ContextBuilder()


def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator(get_llm_provider(), get_settings().llm_model)
