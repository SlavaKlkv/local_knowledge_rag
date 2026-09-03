"""Зависимости FastAPI: сборка компонентов RAG-конвейера."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.api.storage import DocumentStorage
from app.core.config import get_settings
from app.core.errors import ValidationError
from app.hardware.detector import HardwareDetector
from app.hardware.profiles import (
    HardwareProfile,
    ProfileRecommender,
    get_profile_definition,
)
from app.llm.base import LocalLLMProvider
from app.llm.ollama import OllamaProvider
from app.llm.ring import ModelRing
from app.llm.ring_provider import RingLLMProvider
from app.llm.vllm import VLLMProvider
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
def get_base_llm_provider() -> LocalLLMProvider:
    # LLM модель ≠ inference runtime: выбор провайдера не меняет остальной
    # pipeline, только то, куда уходит запрос генерации.
    provider = get_settings().inference_provider
    if provider == "ollama":
        return OllamaProvider()
    if provider == "vllm":
        return VLLMProvider()
    raise ValidationError(
        f"Неизвестный INFERENCE_PROVIDER '{provider}'. Доступны: ollama, vllm"
    )


@lru_cache
def get_active_profile() -> HardwareProfile:
    """Аппаратный профиль: ручное переопределение либо рекомендация детектора.

    Определяется один раз за время жизни процесса — оборудование машины не
    меняется на лету, а повторная детекция на каждый запрос не нужна.
    """
    override = get_settings().hardware_profile_override
    if override:
        try:
            return HardwareProfile(override)
        except ValueError as exc:
            raise ValidationError(
                f"Неизвестный HARDWARE_PROFILE_OVERRIDE '{override}'. "
                f"Доступны: {', '.join(p.value for p in HardwareProfile)}"
            ) from exc
    hardware = HardwareDetector().detect()
    return ProfileRecommender().recommend(hardware)


@lru_cache
def get_llm_provider() -> LocalLLMProvider:
    # Флаг конфигурации переключает реализацию, а не ветвление в вызывающем
    # коде: AnswerGenerator/QueryRewriter всегда работают через единый
    # интерфейс LocalLLMProvider и не знают, кольцо за ним или одна модель.
    base_provider = get_base_llm_provider()
    if not get_settings().model_ring_enabled:
        return base_provider

    settings = get_settings()
    ring = ModelRing(
        base_provider,
        get_profile_definition(get_active_profile()).ring,
        max_attempts=settings.model_ring_max_attempts,
        timeout_budget_s=settings.model_ring_timeout_budget_s,
        cooldown_s=settings.model_ring_cooldown_s,
        failure_threshold=settings.model_ring_failure_threshold,
    )
    return RingLLMProvider(ring)


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
