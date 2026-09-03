"""Конфигурация приложения из переменных окружения."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    # Обязателен к переопределению в проде: дефолт годится только для
    # локальной разработки и намеренно выглядит как заглушка.
    secret_key: str = "dev-only-insecure-secret-change-me"
    access_token_ttl_s: int = Field(default=8 * 60 * 60, gt=0)

    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "local_knowledge_rag"
    postgres_user: str = "rag"
    postgres_password: str = "change-me"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"

    redis_url: str = "redis://localhost:6379/0"

    # Eager-режим выполняет задачи Celery прямо в вызывающем процессе:
    # нужен для тестов и локального запуска без отдельного воркера.
    celery_task_always_eager: bool = False

    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    # Явный выбор runtime пользователем: "ollama" или "vllm". Детектор
    # только рекомендует — переключение всегда по этой настройке.
    inference_provider: str = "ollama"
    llm_model: str = "qwen3:4b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = Field(default=768, gt=0)

    # Пользовательское переопределение профиля: детектор только
    # рекомендует, а не решает окончательно.
    hardware_profile_override: str | None = None

    # Установка системного компонента (Ollama) из приложения — это запуск
    # системной команды по HTTP-запросу, поэтому по умолчанию выключена:
    # приложение отдаёт готовую команду, а выполняет её пользователь сам.
    runtime_install_enabled: bool = False

    inference_timeout_s: float = 120.0

    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=30, gt=0)
    rerank_top_k: int = Field(default=8, gt=0)

    hybrid_retrieval_enabled: bool = True

    model_ring_enabled: bool = True
    model_ring_max_attempts: int = Field(default=3, gt=0)
    model_ring_timeout_budget_s: float = Field(default=45.0, gt=0)
    model_ring_cooldown_s: float = Field(default=60.0, gt=0)
    model_ring_failure_threshold: int = Field(default=2, gt=0)

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
