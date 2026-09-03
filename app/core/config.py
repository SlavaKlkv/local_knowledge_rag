"""Конфигурация приложения из переменных окружения."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "local_knowledge_rag"
    postgres_user: str = "rag"
    postgres_password: str = "change-me"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"

    redis_url: str = "redis://localhost:6379/0"

    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    llm_model: str = "qwen3:4b"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = Field(default=768, gt=0)

    # Пользовательское переопределение профиля: детектор только
    # рекомендует, а не решает окончательно.
    hardware_profile_override: str | None = None

    inference_timeout_s: float = 120.0

    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=30, gt=0)
    rerank_top_k: int = Field(default=8, gt=0)

    hybrid_retrieval_enabled: bool = True

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
