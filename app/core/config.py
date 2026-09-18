"""Конфигурация приложения из переменных окружения."""

from functools import lru_cache
from pathlib import Path

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
    # Каталог blobs нужен только для точечной очистки незавершённого pull.
    # В Docker он явно монтируется с хоста; при локальном запуске используем
    # стандартное хранилище Ollama текущего пользователя.
    ollama_blobs_path: Path = Path("~/.ollama/models/blobs")
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
    reranker_model: str = "DiTy/cross-encoder-russian-msmarco"
    rerank_candidates: int = Field(default=30, gt=0)
    rerank_top_k: int = Field(default=8, gt=0)

    hybrid_retrieval_enabled: bool = True

    # Порог выключен по умолчанию: он живёт в шкале конкретной
    # embedding-модели, и зашитое число врало бы при её замене. Подбирается
    # прогоном evaluation, где видно цену отказа от домысливания: ложные
    # срабатывания падают вместе с Recall. Применяется в retrieval, а не после
    # фьюжна — там скор ещё в своей шкале.
    no_answer_min_score: float | None = None

    # Отсечка нерелевантного в /search. Работает в шкале cross-encoder:
    # штатная русская модель — с одним выходом, и sentence-transformers
    # прогоняет его через сигмоиду, поэтому это вероятность 0..1, а не
    # логит. Применяется только когда reranking включён и только когда
    # reranker уверен (см. rerank_trust_min_score) — у NoOpReranker скор
    # приходит из retrieval и живёт в другой шкале.
    search_min_rerank_score: float = 0.2

    # Порог доверия к reranker. Cross-encoder обучен на запросах-вопросах;
    # на запросе из одного-двух слов он уходит за пределы своего
    # распределения и раздаёт всем кандидатам близкие к нулю скоры почти
    # случайно. Уверенность модели видна по её лидеру: на вопросе, ответ на
    # который в базе есть, лучший кандидат получает 0.5..0.97, а на
    # ключевом слове максимум редко дотягивает до 0.3. Ниже этой границы
    # порядок выдачи отдаётся retrieval и лексическому совпадению.
    rerank_trust_min_score: float = 0.3
    no_answer_require_citations: bool = True

    model_ring_enabled: bool = True
    model_ring_max_attempts: int = Field(default=3, gt=0)
    # Бюджет на весь запрос, а не на попытку: он проверяется перед вызовом
    # следующей модели и не обрывает идущую генерацию. Значение меньше
    # inference_timeout_s отключало fallback по таймауту вовсе — модель,
    # зависшая на все 120 с, выносила приговор всему запросу. Запас взят
    # на две полные попытки плюс загрузку весов соседней модели.
    model_ring_timeout_budget_s: float = Field(default=300.0, gt=0)
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
