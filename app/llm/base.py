"""Контракт локального LLM-провайдера.

Бизнес-логика RAG не должна знать, каким runtime запущена модель:
модель (Qwen, Gemma, Llama) и способ её запуска (Ollama, vLLM) — разные вещи.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerationRequest:
    system: str
    prompt: str
    temperature: float = 0.1
    max_tokens: int | None = None
    json_schema: dict | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class ModelInfo:
    name: str
    provider: str
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None


class LocalLLMProvider(ABC):
    """Единый интерфейс: генерация, health check, доступность моделей."""

    name: str

    @abstractmethod
    def generate(self, request: GenerationRequest, model: str) -> GenerationResult: ...

    @abstractmethod
    def health_check(self) -> bool:
        """Доступен ли runtime как таковой."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]: ...

    def is_model_available(self, model: str) -> bool:
        family = model.split(":")[0]
        return any(info.name.split(":")[0] == family for info in self.list_models())
