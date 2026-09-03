"""Адаптер: ModelRing как обычный LocalLLMProvider.

Позволяет включить кольцевой fallback без изменения AnswerGenerator и
QueryRewriter — они по-прежнему работают через единый интерфейс
LocalLLMProvider и не знают, что за ним стоит кольцо из нескольких моделей.
Параметр model в generate() игнорируется: кольцо само решает, какая модель
отвечает на этот конкретный вызов.
"""

from __future__ import annotations

from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.llm.ring import ModelRing, RingGenerationResult


class RingLLMProvider(LocalLLMProvider):
    name = "model-ring"

    def __init__(self, ring: ModelRing) -> None:
        self._ring = ring
        self.last_outcome: RingGenerationResult | None = None

    @property
    def ring(self) -> ModelRing:
        return self._ring

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        outcome = self._ring.generate(request)
        self.last_outcome = outcome
        return outcome.result

    def health_check(self) -> bool:
        # Здоровье runtime'а, а не конкретной модели — этим управляют
        # health states участников кольца.
        return self._ring.provider.health_check()

    def list_models(self) -> list[ModelInfo]:
        return self._ring.provider.list_models()
