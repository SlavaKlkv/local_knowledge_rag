"""Непрерывное кольцо моделей с health-состояниями и cooldown.

Кольцо непрерывно на уровне сервиса (Qwen → Gemma → Llama → Qwen → ...), но
один пользовательский запрос не должен циркулировать по нему бесконечно:
ограничен max_attempts и timeout_budget. Восстановившаяся после cooldown
модель возвращается в кольцо автоматически — специального действия для
"починки" не требуется.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.errors import InferenceError
from app.hardware.profiles import ModelRingEntry
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider

logger = logging.getLogger("rag.model_ring")


class ModelHealth(enum.StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class RingMember:
    family: str
    model: str
    state: ModelHealth = ModelHealth.HEALTHY
    consecutive_failures: int = 0
    cooldown_until: float | None = None


@dataclass(slots=True)
class FallbackEvent:
    from_model: str
    to_model: str
    reason: str


@dataclass(slots=True)
class RingGenerationResult:
    result: GenerationResult
    attempts: int
    fallback_events: list[FallbackEvent] = field(default_factory=list)


class ModelRing:
    def __init__(
        self,
        provider: LocalLLMProvider,
        entries: list[ModelRingEntry],
        max_attempts: int = 3,
        timeout_budget_s: float = 45.0,
        cooldown_s: float = 60.0,
        failure_threshold: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not entries:
            raise InferenceError("Кольцо моделей не может быть пустым")
        self._provider = provider
        self._members = [RingMember(family=e.family, model=e.model) for e in entries]
        self._max_attempts = max_attempts
        self._timeout_budget_s = timeout_budget_s
        self._cooldown_s = cooldown_s
        self._failure_threshold = failure_threshold
        self._clock = clock
        self._cursor = 0

    @property
    def members(self) -> list[RingMember]:
        return list(self._members)

    def generate(self, request: GenerationRequest) -> RingGenerationResult:
        started = self._clock()
        attempts = 0
        events: list[FallbackEvent] = []
        last_error: Exception | None = None
        previous_model: str | None = None

        for offset in range(len(self._members)):
            if attempts >= self._max_attempts:
                break
            if self._clock() - started >= self._timeout_budget_s:
                break

            index = (self._cursor + offset) % len(self._members)
            member = self._members[index]
            self._recover_if_due(member)

            if member.state in (ModelHealth.COOLDOWN, ModelHealth.UNAVAILABLE):
                continue

            if previous_model is not None:
                events.append(
                    FallbackEvent(
                        from_model=previous_model,
                        to_model=member.model,
                        reason=str(last_error) if last_error else "fallback",
                    )
                )

            attempts += 1
            try:
                result = self._provider.generate(request, model=member.model)
            except InferenceError as exc:
                last_error = exc
                previous_model = member.model
                self._record_failure(member)
                logger.warning(
                    "model_ring_attempt_failed",
                    extra={"model": member.model, "attempt": attempts, "error": str(exc)},
                )
                continue

            self._record_success(member)
            self._cursor = index
            return RingGenerationResult(result=result, attempts=attempts, fallback_events=events)

        raise InferenceError(
            f"Кольцо моделей исчерпано после {attempts} попыток: {last_error}"
        ) from last_error

    def _record_failure(self, member: RingMember) -> None:
        member.consecutive_failures += 1
        if member.consecutive_failures >= self._failure_threshold:
            member.state = ModelHealth.COOLDOWN
            member.cooldown_until = self._clock() + self._cooldown_s
            # Кольцо продолжает вращаться со следующего кандидата — текущий
            # запрос не ждёт cooldown этой модели.
            self._cursor = (self._members.index(member) + 1) % len(self._members)
        else:
            member.state = ModelHealth.DEGRADED

    def _record_success(self, member: RingMember) -> None:
        member.consecutive_failures = 0
        member.state = ModelHealth.HEALTHY
        member.cooldown_until = None

    def _recover_if_due(self, member: RingMember) -> None:
        if member.state != ModelHealth.COOLDOWN:
            return
        if member.cooldown_until is None or self._clock() < member.cooldown_until:
            return
        # Cooldown истёк: даём модели ещё один шанс, а не сразу HEALTHY —
        # реальный health check выполнит вызывающий код через is_model_available.
        member.state = ModelHealth.DEGRADED
        member.consecutive_failures = 0
        member.cooldown_until = None
