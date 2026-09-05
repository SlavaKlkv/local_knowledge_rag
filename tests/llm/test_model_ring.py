import pytest

from app.core.errors import InferenceError
from app.hardware.profiles import ModelRingEntry
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.llm.ring import ModelHealth, ModelRing


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeProvider(LocalLLMProvider):
    name = "fake"

    def __init__(self, failing_models: set[str] | None = None) -> None:
        self.failing_models = failing_models or set()
        self.calls: list[str] = []

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        self.calls.append(model)
        if model in self.failing_models:
            raise InferenceError(f"модель '{model}' недоступна")
        return GenerationResult(text="ответ", model=model, provider=self.name, latency_ms=1)

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return []


def _entries() -> list[ModelRingEntry]:
    return [
        ModelRingEntry("qwen", "qwen3:4b"),
        ModelRingEntry("gemma", "gemma3:4b"),
        ModelRingEntry("llama", "llama3.1:8b"),
    ]


def _request() -> GenerationRequest:
    return GenerationRequest(system="s", prompt="p")


def test_healthy_model_answers_without_fallback():
    provider = FakeProvider()
    ring = ModelRing(provider, _entries())

    outcome = ring.generate(_request())

    assert outcome.result.model == "qwen3:4b"
    assert outcome.attempts == 1
    assert outcome.fallback_events == []


def test_failing_model_falls_back_to_next_in_ring():
    provider = FakeProvider(failing_models={"qwen3:4b"})
    ring = ModelRing(provider, _entries())

    outcome = ring.generate(_request())

    assert outcome.result.model == "gemma3:4b"
    assert outcome.attempts == 2
    assert [e.to_model for e in outcome.fallback_events] == ["gemma3:4b"]


def test_ring_is_continuous_and_wraps_around():
    provider = FakeProvider(failing_models={"qwen3:4b", "gemma3:4b", "llama3.1:8b"})
    ring = ModelRing(provider, _entries(), max_attempts=5)

    with pytest.raises(InferenceError, match="исчерпано"):
        ring.generate(_request())

    # Все три модели были опрошены ровно один раз каждая за один запрос.
    assert provider.calls == ["qwen3:4b", "gemma3:4b", "llama3.1:8b"]


def test_max_attempts_limits_a_single_request():
    provider = FakeProvider(failing_models={"qwen3:4b", "gemma3:4b", "llama3.1:8b"})
    ring = ModelRing(provider, _entries(), max_attempts=2)

    with pytest.raises(InferenceError):
        ring.generate(_request())

    assert len(provider.calls) == 2


def test_timeout_budget_stops_further_attempts():
    clock = FakeClock()

    class SlowFailingProvider(FakeProvider):
        def generate(self, request, model):
            clock.advance(100)
            return super().generate(request, model)

    provider = SlowFailingProvider(failing_models={"qwen3:4b", "gemma3:4b", "llama3.1:8b"})
    ring = ModelRing(
        provider, _entries(), max_attempts=10, timeout_budget_s=50.0, clock=clock
    )

    with pytest.raises(InferenceError):
        ring.generate(_request())

    # Бюджет 50с исчерпывается уже после первой попытки длиной 100с.
    assert len(provider.calls) == 1


def test_repeated_failures_trigger_cooldown_and_skip_the_model():
    clock = FakeClock()
    provider = FakeProvider(failing_models={"qwen3:4b"})
    ring = ModelRing(
        provider, _entries(), max_attempts=2, failure_threshold=2, cooldown_s=60.0, clock=clock
    )

    outcome = ring.generate(_request())  # 1-й провал qwen -> DEGRADED, ответ от gemma
    assert outcome.result.model == "gemma3:4b"

    qwen = next(m for m in ring.members if m.model == "qwen3:4b")
    assert qwen.state == ModelHealth.DEGRADED


def test_model_enters_cooldown_after_failure_threshold():
    clock = FakeClock()
    provider = FakeProvider(failing_models={"qwen3:4b"})
    ring = ModelRing(
        provider, _entries(), max_attempts=2, failure_threshold=1, cooldown_s=60.0, clock=clock
    )

    ring.generate(_request())

    qwen = next(m for m in ring.members if m.model == "qwen3:4b")
    assert qwen.state == ModelHealth.COOLDOWN
    assert qwen.cooldown_until == 60.0


def test_model_recovers_after_cooldown_expires():
    clock = FakeClock()
    provider = FakeProvider(failing_models={"qwen3:4b"})
    ring = ModelRing(
        provider,
        _entries(),
        max_attempts=3,
        failure_threshold=1,
        cooldown_s=60.0,
        clock=clock,
    )
    ring.generate(_request())  # qwen -> COOLDOWN, cursor на gemma

    clock.advance(61.0)
    provider.failing_models.clear()  # модель "поднялась"
    provider.calls.clear()
    ring._cursor = 0  # смотрим с начала кольца, как будто новый запрос дошёл до qwen

    outcome = ring.generate(_request())

    assert outcome.result.model == "qwen3:4b"


def test_successful_generation_resets_failure_state():
    provider = FakeProvider(failing_models={"qwen3:4b"})
    ring = ModelRing(provider, _entries(), max_attempts=2, failure_threshold=5)

    ring.generate(_request())  # qwen проваливается, DEGRADED
    provider.failing_models.clear()
    ring._cursor = 0
    ring.generate(_request())  # qwen отвечает успешно

    qwen = next(m for m in ring.members if m.model == "qwen3:4b")
    assert qwen.state == ModelHealth.HEALTHY
    assert qwen.consecutive_failures == 0


def test_empty_ring_is_rejected():
    with pytest.raises(InferenceError):
        ModelRing(FakeProvider(), [])
