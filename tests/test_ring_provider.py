import pytest

from app.core.errors import InferenceError
from app.hardware.profiles import ModelRingEntry
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.llm.ring import ModelRing
from app.llm.ring_provider import RingLLMProvider


class FakeProvider(LocalLLMProvider):
    name = "fake"

    def __init__(self, failing_models: set[str] | None = None) -> None:
        self.failing_models = failing_models or set()
        self.healthy = True

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        if model in self.failing_models:
            raise InferenceError(f"модель '{model}' недоступна")
        return GenerationResult(text="ответ", model=model, provider=self.name, latency_ms=1)

    def health_check(self) -> bool:
        return self.healthy

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="qwen3:4b", provider=self.name)]


def _entries() -> list[ModelRingEntry]:
    return [
        ModelRingEntry("qwen", "qwen3:4b"),
        ModelRingEntry("gemma", "gemma3:4b"),
    ]


def test_generate_delegates_to_the_ring_and_ignores_the_model_argument():
    ring = ModelRing(FakeProvider(), _entries())
    provider = RingLLMProvider(ring)

    result = provider.generate(
        GenerationRequest(system="s", prompt="p"), model="это имя игнорируется"
    )

    assert result.model == "qwen3:4b"


def test_generate_exposes_the_ring_outcome_for_observability():
    ring = ModelRing(FakeProvider(failing_models={"qwen3:4b"}), _entries(), max_attempts=2)
    provider = RingLLMProvider(ring)

    provider.generate(GenerationRequest(system="s", prompt="p"), model="qwen3:4b")

    assert provider.last_outcome is not None
    assert provider.last_outcome.result.model == "gemma3:4b"
    assert len(provider.last_outcome.fallback_events) == 1


def test_ring_exhaustion_propagates_as_inference_error():
    ring = ModelRing(
        FakeProvider(failing_models={"qwen3:4b", "gemma3:4b"}), _entries(), max_attempts=2
    )
    provider = RingLLMProvider(ring)

    with pytest.raises(InferenceError, match="исчерпано"):
        provider.generate(GenerationRequest(system="s", prompt="p"), model="qwen3:4b")


def test_health_check_and_list_models_delegate_to_the_underlying_provider():
    base = FakeProvider()
    ring = ModelRing(base, _entries())
    provider = RingLLMProvider(ring)

    assert provider.health_check() is True
    assert [m.name for m in provider.list_models()] == ["qwen3:4b"]

    base.healthy = False
    assert provider.health_check() is False
