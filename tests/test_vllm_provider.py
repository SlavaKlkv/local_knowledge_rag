import httpx
import pytest

from app.core.errors import InferenceError
from app.llm.base import GenerationRequest
from app.llm.vllm import VLLMProvider


@pytest.fixture
def provider() -> VLLMProvider:
    return VLLMProvider(base_url="http://localhost:8000", timeout_s=5.0)


@pytest.fixture
def request_obj() -> GenerationRequest:
    return GenerationRequest(system="system", prompt="prompt")


def test_generation_returns_text_and_metrics(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        assert json["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": " ответ "}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("app.llm.vllm.httpx.post", handler)

    result = provider.generate(request_obj, model="qwen3-14b")

    assert result.text == "ответ"
    assert result.model == "qwen3-14b"
    assert result.provider == "vllm"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4


def test_json_schema_is_passed_as_guided_json(monkeypatch, provider):
    captured: dict = {}

    def handler(url, json, timeout):
        captured.update(json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("app.llm.vllm.httpx.post", handler)
    schema = {"type": "object"}

    provider.generate(
        GenerationRequest(system="s", prompt="p", json_schema=schema, max_tokens=128),
        model="qwen3-14b",
    )

    assert captured["extra_body"]["guided_json"] == schema
    assert captured["max_tokens"] == 128


def test_timeout_becomes_inference_error(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("app.llm.vllm.httpx.post", handler)

    with pytest.raises(InferenceError, match="не ответила"):
        provider.generate(request_obj, model="qwen3-14b")


def test_runtime_failure_becomes_inference_error(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.llm.vllm.httpx.post", handler)

    with pytest.raises(InferenceError, match="Ошибка vLLM"):
        provider.generate(request_obj, model="qwen3-14b")


def test_empty_response_is_treated_as_failure(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("app.llm.vllm.httpx.post", handler)

    with pytest.raises(InferenceError, match="пустой ответ"):
        provider.generate(request_obj, model="qwen3-14b")


def test_no_choices_is_treated_as_empty_response(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        return httpx.Response(200, json={"choices": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr("app.llm.vllm.httpx.post", handler)

    with pytest.raises(InferenceError, match="пустой ответ"):
        provider.generate(request_obj, model="qwen3-14b")


def test_health_check_reflects_runtime_availability(monkeypatch, provider):
    monkeypatch.setattr(
        "app.llm.vllm.httpx.get",
        lambda url, timeout: httpx.Response(200, request=httpx.Request("GET", url)),
    )
    assert provider.health_check() is True

    def failing(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("app.llm.vllm.httpx.get", failing)
    assert provider.health_check() is False


def test_list_models_maps_openai_style_response(monkeypatch, provider):
    monkeypatch.setattr(
        "app.llm.vllm.httpx.get",
        lambda url, timeout: httpx.Response(
            200,
            json={"data": [{"id": "qwen3-14b"}, {"id": "gemma3-12b"}]},
            request=httpx.Request("GET", url),
        ),
    )

    models = provider.list_models()

    assert [m.name for m in models] == ["qwen3-14b", "gemma3-12b"]
    assert all(m.provider == "vllm" for m in models)


def test_list_models_raises_inference_error_when_unreachable(monkeypatch, provider):
    def failing(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("app.llm.vllm.httpx.get", failing)

    with pytest.raises(InferenceError, match="недоступен"):
        provider.list_models()


def _clear_provider_caches():
    from app.api import dependencies

    dependencies.get_base_llm_provider.cache_clear()
    dependencies.get_llm_provider.cache_clear()
    dependencies.get_settings.cache_clear()


def test_dependency_selects_ollama_by_default(monkeypatch):
    from app.api import dependencies
    from app.llm.ollama import OllamaProvider

    _clear_provider_caches()
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    dependencies.get_settings.cache_clear()

    assert isinstance(dependencies.get_base_llm_provider(), OllamaProvider)

    _clear_provider_caches()


def test_dependency_selects_vllm_when_configured(monkeypatch):
    from app.api import dependencies

    _clear_provider_caches()
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")
    dependencies.get_settings.cache_clear()

    assert isinstance(dependencies.get_base_llm_provider(), VLLMProvider)

    _clear_provider_caches()


def test_dependency_rejects_unknown_provider(monkeypatch):
    from app.api import dependencies
    from app.core.errors import ValidationError

    _clear_provider_caches()
    monkeypatch.setenv("INFERENCE_PROVIDER", "something-else")
    dependencies.get_settings.cache_clear()

    with pytest.raises(ValidationError, match="Неизвестный INFERENCE_PROVIDER"):
        dependencies.get_base_llm_provider()

    _clear_provider_caches()


def test_llm_provider_is_ring_backed_by_default(monkeypatch):
    from app.api import dependencies
    from app.llm.ring_provider import RingLLMProvider

    _clear_provider_caches()
    monkeypatch.setenv("MODEL_RING_ENABLED", "true")
    dependencies.get_settings.cache_clear()
    dependencies.get_active_profile.cache_clear()

    assert isinstance(dependencies.get_llm_provider(), RingLLMProvider)

    _clear_provider_caches()
    dependencies.get_active_profile.cache_clear()


def test_llm_provider_is_the_base_provider_when_ring_disabled(monkeypatch):
    from app.api import dependencies
    from app.llm.ollama import OllamaProvider

    _clear_provider_caches()
    monkeypatch.setenv("MODEL_RING_ENABLED", "false")
    dependencies.get_settings.cache_clear()

    assert isinstance(dependencies.get_llm_provider(), OllamaProvider)

    _clear_provider_caches()
