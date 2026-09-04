import httpx
import pytest

from app.core.errors import InferenceError
from app.llm.base import GenerationRequest
from app.llm.ollama import OllamaProvider


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(base_url="http://localhost:11434", timeout_s=5.0)


@pytest.fixture
def request_obj() -> GenerationRequest:
    return GenerationRequest(system="system", prompt="prompt")


def test_generation_returns_text_and_metrics(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        assert json["stream"] is False
        assert json["system"] == "system"
        return httpx.Response(
            200,
            json={"response": " ответ ", "prompt_eval_count": 10, "eval_count": 4},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("app.llm.ollama.httpx.post", handler)

    result = provider.generate(request_obj, model="qwen3:4b")

    assert result.text == "ответ"
    assert result.model == "qwen3:4b"
    assert result.provider == "ollama"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4
    assert result.latency_ms >= 0


def test_json_schema_is_passed_as_structured_output_format(monkeypatch, provider):
    captured: dict = {}

    def handler(url, json, timeout):
        captured.update(json)
        return httpx.Response(
            200, json={"response": "{}"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("app.llm.ollama.httpx.post", handler)
    schema = {"type": "object"}

    provider.generate(
        GenerationRequest(system="s", prompt="p", json_schema=schema, max_tokens=128),
        model="qwen3:4b",
    )

    assert captured["format"] == schema
    assert captured["options"]["num_predict"] == 128


def test_timeout_becomes_inference_error(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("app.llm.ollama.httpx.post", handler)

    with pytest.raises(InferenceError, match="не ответила"):
        provider.generate(request_obj, model="qwen3:4b")


def test_runtime_failure_becomes_inference_error(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.llm.ollama.httpx.post", handler)

    with pytest.raises(InferenceError, match="Ошибка Ollama"):
        provider.generate(request_obj, model="qwen3:4b")


def test_empty_response_is_treated_as_failure(monkeypatch, provider, request_obj):
    def handler(url, json, timeout):
        return httpx.Response(
            200, json={"response": "   "}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr("app.llm.ollama.httpx.post", handler)

    with pytest.raises(InferenceError, match="пустой ответ"):
        provider.generate(request_obj, model="qwen3:4b")


def test_health_check_reflects_runtime_availability(monkeypatch, provider):
    monkeypatch.setattr(
        "app.llm.ollama.httpx.get",
        lambda url, timeout: httpx.Response(
            200, json={"version": "0.5.0"}, request=httpx.Request("GET", url)
        ),
    )
    assert provider.health_check() is True

    def failing(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("app.llm.ollama.httpx.get", failing)
    assert provider.health_check() is False


def test_model_availability_ignores_tag_differences(monkeypatch, provider):
    monkeypatch.setattr(
        "app.llm.ollama.httpx.get",
        lambda url, timeout: httpx.Response(
            200,
            json={"models": [{"name": "qwen3:14b", "size": 9_000_000_000,
                              "details": {"parameter_size": "14B"}}]},
            request=httpx.Request("GET", url),
        ),
    )

    assert provider.is_model_available("qwen3:4b") is True
    assert provider.is_model_available("gemma3:12b") is False
    assert provider.list_models()[0].parameter_size == "14B"
