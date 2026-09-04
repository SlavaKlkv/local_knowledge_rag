import httpx
import pytest

from app.core.errors import InferenceError
from app.rag.embeddings import OllamaEmbeddingProvider


class _StubTransport(httpx.MockTransport):
    pass


@pytest.fixture
def provider() -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="test-embed", dimension=3
    )


def _patch_post(monkeypatch, handler):
    monkeypatch.setattr("app.rag.embeddings.httpx.post", handler)


def test_embeds_batch_of_texts(monkeypatch, provider):
    def handler(url, json, timeout):
        assert json["input"] == ["a", "b"]
        return httpx.Response(
            200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            request=httpx.Request("POST", url),
        )

    _patch_post(monkeypatch, handler)

    assert provider.embed_texts(["a", "b"]) == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_empty_batch_does_not_call_the_model(monkeypatch, provider):
    def handler(*args, **kwargs):  # pragma: no cover - не должен вызываться
        raise AssertionError("модель не должна вызываться для пустого батча")

    _patch_post(monkeypatch, handler)

    assert provider.embed_texts([]) == []


def test_unreachable_runtime_raises_instead_of_falling_back(monkeypatch, provider):
    def handler(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    _patch_post(monkeypatch, handler)

    with pytest.raises(InferenceError, match="недоступна"):
        provider.embed_texts(["a"])


def test_dimension_mismatch_is_reported(monkeypatch, provider):
    def handler(url, json, timeout):
        return httpx.Response(
            200, json={"embeddings": [[0.1, 0.2]]}, request=httpx.Request("POST", url)
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(InferenceError, match="Размерность"):
        provider.embed_texts(["a"])


def test_truncated_response_is_rejected(monkeypatch, provider):
    def handler(url, json, timeout):
        return httpx.Response(
            200, json={"embeddings": [[0.1, 0.2, 0.3]]},
            request=httpx.Request("POST", url),
        )

    _patch_post(monkeypatch, handler)

    with pytest.raises(InferenceError, match="ожидалось 2"):
        provider.embed_texts(["a", "b"])


def test_version_changes_with_model(provider):
    other = OllamaEmbeddingProvider(model="another", dimension=3)
    assert provider.version != other.version
