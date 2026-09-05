import httpx

from app.hardware.runtime_detector import InferenceRuntime, RuntimeDetector


def _detector(handler) -> RuntimeDetector:
    return RuntimeDetector(
        ollama_url="http://localhost:11434",
        vllm_url="http://localhost:8000",
        http_get=handler,
    )


def test_both_runtimes_available():
    def handler(url, timeout):
        return httpx.Response(200, request=httpx.Request("GET", url))

    result = _detector(handler).detect()

    assert result.is_available(InferenceRuntime.OLLAMA)
    assert result.is_available(InferenceRuntime.VLLM)
    assert result.recommended == InferenceRuntime.OLLAMA


def test_only_vllm_available_is_recommended():
    def handler(url, timeout):
        if "11434" in url:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, request=httpx.Request("GET", url))

    result = _detector(handler).detect()

    assert result.is_available(InferenceRuntime.OLLAMA) is False
    assert result.is_available(InferenceRuntime.VLLM) is True
    assert result.recommended == InferenceRuntime.VLLM


def test_no_runtime_available_recommends_nothing():
    def handler(url, timeout):
        raise httpx.ConnectError("refused")

    result = _detector(handler).detect()

    assert result.recommended is None
    assert all(not r.available for r in result.runtimes)


def test_non_2xx_status_counts_as_unavailable():
    def handler(url, timeout):
        return httpx.Response(500, request=httpx.Request("GET", url))

    result = _detector(handler).detect()

    assert result.is_available(InferenceRuntime.OLLAMA) is False
    unavailable = next(r for r in result.runtimes if r.runtime == InferenceRuntime.OLLAMA)
    assert "500" in unavailable.detail


def test_timeout_is_reported_as_unavailable_with_detail():
    def handler(url, timeout):
        raise httpx.TimeoutException("timed out")

    result = _detector(handler).detect()

    ollama = next(r for r in result.runtimes if r.runtime == InferenceRuntime.OLLAMA)
    assert ollama.available is False
    assert ollama.detail
