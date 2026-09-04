"""Эндпоинт /metrics: доступен Prometheus и не требует токена."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_metrics_endpoint_is_scrapeable_without_a_token():
    """Scrape выполняет Prometheus внутри сети, а не человек с токеном."""
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "rag_queries_total" in response.text


def test_metrics_are_hidden_from_the_public_schema():
    client = TestClient(create_app())

    schema = client.get("/openapi.json").json()

    assert "/metrics" not in schema["paths"]


def test_a_chat_request_shows_up_in_metrics():
    """Метрики питаются тем же событием, что и структурный лог запроса."""
    from app.observability.events import traced_query

    client = TestClient(create_app())
    before = client.get("/metrics").text

    with traced_query("kb-1", "вопрос") as trace:
        trace.has_answer = True
        trace.llm_model = "qwen3:4b"
        trace.llm_provider = "ollama"

    after = client.get("/metrics").text
    assert before != after
    assert 'rag_model_usage_total{model="qwen3:4b",provider="ollama"}' in after


def test_retrieval_latency_is_actually_measured(monkeypatch):
    """Поле retrieval_latency_ms существовало, но его никто не заполнял.

    Метрика rag_retrieval_latency_seconds из-за этого оставалась пустой
    навсегда, и «куда уходит время» по графику было не понять.
    """
    from app.observability import events

    recorded: list[dict] = []
    monkeypatch.setattr(events, "record_query", recorded.append)

    with events.traced_query("kb-1", "вопрос") as trace:
        trace.retrieval_latency_ms = 12.5

    assert recorded[0]["retrieval_latency_ms"] == 12.5
