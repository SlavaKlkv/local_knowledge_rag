"""Метрики Prometheus: что считается и, главное, чего в метках нет."""

import pytest
from prometheus_client.parser import text_string_to_metric_families

from app.observability.metrics import REGISTRY, record_indexing, record_query, render


def sample(name: str, **labels) -> float:
    payload, _ = render()
    for family in text_string_to_metric_families(payload.decode()):
        for metric in family.samples:
            if metric.name == name and all(
                metric.labels.get(key) == value for key, value in labels.items()
            ):
                return metric.value
    return 0.0


def trace(**overrides) -> dict:
    base = {
        "knowledge_base_id": "kb-1",
        "original_query": "Сколько дней отпуска?",
        "has_answer": True,
        "llm_model": "qwen3:4b",
        "llm_provider": "ollama",
        "retrieved_chunk_ids": ["c1", "c2"],
        "total_latency_ms": 1500.0,
        "retrieval_latency_ms": 120.0,
        "llm_latency_ms": 1300.0,
    }
    base.update(overrides)
    return base


def test_successful_query_is_counted_with_its_answer_flag():
    before = sample("rag_queries_total", has_answer="true")

    record_query(trace())

    assert sample("rag_queries_total", has_answer="true") == before + 1


def test_no_answer_is_counted_separately():
    before = sample("rag_queries_total", has_answer="false")

    record_query(trace(has_answer=False))

    assert sample("rag_queries_total", has_answer="false") == before + 1


def test_failed_query_is_not_counted_as_a_successful_one():
    before_ok = sample("rag_queries_total", has_answer="true")
    before_err = sample("rag_query_errors_total", error_type="inference")

    record_query(trace(error="InferenceError: Ollama недоступен"))

    assert sample("rag_queries_total", has_answer="true") == before_ok
    assert sample("rag_query_errors_total", error_type="inference") == before_err + 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("Ошибка Ollama при генерации моделью 'qwen3:4b'", "inference"),
        ("Документ не найден", "not_found"),
        ("Нет доступа к этой базе знаний", "forbidden"),
        ("Что-то пошло не так", "other"),
    ],
)
def test_error_label_is_a_class_not_the_message(error, expected):
    """Иначе имена моделей и идентификаторы из текста ошибки попали бы в метку."""
    before = sample("rag_query_errors_total", error_type=expected)

    record_query(trace(error=error))

    assert sample("rag_query_errors_total", error_type=expected) == before + 1


def test_no_metric_carries_a_knowledge_base_or_document_identifier():
    """Главная защита от взрыва кардинальности: идентификаторов в метках нет."""
    record_query(trace())
    record_indexing("succeeded", chunk_count=3)
    payload, _ = render()

    text = payload.decode()
    assert "kb-1" not in text
    assert "knowledge_base" not in text
    assert "document_id" not in text
    assert "Сколько дней отпуска" not in text


def test_model_usage_is_counted_per_model():
    before = sample("rag_model_usage_total", model="qwen3:4b", provider="ollama")

    record_query(trace())

    assert sample("rag_model_usage_total", model="qwen3:4b", provider="ollama") == before + 1


def test_fallback_is_counted_only_when_the_ring_switched():
    before = sample("rag_model_fallback_total", from_model="qwen3:4b", to_model="gemma3:4b")

    record_query(trace())
    assert sample(
        "rag_model_fallback_total", from_model="qwen3:4b", to_model="gemma3:4b"
    ) == before

    record_query(trace(fallback_from="qwen3:4b", llm_model="gemma3:4b"))
    assert sample(
        "rag_model_fallback_total", from_model="qwen3:4b", to_model="gemma3:4b"
    ) == before + 1


def test_latency_is_recorded_in_seconds_not_milliseconds():
    before = sample("rag_query_latency_seconds_sum")

    record_query(trace(total_latency_ms=2500.0))

    assert sample("rag_query_latency_seconds_sum") == pytest.approx(before + 2.5)


def test_missing_latency_is_not_recorded_as_zero():
    before_sum = sample("rag_generation_latency_seconds_sum")
    before_count = sample("rag_generation_latency_seconds_count")

    record_query(trace(llm_latency_ms=None))

    assert sample("rag_generation_latency_seconds_sum") == before_sum
    assert sample("rag_generation_latency_seconds_count") == before_count


def test_indexing_records_documents_and_chunks():
    before_docs = sample("rag_documents_indexed_total", status="succeeded")
    before_chunks = sample("rag_chunks_indexed_total")

    record_indexing("succeeded", chunk_count=7)

    assert sample("rag_documents_indexed_total", status="succeeded") == before_docs + 1
    assert sample("rag_chunks_indexed_total") == before_chunks + 7


def test_failed_indexing_counts_the_document_but_no_chunks():
    before_chunks = sample("rag_chunks_indexed_total")
    before_docs = sample("rag_documents_indexed_total", status="failed")

    record_indexing("failed")

    assert sample("rag_documents_indexed_total", status="failed") == before_docs + 1
    assert sample("rag_chunks_indexed_total") == before_chunks


def test_registry_is_private_to_the_application():
    """Общий глобальный реестр падал бы при повторном создании приложения."""
    from prometheus_client import REGISTRY as GLOBAL_REGISTRY

    assert REGISTRY is not GLOBAL_REGISTRY


def test_exposition_uses_the_prometheus_content_type():
    payload, content_type = render()

    assert "text/plain" in content_type
    assert b"rag_queries_total" in payload
