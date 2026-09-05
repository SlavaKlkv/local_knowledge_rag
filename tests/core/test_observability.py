import logging

import pytest

from app.observability.events import traced_query


def test_trace_captures_query_and_latency(caplog):
    caplog.set_level(logging.INFO, logger="rag.query")

    with traced_query("kb-1", "вопрос") as trace:
        trace.llm_model = "qwen3:4b"
        trace.has_answer = True

    record = next(r for r in caplog.records if r.name == "rag.query")
    payload = record.rag_query

    assert payload["knowledge_base_id"] == "kb-1"
    assert payload["original_query"] == "вопрос"
    assert payload["llm_model"] == "qwen3:4b"
    assert payload["total_latency_ms"] >= 0


def test_trace_records_error_and_reraises(caplog):
    caplog.set_level(logging.INFO, logger="rag.query")

    with pytest.raises(ValueError), traced_query("kb-1", "вопрос") as trace:
        trace.llm_model = "qwen3:4b"
        raise ValueError("модель недоступна")

    record = next(r for r in caplog.records if r.name == "rag.query")
    assert record.rag_query["error"] == "модель недоступна"


def test_none_fields_are_omitted_from_payload():
    from app.observability.events import QueryTrace

    trace = QueryTrace(knowledge_base_id="kb-1", original_query="вопрос")

    assert "llm_model" not in trace.as_dict()
    assert "error" not in trace.as_dict()
    assert trace.as_dict()["original_query"] == "вопрос"
