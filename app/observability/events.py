"""Структурные события RAG-конвейера для наблюдаемости.

Логируется как единое событие на запрос: исходный и переписанный запрос,
retrieved chunks, скоры, активная модель, fallback, латентность, ошибки,
no-answer. Формат — плоский JSON-совместимый dict, чтобы события были
пригодны и для логов, и для последующей агрегации в Prometheus/Grafana.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any

from app.observability.metrics import record_query

logger = logging.getLogger("rag.query")


@dataclass(slots=True)
class QueryTrace:
    """Накопитель события одного запроса /chat или /search."""

    knowledge_base_id: str
    original_query: str
    rewritten_query: str | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    reranking_scores: list[float] = field(default_factory=list)
    llm_model: str | None = None
    llm_provider: str | None = None
    fallback_from: str | None = None
    inference_runtime: str | None = None
    hardware_profile: str | None = None
    has_answer: bool | None = None
    error: str | None = None
    retrieval_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    total_latency_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def emit(self) -> None:
        payload = self.as_dict()
        logger.info("rag_query", extra={"rag_query": payload})
        # Метрики питаются тем же событием, что и лог: одно место сбора,
        # два потребителя — иначе они неизбежно разъедутся.
        record_query(payload)


@contextmanager
def traced_query(knowledge_base_id: str, query: str):
    """Замеряет общую латентность запроса и логирует событие по выходу
    из блока — в том числе если внутри произошла ошибка."""
    trace = QueryTrace(knowledge_base_id=knowledge_base_id, original_query=query)
    started = time.perf_counter()
    try:
        yield trace
    except Exception as exc:  # noqa: BLE001 - фиксируем факт ошибки и пробрасываем дальше
        trace.error = str(exc)
        raise
    finally:
        trace.total_latency_ms = (time.perf_counter() - started) * 1000
        trace.emit()
