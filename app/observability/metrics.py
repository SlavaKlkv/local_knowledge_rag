"""Метрики Prometheus для RAG-конвейера.

Метрики умышленно не размечаются идентификаторами базы знаний, документа или
пользователя. Каждое уникальное значение метки — это отдельный временной ряд:
пара сотен баз знаний превратит десяток метрик в тысячи рядов и положит
Prometheus задолго до того, как такая детализация кому-то понадобится.
Разбор конкретного запроса — задача структурных логов (QueryTrace), а не
метрик; метрики отвечают на вопрос «как система живёт в целом».

Гистограммы латентности заданы в секундах, как принято в Prometheus, и с
явными границами: дефолтные бакеты рассчитаны на веб-запросы в десятки
миллисекунд, тогда как локальная генерация занимает единицы и десятки секунд,
и всё интересное схлопывалось бы в верхний бакет.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

# Собственный реестр вместо глобального: тесты создают приложение много раз,
# и на общем реестре повторная регистрация тех же метрик падала бы.
REGISTRY = CollectorRegistry()

_RETRIEVAL_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_GENERATION_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0)

queries_total = Counter(
    "rag_queries_total",
    "Запросы к RAG-конвейеру",
    # has_answer — ограниченный набор значений, поэтому меткой быть может:
    # доля отказов и есть первое, что хочется видеть на графике.
    labelnames=("has_answer",),
    registry=REGISTRY,
)

no_answer_total = Counter(
    "rag_no_answer_total",
    "Отказы отвечать с разбивкой по причине",
    # Код, а не текст причины: в тексте есть скор и порог, и меткой он дал бы
    # неограниченную кардинальность.
    labelnames=("reason",),
    registry=REGISTRY,
)

query_errors_total = Counter(
    "rag_query_errors_total",
    "Запросы, завершившиеся ошибкой",
    labelnames=("error_type",),
    registry=REGISTRY,
)

model_usage_total = Counter(
    "rag_model_usage_total",
    "Обращения к моделям генерации",
    labelnames=("model", "provider"),
    registry=REGISTRY,
)

model_fallback_total = Counter(
    "rag_model_fallback_total",
    "Переключения кольца моделей на следующую модель",
    labelnames=("from_model", "to_model"),
    registry=REGISTRY,
)

query_latency_seconds = Histogram(
    "rag_query_latency_seconds",
    "Полная латентность запроса",
    buckets=_GENERATION_BUCKETS,
    registry=REGISTRY,
)

retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Латентность retrieval",
    buckets=_RETRIEVAL_BUCKETS,
    registry=REGISTRY,
)

generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Латентность генерации ответа",
    buckets=_GENERATION_BUCKETS,
    registry=REGISTRY,
)

retrieved_chunks = Histogram(
    "rag_retrieved_chunks",
    "Сколько фрагментов попало в контекст",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21),
    registry=REGISTRY,
)

documents_indexed_total = Counter(
    "rag_documents_indexed_total",
    "Завершённые индексации документов",
    labelnames=("status",),
    registry=REGISTRY,
)

chunks_indexed_total = Counter(
    "rag_chunks_indexed_total",
    "Записанные в векторное хранилище фрагменты",
    registry=REGISTRY,
)


def record_query(trace_data: dict) -> None:
    """Переносит завершённое событие запроса в метрики.

    Принимает уже собранный dict, а не QueryTrace: метрики не должны быть
    ещё одной причиной менять структуру трассировки, и наоборот.
    """
    if trace_data.get("error"):
        query_errors_total.labels(error_type=_error_type(trace_data["error"])).inc()
    else:
        has_answer = trace_data.get("has_answer")
        queries_total.labels(has_answer=str(bool(has_answer)).lower()).inc()
        if not has_answer:
            no_answer_total.labels(
                reason=trace_data.get("no_answer_code") or "unknown"
            ).inc()

    model = trace_data.get("llm_model")
    if model:
        model_usage_total.labels(
            model=model, provider=trace_data.get("llm_provider") or "unknown"
        ).inc()
    fallback_from = trace_data.get("fallback_from")
    if fallback_from and model:
        model_fallback_total.labels(from_model=fallback_from, to_model=model).inc()

    chunk_ids = trace_data.get("retrieved_chunk_ids")
    if chunk_ids is not None:
        retrieved_chunks.observe(len(chunk_ids))

    _observe_ms(query_latency_seconds, trace_data.get("total_latency_ms"))
    _observe_ms(retrieval_latency_seconds, trace_data.get("retrieval_latency_ms"))
    _observe_ms(generation_latency_seconds, trace_data.get("llm_latency_ms"))


def record_indexing(status: str, chunk_count: int = 0) -> None:
    documents_indexed_total.labels(status=status).inc()
    if chunk_count:
        chunks_indexed_total.inc(chunk_count)


def render() -> tuple[bytes, str]:
    """Отдаёт метрики в текстовом формате Prometheus вместе с content-type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def _observe_ms(histogram: Histogram, value_ms: float | None) -> None:
    if value_ms is not None:
        histogram.observe(value_ms / 1000)


def _error_type(error: str) -> str:
    """Класс ошибки, а не её текст.

    В тексте ошибки бывают имена моделей, пути и идентификаторы — в метке
    это дало бы неограниченную кардинальность.
    """
    lowered = error.lower()
    if "inference" in lowered or "ollama" in lowered or "vllm" in lowered:
        return "inference"
    if "не найден" in lowered or "not found" in lowered:
        return "not_found"
    if "доступ" in lowered or "forbidden" in lowered:
        return "forbidden"
    return "other"
