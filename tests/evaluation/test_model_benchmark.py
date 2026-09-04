"""Бенчмарк сравнивает модели на одном датасете: качество рядом со скоростью."""

import pytest

from app.core.errors import InferenceError, ValidationError
from app.evaluation.benchmark import ModelBenchmark, percentile
from app.evaluation.dataset import EvaluationDataset, EvaluationExample, RelevanceLabel
from app.evaluation.rag import AnsweredQuestion
from app.rag.generation import Answer, Citation


def example(question: str, *documents: str) -> EvaluationExample:
    return EvaluationExample(
        question, question, "kb-1", tuple(RelevanceLabel(d) for d in documents)
    )


def citation(document_id: str) -> Citation:
    return Citation(
        ref=1, chunk_id=f"{document_id}-chunk", document_id=document_id,
        document_name=None, page=None, section=None,
    )


class StubAnswerer:
    def __init__(self, text: str, latency_ms: int, source: str) -> None:
        self._text = text
        self._latency_ms = latency_ms
        self._source = source

    def answer(self, question: str, knowledge_base_id: str) -> AnsweredQuestion:
        return AnsweredQuestion(
            Answer(
                text=self._text, has_answer=True, citations=[citation("doc")],
                latency_ms=self._latency_ms,
            ),
            [self._source],
        )


class FailingAnswerer:
    def answer(self, question: str, knowledge_base_id: str) -> AnsweredQuestion:
        raise InferenceError("Модель не установлена")


SOURCE = "Отпуск составляет 28 календарных дней подряд."
DATASET = EvaluationDataset([example("q1", "doc"), example("q2", "doc")])


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.0, 10), (0.5, 30), (0.95, 50), (1.0, 50)],
)
def test_percentile_uses_the_nearest_rank(share, expected):
    assert percentile([30, 10, 50, 20, 40], share) == expected


def test_percentile_of_an_empty_sample_is_zero():
    assert percentile([], 0.95) == 0


def test_benchmark_reports_quality_and_latency_side_by_side():
    def factory(model: str):
        return {
            "быстрая": StubAnswerer(
                "Отпуск составляет 28 дней, компенсация начисляется автоматически.",
                100,
                SOURCE,
            ),
            "медленная": StubAnswerer("Отпуск составляет 28 календарных дней.", 900, SOURCE),
        }[model]

    result = ModelBenchmark(factory, ["быстрая", "медленная"]).run(DATASET)

    rows = {row.model: row for row in result.rows}
    assert rows["быстрая"].median_latency_ms == 100
    assert rows["медленная"].median_latency_ms == 900
    assert rows["медленная"].report.groundedness > rows["быстрая"].report.groundedness


def test_ranking_puts_the_most_grounded_model_first():
    def factory(model: str):
        return {
            "слабая": StubAnswerer("Совершенно посторонний придуманный текст.", 50, SOURCE),
            "сильная": StubAnswerer("Отпуск составляет 28 календарных дней.", 800, SOURCE),
        }[model]

    result = ModelBenchmark(factory, ["слабая", "сильная"]).run(DATASET)

    # Скорость не искупает неопирающийся на источник ответ.
    assert [row.model for row in result.ranked_by_quality()] == ["сильная", "слабая"]


def test_equal_quality_is_broken_by_latency():
    def factory(model: str):
        latency = {"быстрая": 100, "медленная": 900}[model]
        return StubAnswerer("Отпуск составляет 28 дней.", latency, SOURCE)

    result = ModelBenchmark(factory, ["медленная", "быстрая"]).run(DATASET)

    assert [row.model for row in result.ranked_by_quality()] == ["быстрая", "медленная"]


def test_unavailable_model_is_skipped_without_failing_the_whole_run():
    def factory(model: str):
        return FailingAnswerer() if model == "нет-такой" else StubAnswerer(
            "Отпуск составляет 28 дней.", 100, SOURCE
        )

    result = ModelBenchmark(factory, ["нет-такой", "рабочая"]).run(DATASET)

    failed = next(row for row in result.rows if row.model == "нет-такой")
    assert failed.failed is True
    # Пропуск показан честно, а не выдан за нулевое качество.
    assert failed.report is None
    assert "не установлена" in failed.error
    assert failed.as_dict() == {"model": "нет-такой", "error": failed.error}
    assert [row.model for row in result.succeeded] == ["рабочая"]


def test_failed_models_are_absent_from_the_ranking():
    def factory(model: str):
        return FailingAnswerer()

    result = ModelBenchmark(factory, ["нет-1", "нет-2"]).run(DATASET)

    assert result.ranked_by_quality() == []


def test_benchmark_without_models_is_rejected():
    with pytest.raises(ValidationError, match="ни одной модели"):
        ModelBenchmark(lambda model: StubAnswerer("текст", 1, SOURCE), [])


def test_row_serialization_carries_both_quality_and_latency():
    benchmark = ModelBenchmark(
        lambda model: StubAnswerer("Отпуск составляет 28 дней.", 250, SOURCE), ["модель"]
    )

    row = benchmark.run(DATASET).rows[0].as_dict()

    assert row["model"] == "модель"
    assert row["median_latency_ms"] == 250
    assert row["p95_latency_ms"] == 250
    assert "groundedness" in row
