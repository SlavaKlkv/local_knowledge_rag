"""Сравнение локальных моделей на одном датасете: качество и скорость.

Смысл бенчмарка — не «какая модель лучше вообще», а «какая модель окупается
на этих документах и этом железе». Поэтому качество и скорость показываются
рядом: модель, выигрывающая половину процента обоснованности ценой
четырёхкратной латентности, — плохой выбор для интерактивного поиска, и
таблица должна делать это очевидным.

Отсюда же требование к прогону: все модели идут по одному и тому же датасету
и одному и тому же retrieval, иначе разница в числах говорит о чём угодно,
кроме моделей.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

from app.core.errors import InferenceError, ValidationError
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.rag import QuestionAnswerer, RAGEvaluator, RAGReport


def percentile(values: list[int], share: float) -> int:
    """Перцентиль по ближайшему рангу.

    Интерполяции здесь намеренно нет: на десятке примеров она создаёт
    видимость точности, которой в таком прогоне нет.
    """
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(share * (len(ordered) - 1))))
    return ordered[index]


@dataclass(slots=True, frozen=True)
class BenchmarkRow:
    model: str
    report: RAGReport | None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.report is None

    @property
    def median_latency_ms(self) -> int:
        if self.report is None or not self.report.latencies_ms:
            return 0
        return int(median(self.report.latencies_ms))

    @property
    def p95_latency_ms(self) -> int:
        if self.report is None:
            return 0
        return percentile(list(self.report.latencies_ms), 0.95)

    def as_dict(self) -> dict:
        if self.report is None:
            return {"model": self.model, "error": self.error}
        return {
            "model": self.model,
            **self.report.as_dict(),
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


@dataclass(slots=True)
class BenchmarkResult:
    rows: list[BenchmarkRow]

    @property
    def succeeded(self) -> list[BenchmarkRow]:
        return [row for row in self.rows if not row.failed]

    def ranked_by_quality(self) -> list[BenchmarkRow]:
        """Упорядочивает по обоснованности, при равенстве — по латентности.

        Обоснованность выбрана главным критерием сознательно: система, чей
        ответ не опирается на источник, бесполезна независимо от скорости.
        """
        return sorted(
            self.succeeded,
            key=lambda row: (-(row.report.groundedness), row.median_latency_ms),
        )

    def as_dict(self) -> dict:
        return {"models": [row.as_dict() for row in self.rows]}


class ModelBenchmark:
    def __init__(
        self,
        answerer_factory: Callable[[str], QuestionAnswerer],
        models: list[str],
    ) -> None:
        if not models:
            raise ValidationError("Не задано ни одной модели для сравнения")
        self._factory = answerer_factory
        self._models = models

    def run(self, dataset: EvaluationDataset) -> BenchmarkResult:
        rows: list[BenchmarkRow] = []
        for model in self._models:
            try:
                report = RAGEvaluator(self._factory(model)).evaluate(dataset, name=model)
            except InferenceError as exc:
                # Недоступная модель не должна ронять весь прогон: остальные
                # сравнить всё ещё можно, а пропуск нужно показать честно,
                # а не выдать за нулевое качество.
                rows.append(BenchmarkRow(model=model, report=None, error=str(exc)))
                continue
            rows.append(BenchmarkRow(model=model, report=report))
        return BenchmarkResult(rows=rows)
