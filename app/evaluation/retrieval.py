"""Прогон retrieval по размеченному датасету и сводный отчёт.

Evaluator принимает любой объект с методом `retrieve` (Retriever), поэтому
одним и тем же кодом сравниваются dense, sparse и hybrid — и любая из них
с включённым reranking. Именно это сравнение, а не абсолютные числа, и есть
рабочий инструмент: «стало лучше или хуже после изменения» проверяется
одинаковым прогоном на одном датасете.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from app.core.errors import ValidationError
from app.evaluation.dataset import EvaluationDataset, EvaluationExample
from app.evaluation.metrics import ExampleScores, reciprocal_rank, score_example, unique_documents
from app.rag.retriever import RetrievalQuery, Retriever


@dataclass(slots=True, frozen=True)
class AggregateScores:
    k: int
    precision: float
    recall: float
    ndcg: float

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "precision_at_k": round(self.precision, 4),
            "recall_at_k": round(self.recall, 4),
            "ndcg_at_k": round(self.ndcg, 4),
        }


@dataclass(slots=True)
class RetrievalReport:
    retriever: str
    examples_total: int
    answerable_total: int
    unanswerable_total: int
    mrr: float
    by_k: dict[int, AggregateScores] = field(default_factory=dict)
    per_example: list[ExampleScores] = field(default_factory=list)
    false_positive_rate: float | None = None

    def as_dict(self) -> dict:
        report = {
            "retriever": self.retriever,
            "examples_total": self.examples_total,
            "answerable_total": self.answerable_total,
            "unanswerable_total": self.unanswerable_total,
            "mrr": round(self.mrr, 4),
            "by_k": [scores.as_dict() for scores in self.by_k.values()],
        }
        if self.false_positive_rate is not None:
            report["false_positive_rate"] = round(self.false_positive_rate, 4)
        return report


class RetrievalEvaluator:
    def __init__(
        self,
        retriever: Retriever,
        k_values: tuple[int, ...] = (1, 3, 5, 10),
        score_threshold: float | None = None,
    ) -> None:
        if not k_values:
            raise ValidationError("Нужно хотя бы одно значение k")
        if any(k <= 0 for k in k_values):
            raise ValidationError("Все значения k должны быть положительными")
        self._retriever = retriever
        self._k_values = tuple(sorted(set(k_values)))
        self._score_threshold = score_threshold

    def evaluate(self, dataset: EvaluationDataset, name: str = "retriever") -> RetrievalReport:
        if len(dataset) == 0:
            raise ValidationError("Датасет пуст: оценивать нечего")

        # Retrieval выполняется один раз на пример, с самым большим k;
        # метрики для меньших k считаются срезами того же списка. Иначе
        # один и тот же вопрос гонялся бы через embeddings четырежды.
        max_k = self._k_values[-1]
        retrieved_by_example: dict[str, list[str]] = {}
        for example in dataset:
            retrieved_by_example[example.id] = self._retrieve(example, max_k)

        by_k: dict[int, AggregateScores] = {}
        per_example: list[ExampleScores] = []
        answerable = dataset.answerable
        for k in self._k_values:
            scores = [
                score_example(retrieved_by_example[example.id], example, k)
                for example in answerable
            ]
            if k == max_k:
                per_example = scores
            by_k[k] = AggregateScores(
                k=k,
                precision=mean(s.precision for s in scores) if scores else 0.0,
                recall=mean(s.recall for s in scores) if scores else 0.0,
                ndcg=mean(s.ndcg for s in scores) if scores else 0.0,
            )

        mrr = (
            mean(
                reciprocal_rank(unique_documents(retrieved_by_example[example.id]), example)
                for example in answerable
            )
            if answerable
            else 0.0
        )

        unanswerable = dataset.unanswerable
        false_positive_rate = (
            mean(1.0 if retrieved_by_example[example.id] else 0.0 for example in unanswerable)
            if unanswerable
            else None
        )

        return RetrievalReport(
            retriever=name,
            examples_total=len(dataset),
            answerable_total=len(answerable),
            unanswerable_total=len(unanswerable),
            mrr=mrr,
            by_k=by_k,
            per_example=per_example,
            false_positive_rate=false_positive_rate,
        )

    def _retrieve(self, example: EvaluationExample, top_k: int) -> list[str]:
        chunks = self._retriever.retrieve(
            RetrievalQuery(
                text=example.question,
                knowledge_base_id=example.knowledge_base_id,
                top_k=top_k,
                score_threshold=self._score_threshold,
            )
        )
        return [chunk.document_id for chunk in chunks]
