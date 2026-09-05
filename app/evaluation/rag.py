"""Прогон полного RAG-пайплайна по датасету и сводный отчёт по ответам.

Evaluator работает не с retriever'ом, а с чем-то, что умеет отвечать на
вопрос, — поэтому одинаково применим и к боевой сборке пайплайна, и к любой
экспериментальной. Тексты процитированных фрагментов возвращаются вместе с
ответом: без них обоснованность проверить не по чему.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol

from app.core.errors import ValidationError
from app.evaluation.answers import AnswerScores, score_answer
from app.evaluation.dataset import EvaluationDataset
from app.rag.context_builder import ContextBuilder
from app.rag.generation import Answer, AnswerGenerator
from app.rag.retriever import RetrievalQuery, Retriever


@dataclass(slots=True, frozen=True)
class AnsweredQuestion:
    answer: Answer
    cited_texts: list[str]


class QuestionAnswerer(Protocol):
    def answer(self, question: str, knowledge_base_id: str) -> AnsweredQuestion: ...


class PipelineAnswerer:
    """Боевая сборка: retrieval → контекст → генерация."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: AnswerGenerator,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator
        self._top_k = top_k
        self._score_threshold = score_threshold

    def answer(self, question: str, knowledge_base_id: str) -> AnsweredQuestion:
        chunks = self._retriever.retrieve(
            RetrievalQuery(
                text=question,
                knowledge_base_id=knowledge_base_id,
                top_k=self._top_k,
                score_threshold=self._score_threshold,
            )
        )
        context = self._context_builder.build(chunks)
        answer = self._generator.generate(question, context)
        # Обоснованность считается по тем фрагментам, на которые ответ
        # сослался, а не по всему контексту: иначе метрика поощряла бы
        # ответ, слова которого нашлись где-то в непроцитированном чанке.
        by_chunk = {item.chunk.chunk_id: item.chunk.text for item in context.items}
        cited_texts = [
            by_chunk[citation.chunk_id]
            for citation in answer.citations
            if citation.chunk_id in by_chunk
        ]
        return AnsweredQuestion(answer=answer, cited_texts=cited_texts)


@dataclass(slots=True)
class RAGReport:
    name: str
    examples_total: int
    answerable_total: int
    unanswerable_total: int
    answer_rate: float
    citation_precision: float
    citation_recall: float
    groundedness: float
    unsupported_numbers_rate: float
    # Латентности всех примеров прогона: агрегаты считает бенчмарк, отчёту
    # достаточно сохранить сырые значения, чтобы не терять хвост распределения.
    latencies_ms: tuple[int, ...] = ()
    correct_abstention_rate: float | None = None
    hallucination_rate: float | None = None
    per_example: list[AnswerScores] = field(default_factory=list)

    def as_dict(self) -> dict:
        report = {
            "name": self.name,
            "examples_total": self.examples_total,
            "answerable_total": self.answerable_total,
            "unanswerable_total": self.unanswerable_total,
            "answer_rate": round(self.answer_rate, 4),
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "groundedness": round(self.groundedness, 4),
            "unsupported_numbers_rate": round(self.unsupported_numbers_rate, 4),
        }
        if self.correct_abstention_rate is not None:
            report["correct_abstention_rate"] = round(self.correct_abstention_rate, 4)
            report["hallucination_rate"] = round(self.hallucination_rate or 0.0, 4)
        return report


class RAGEvaluator:
    def __init__(self, answerer: QuestionAnswerer) -> None:
        self._answerer = answerer

    def evaluate(self, dataset: EvaluationDataset, name: str = "rag") -> RAGReport:
        if len(dataset) == 0:
            raise ValidationError("Датасет пуст: оценивать нечего")

        scores: list[AnswerScores] = []
        latencies: list[int] = []
        for example in dataset:
            answered = self._answerer.answer(example.question, example.knowledge_base_id)
            if answered.answer.latency_ms is not None:
                latencies.append(answered.answer.latency_ms)
            scores.append(
                score_answer(answered.answer, example, cited_texts=answered.cited_texts)
            )

        answerable = [s for s in scores if s.correct_abstention is None]
        unanswerable = [s for s in scores if s.correct_abstention is not None]

        return RAGReport(
            name=name,
            examples_total=len(scores),
            answerable_total=len(answerable),
            unanswerable_total=len(unanswerable),
            answer_rate=_mean(s.answered for s in answerable),
            citation_precision=_mean(s.citation_precision for s in answerable),
            citation_recall=_mean(s.citation_recall for s in answerable),
            groundedness=_mean(s.groundedness for s in answerable),
            unsupported_numbers_rate=_mean(
                bool(s.unsupported_numbers) for s in answerable
            ),
            latencies_ms=tuple(latencies),
            correct_abstention_rate=(
                _mean(s.correct_abstention for s in unanswerable) if unanswerable else None
            ),
            hallucination_rate=(
                _mean(s.hallucinated_answer for s in unanswerable) if unanswerable else None
            ),
            per_example=scores,
        )


def _mean(values) -> float:
    collected = [float(value) for value in values if value is not None]
    return mean(collected) if collected else 0.0
