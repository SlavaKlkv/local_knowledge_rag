"""Прогон полного пайплайна по датасету и агрегация метрик ответа."""

import pytest

from app.core.errors import ValidationError
from app.evaluation.dataset import EvaluationDataset, EvaluationExample, RelevanceLabel
from app.evaluation.rag import AnsweredQuestion, PipelineAnswerer, RAGEvaluator
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.rag.context_builder import ContextBuilder
from app.rag.generation import Answer, AnswerGenerator, Citation
from app.rag.vector_store import RetrievedChunk


class ScriptedAnswerer:
    def __init__(self, answers: dict[str, AnsweredQuestion]) -> None:
        self._answers = answers

    def answer(self, question: str, knowledge_base_id: str) -> AnsweredQuestion:
        return self._answers[question]


def citation(document_id: str, ref: int = 1) -> Citation:
    return Citation(
        ref=ref, chunk_id=f"{document_id}-chunk", document_id=document_id,
        document_name=None, page=None, section=None,
    )


def example(question: str, *documents: str) -> EvaluationExample:
    return EvaluationExample(
        question, question, "kb-1", tuple(RelevanceLabel(d) for d in documents)
    )


def test_report_separates_answerable_and_unanswerable_questions():
    data = EvaluationDataset([example("есть", "doc"), example("нет")])
    answerer = ScriptedAnswerer({
        "есть": AnsweredQuestion(
            Answer(text="Ответ 28 дней.", has_answer=True, citations=[citation("doc")]),
            ["Ответ 28 дней."],
        ),
        "нет": AnsweredQuestion(Answer(text="Нет ответа", has_answer=False), []),
    })

    report = RAGEvaluator(answerer).evaluate(data)

    assert report.answerable_total == 1
    assert report.unanswerable_total == 1
    assert report.answer_rate == pytest.approx(1.0)
    assert report.correct_abstention_rate == pytest.approx(1.0)
    assert report.hallucination_rate == pytest.approx(0.0)


def test_hallucination_rate_counts_answers_to_unanswerable_questions():
    data = EvaluationDataset([example("нет-1"), example("нет-2")])
    answerer = ScriptedAnswerer({
        "нет-1": AnsweredQuestion(Answer(text="Придумал.", has_answer=True), []),
        "нет-2": AnsweredQuestion(Answer(text="Нет ответа", has_answer=False), []),
    })

    report = RAGEvaluator(answerer).evaluate(data)

    assert report.hallucination_rate == pytest.approx(0.5)
    assert report.correct_abstention_rate == pytest.approx(0.5)


def test_abstention_metrics_are_absent_without_unanswerable_examples():
    data = EvaluationDataset([example("есть", "doc")])
    answerer = ScriptedAnswerer({
        "есть": AnsweredQuestion(
            Answer(text="Ответ.", has_answer=True, citations=[citation("doc")]), ["Ответ."]
        )
    })

    report = RAGEvaluator(answerer).evaluate(data)

    assert report.correct_abstention_rate is None
    assert "correct_abstention_rate" not in report.as_dict()


def test_unsupported_numbers_rate_counts_examples_not_numbers():
    data = EvaluationDataset([example("q1", "doc"), example("q2", "doc")])
    answerer = ScriptedAnswerer({
        "q1": AnsweredQuestion(
            Answer(text="Срок 14 и 3 дня.", has_answer=True, citations=[citation("doc")]),
            ["Срок 28 дней."],
        ),
        "q2": AnsweredQuestion(
            Answer(text="Срок 28 дней.", has_answer=True, citations=[citation("doc")]),
            ["Срок 28 дней."],
        ),
    })

    report = RAGEvaluator(answerer).evaluate(data)

    # Два выдуманных числа в одном примере из двух — это 0.5, а не 1.0.
    assert report.unsupported_numbers_rate == pytest.approx(0.5)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValidationError, match="пуст"):
        RAGEvaluator(ScriptedAnswerer({})).evaluate(EvaluationDataset([]))


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query):
        return self._chunks


class FakeLLM(LocalLLMProvider):
    name = "fake"

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        return GenerationResult(
            text=self._payload, model=model, provider=self.name, latency_ms=1
        )

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="fake-model", provider=self.name)]


def chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id="doc", document_name="doc.pdf",
        text=text, score=0.9, chunk_index=0,
    )


def test_pipeline_answerer_returns_only_the_cited_chunk_texts():
    """Обоснованность считается по процитированному, а не по всему контексту.

    Иначе ответ, слова которого случайно нашлись в непроцитированном чанке,
    выглядел бы обоснованным.
    """
    retriever = FakeRetriever([
        chunk("c1", "Отпуск составляет 28 календарных дней."),
        chunk("c2", "Совершенно посторонний фрагмент про парковку."),
    ])
    llm = FakeLLM('{"answer": "Отпуск 28 дней.", "has_answer": true, "citations": [1]}')
    answerer = PipelineAnswerer(
        retriever, ContextBuilder(), AnswerGenerator(llm, model="fake-model")
    )

    answered = answerer.answer("Сколько дней отпуска?", "kb-1")

    assert answered.answer.has_answer is True
    assert answered.cited_texts == ["Отпуск составляет 28 календарных дней."]


def test_pipeline_answerer_abstains_when_retrieval_returns_nothing():
    answerer = PipelineAnswerer(
        FakeRetriever([]),
        ContextBuilder(),
        AnswerGenerator(FakeLLM("{}"), model="fake-model"),
    )

    answered = answerer.answer("Что-нибудь?", "kb-1")

    assert answered.answer.has_answer is False
    assert answered.cited_texts == []
