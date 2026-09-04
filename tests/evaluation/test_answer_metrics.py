"""Метрики ответа ловят выдумывание, а не непохожесть на эталон."""

import pytest

from app.evaluation.answers import (
    lexical_groundedness,
    score_answer,
    unsupported_numbers,
)
from app.evaluation.dataset import EvaluationExample, RelevanceLabel
from app.rag.generation import Answer, Citation


def citation(document_id: str, ref: int = 1) -> Citation:
    return Citation(
        ref=ref,
        chunk_id=f"{document_id}-chunk",
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        page=None,
        section=None,
    )


def answerable(*documents: str) -> EvaluationExample:
    return EvaluationExample(
        "q1", "Сколько дней отпуска?", "kb-1",
        tuple(RelevanceLabel(d) for d in documents),
    )


def unanswerable() -> EvaluationExample:
    return EvaluationExample("q-no", "Есть ли парковка?", "kb-1")


def test_groundedness_is_one_when_the_answer_reuses_the_source():
    assert lexical_groundedness(
        "Отпуск составляет 28 календарных дней.",
        ["Основной ежегодный отпуск составляет 28 календарных дней."],
    ) == pytest.approx(1.0)


def test_groundedness_drops_when_the_answer_adds_its_own_words():
    grounded = lexical_groundedness("отпуск составляет", ["отпуск составляет"])
    invented = lexical_groundedness(
        "отпуск составляет, компенсация начисляется автоматически",
        ["отпуск составляет"],
    )

    assert invented < grounded


def test_groundedness_ignores_short_function_words():
    """Совпадение предлогов не должно выглядеть как опора на источник."""
    assert lexical_groundedness("но и за то", ["совершенно другой текст"]) == 0.0


def test_empty_answer_is_not_grounded():
    assert lexical_groundedness("", ["источник"]) == 0.0


def test_numbers_absent_from_the_source_are_reported():
    found = unsupported_numbers(
        "Отпуск 14 дней, отчёт через 3 дня.", ["Отпуск составляет 28 дней."]
    )

    assert found == {"14", "3"}


def test_numbers_present_in_the_source_are_not_reported():
    assert unsupported_numbers("28 дней", ["ровно 28 календарных дней"]) == set()


def test_decimal_separators_do_not_create_false_positives():
    assert unsupported_numbers("ставка 1,5", ["ставка 1.5"]) == set()


def test_citation_precision_punishes_citing_irrelevant_documents():
    answer = Answer(
        text="Ответ.", has_answer=True,
        citations=[citation("vacation"), citation("trips", ref=2)],
    )

    scores = score_answer(answer, answerable("vacation"), cited_texts=["источник"])

    assert scores.citation_precision == pytest.approx(0.5)
    assert scores.citation_recall == pytest.approx(1.0)


def test_citation_recall_punishes_missing_a_required_document():
    answer = Answer(text="Ответ.", has_answer=True, citations=[citation("vacation")])

    scores = score_answer(answer, answerable("vacation", "trips"), cited_texts=["источник"])

    assert scores.citation_recall == pytest.approx(0.5)
    assert scores.citation_precision == pytest.approx(1.0)


def test_answer_without_citations_scores_zero_precision():
    answer = Answer(text="Ответ без ссылок.", has_answer=True)

    scores = score_answer(answer, answerable("vacation"), cited_texts=[])

    assert scores.citation_precision == 0.0
    assert scores.citation_recall == 0.0


def test_refusal_where_an_answer_exists_is_a_miss_but_not_a_hallucination():
    scores = score_answer(Answer(text="Нет ответа", has_answer=False), answerable("vacation"))

    assert scores.answered is False
    assert scores.hallucinated_answer is False
    assert scores.citation_precision == 0.0
    assert scores.groundedness == 0.0


def test_refusal_on_an_unanswerable_question_is_correct_abstention():
    scores = score_answer(Answer(text="Нет ответа", has_answer=False), unanswerable())

    assert scores.correct_abstention is True
    assert scores.hallucinated_answer is False


def test_answering_an_unanswerable_question_is_a_hallucination():
    answer = Answer(text="Да, компенсирует.", has_answer=True, citations=[citation("trips")])

    scores = score_answer(answer, unanswerable(), cited_texts=["источник"])

    assert scores.correct_abstention is False
    assert scores.hallucinated_answer is True
    # Метрики цитат для вопроса без ответа смысла не имеют.
    assert scores.citation_precision is None


def test_unsupported_numbers_are_attached_to_the_answer_score():
    answer = Answer(text="Отпуск 14 дней.", has_answer=True, citations=[citation("vacation")])

    scores = score_answer(
        answer, answerable("vacation"), cited_texts=["Отпуск составляет 28 дней."]
    )

    assert scores.unsupported_numbers == ("14",)
