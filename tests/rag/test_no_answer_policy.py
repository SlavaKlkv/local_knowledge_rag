"""Политика честного отказа: когда система обязана промолчать."""

import pytest

from app.rag.context_builder import BuiltContext, ContextItem
from app.rag.no_answer import NoAnswerCode, NoAnswerPolicy
from app.rag.vector_store import RetrievedChunk


def chunk(score: float, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id="doc", document_name="doc.pdf",
        text="фрагмент", score=score, chunk_index=0,
    )


def context(*scores: float) -> BuiltContext:
    items = [
        ContextItem(ref=index + 1, chunk=chunk(score, f"c{index}"))
        for index, score in enumerate(scores)
    ]
    return BuiltContext(items=items, text="контекст", token_count=10)


def empty_context() -> BuiltContext:
    return BuiltContext(items=[], text="", token_count=0)


def test_empty_context_always_refuses():
    decision = NoAnswerPolicy().before_generation(empty_context())

    assert decision.refuse is True
    assert decision.code is NoAnswerCode.EMPTY_CONTEXT


def test_policy_does_not_judge_scores_itself():
    """Скор чанка живёт в разных шкалах: у dense косинус, у hybrid уже RRF.

    Сравнивать их с одним порогом внутри политики значит молча врать, поэтому
    отсечение по скору выполняет retrieval, а политика на скор не смотрит:
    низкий, но непустой контекст она пропускает.
    """
    assert not hasattr(NoAnswerPolicy(), "min_retrieval_score")
    assert NoAnswerPolicy().before_generation(context(0.016)).refuse is False


def test_empty_result_with_a_threshold_is_reported_differently():
    """«Слишком строгий порог» и «в базе ничего нет» — разные ситуации."""
    without = NoAnswerPolicy().before_generation(empty_context())
    with_threshold = NoAnswerPolicy().before_generation(
        empty_context(), threshold_applied=True
    )

    assert without.code is NoAnswerCode.EMPTY_CONTEXT
    assert with_threshold.code is NoAnswerCode.BELOW_THRESHOLD


def test_threshold_flag_does_not_affect_a_non_empty_context():
    decision = NoAnswerPolicy().before_generation(context(0.9), threshold_applied=True)

    assert decision.refuse is False


def test_model_declining_is_respected():
    decision = NoAnswerPolicy().after_generation(has_answer=False, citation_count=3)

    assert decision.refuse is True
    assert decision.code is NoAnswerCode.MODEL_DECLINED


def test_answer_without_citations_is_refused_by_default():
    """Ответ, не сославшийся ни на один фрагмент, проверить нечем."""
    decision = NoAnswerPolicy().after_generation(has_answer=True, citation_count=0)

    assert decision.refuse is True
    assert decision.code is NoAnswerCode.NO_CITATIONS


def test_citation_requirement_can_be_switched_off():
    policy = NoAnswerPolicy(require_citations=False)

    assert policy.after_generation(has_answer=True, citation_count=0).refuse is False


def test_answer_with_citations_passes():
    decision = NoAnswerPolicy().after_generation(has_answer=True, citation_count=2)

    assert decision.refuse is False
    assert decision.code is None


@pytest.mark.parametrize("code", list(NoAnswerCode))
def test_reason_codes_are_a_bounded_set(code):
    """Метка Prometheus строится из кода, поэтому набор обязан быть конечным."""
    assert isinstance(str(code), str)
    assert " " not in str(code)
