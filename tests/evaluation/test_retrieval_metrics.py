import math

import pytest

from app.core.errors import ValidationError
from app.evaluation.dataset import EvaluationExample, RelevanceLabel
from app.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_example,
    unique_documents,
)


def make(*labels: RelevanceLabel) -> EvaluationExample:
    return EvaluationExample("q1", "вопрос", "kb-1", labels)


def test_chunks_of_one_document_collapse_preserving_order():
    assert unique_documents(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_precision_counts_a_document_once_even_with_many_chunks():
    example = make(RelevanceLabel("a"))

    # Три чанка одного документа — одно попадание, а не три.
    assert precision_at_k(unique_documents(["a", "a", "a"]), example, k=3) == pytest.approx(1 / 3)


def test_precision_divides_by_k_even_if_output_is_shorter():
    example = make(RelevanceLabel("a"))

    assert precision_at_k(["a"], example, k=5) == pytest.approx(0.2)


def test_recall_is_share_of_found_relevant_documents():
    example = make(RelevanceLabel("a"), RelevanceLabel("b"), RelevanceLabel("c"))

    assert recall_at_k(["a", "x", "b"], example, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_ignores_relevant_documents_below_the_cutoff():
    example = make(RelevanceLabel("a"), RelevanceLabel("b"))

    assert recall_at_k(["x", "a", "b"], example, k=2) == pytest.approx(0.5)


def test_recall_is_undefined_without_relevant_documents():
    with pytest.raises(ValidationError, match="Recall"):
        recall_at_k(["a"], make(), k=3)


def test_reciprocal_rank_uses_the_first_relevant_position():
    example = make(RelevanceLabel("b"))

    assert reciprocal_rank(["x", "y", "b"], example) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_found():
    assert reciprocal_rank(["x", "y"], make(RelevanceLabel("b"))) == 0.0


def test_ndcg_is_one_for_the_ideal_order():
    example = make(RelevanceLabel("a", grade=3), RelevanceLabel("b", grade=1))

    assert ndcg_at_k(["a", "b"], example, k=2) == pytest.approx(1.0)


def test_ndcg_punishes_putting_the_more_relevant_document_lower():
    example = make(RelevanceLabel("a", grade=3), RelevanceLabel("b", grade=1))

    swapped = ndcg_at_k(["b", "a"], example, k=2)

    assert swapped < 1.0
    # Оба документа найдены, значит Recall@2 одинаков — разницу видит только nDCG.
    assert recall_at_k(["b", "a"], example, k=2) == recall_at_k(["a", "b"], example, k=2)


def test_ndcg_matches_the_manual_formula():
    example = make(RelevanceLabel("a", grade=2))

    # Единственный релевантный документ на второй позиции:
    # DCG = (2^2-1)/log2(3), IDCG = (2^2-1)/log2(2).
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert ndcg_at_k(["x", "a"], example, k=3) == pytest.approx(expected)


def test_ndcg_is_zero_without_relevant_documents():
    assert ndcg_at_k(["x"], make(), k=3) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected(k):
    example = make(RelevanceLabel("a"))

    with pytest.raises(ValidationError):
        precision_at_k(["a"], example, k)
    with pytest.raises(ValidationError):
        ndcg_at_k(["a"], example, k)


def test_score_example_deduplicates_before_counting():
    example = make(RelevanceLabel("a"), RelevanceLabel("b"))

    scores = score_example(["a", "a", "b"], example, k=2)

    assert scores.retrieved == ("a", "b")
    assert scores.recall == pytest.approx(1.0)
    assert scores.precision == pytest.approx(1.0)
