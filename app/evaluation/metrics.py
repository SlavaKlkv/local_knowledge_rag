"""Метрики качества ранжирования.

Все функции работают со списком идентификаторов документов в порядке выдачи
и эталонной разметкой примера — они ничего не знают ни о Qdrant, ни о
retriever, поэтому одинаково применимы к dense, sparse, hybrid и к выдаче
после reranking.

Метрики отвечают на разные вопросы, поэтому считаются вместе:
- Recall@K   — «нашли ли вообще то, что нужно» (главное для RAG: чего нет
               в контексте, того не будет и в ответе);
- Precision@K — «сколько мусора мы кладём в контекст» (мусор вытесняет
               полезное из окна и провоцирует галлюцинации);
- MRR        — «как высоко первый релевантный»;
- nDCG@K     — единственная здесь метрика, учитывающая градацию
               релевантности и позицию одновременно.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.errors import ValidationError
from app.evaluation.dataset import EvaluationExample


def unique_documents(document_ids: list[str]) -> list[str]:
    """Схлопывает чанки одного документа, сохраняя порядок первого вхождения.

    Retrieval возвращает чанки, а размечены документы: без дедупликации три
    чанка одного документа выглядели бы как три попадания и завышали бы
    Precision@K.
    """
    seen: set[str] = set()
    result: list[str] = []
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            result.append(document_id)
    return result


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValidationError("k должен быть положительным")


def precision_at_k(retrieved: list[str], example: EvaluationExample, k: int) -> float:
    _check_k(k)
    top = retrieved[:k]
    hits = sum(1 for document_id in top if document_id in example.relevant_ids)
    # Делим на k, а не на len(top): выдача короче k — это тоже недоработка
    # retrieval, и метрика не должна её маскировать.
    return hits / k


def recall_at_k(retrieved: list[str], example: EvaluationExample, k: int) -> float:
    _check_k(k)
    if not example.relevant_ids:
        raise ValidationError(
            f"Recall не определён для примера '{example.id}' без релевантных документов"
        )
    hits = sum(1 for document_id in retrieved[:k] if document_id in example.relevant_ids)
    return hits / len(example.relevant_ids)


def reciprocal_rank(retrieved: list[str], example: EvaluationExample) -> float:
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in example.relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list[str], example: EvaluationExample, k: int) -> float:
    _check_k(k)
    return sum(
        (2 ** example.grade_of(document_id) - 1) / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
    )


def ndcg_at_k(retrieved: list[str], example: EvaluationExample, k: int) -> float:
    _check_k(k)
    ideal_order = [
        label.document_id
        for label in sorted(example.relevant, key=lambda item: item.grade, reverse=True)
    ]
    ideal = dcg_at_k(ideal_order, example, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, example, k) / ideal


@dataclass(slots=True, frozen=True)
class ExampleScores:
    example_id: str
    retrieved: tuple[str, ...]
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float


def score_example(
    retrieved_document_ids: list[str], example: EvaluationExample, k: int
) -> ExampleScores:
    retrieved = unique_documents(retrieved_document_ids)
    return ExampleScores(
        example_id=example.id,
        retrieved=tuple(retrieved[:k]),
        precision=precision_at_k(retrieved, example, k),
        recall=recall_at_k(retrieved, example, k),
        reciprocal_rank=reciprocal_rank(retrieved, example),
        ndcg=ndcg_at_k(retrieved, example, k),
    )
