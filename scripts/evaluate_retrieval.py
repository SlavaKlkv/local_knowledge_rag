"""Прогон датасета через dense, sparse и hybrid retrieval со сравнением.

Запуск (нужны поднятые Qdrant и локальные embeddings):

    uv run python -m scripts.evaluate_retrieval docs/evaluation/example_dataset.json

Печатает по строке на конфигурацию, поэтому результат сразу читается как
ответ на вопрос «какая стратегия retrieval лучше на этих данных».
"""

from __future__ import annotations

import argparse
import json
import sys

from app.api import dependencies
from app.core.errors import AppError
from app.evaluation.dataset import load_dataset
from app.evaluation.retrieval import RetrievalEvaluator, RetrievalReport


def build_retrievers() -> dict:
    dense = dependencies.get_dense_retriever()
    sparse = dependencies.get_sparse_retriever()
    from app.rag.retriever import HybridRetriever

    return {"dense": dense, "sparse": sparse, "hybrid": HybridRetriever(dense, sparse)}


def format_report(report: RetrievalReport) -> str:
    parts = [f"{report.retriever:<8} MRR={report.mrr:.3f}"]
    for scores in report.by_k.values():
        parts.append(
            f"R@{scores.k}={scores.recall:.3f} "
            f"P@{scores.k}={scores.precision:.3f} "
            f"nDCG@{scores.k}={scores.ndcg:.3f}"
        )
    if report.false_positive_rate is not None:
        parts.append(f"FP={report.false_positive_rate:.3f}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка качества retrieval")
    parser.add_argument("dataset", help="путь к JSON-датасету")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="отсекать кандидатов ниже порога; напрямую влияет на FP по вопросам без ответа",
    )
    parser.add_argument("--json", action="store_true", help="вывести отчёты как JSON")
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
        reports = [
            RetrievalEvaluator(
                retriever,
                k_values=tuple(args.k),
                score_threshold=args.score_threshold,
            ).evaluate(dataset, name=name)
            for name, retriever in build_retrievers().items()
        ]
    except AppError as exc:
        # Локальный inference недоступен — это ошибка, а не повод молча
        # показать «нулевое качество».
        print(f"Оценка не выполнена: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([report.as_dict() for report in reports], ensure_ascii=False, indent=2))
    else:
        print(f"Примеров: {len(dataset)} (с ответом: {len(dataset.answerable)})")
        for report in reports:
            print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
