"""Прогон полного RAG-пайплайна по датасету: обоснованность, цитаты, отказы.

Запуск (нужны поднятый Qdrant, локальные embeddings и локальная LLM):

    uv run python -m scripts.evaluate_rag docs/evaluation/example_dataset.json

В отличие от `evaluate_retrieval`, здесь на каждый пример выполняется
генерация, поэтому прогон заметно дольше и упирается в скорость локальной
модели.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.api import dependencies
from app.core.errors import AppError
from app.evaluation.dataset import load_dataset
from app.evaluation.rag import PipelineAnswerer, RAGEvaluator, RAGReport


def build_answerer(top_k: int, score_threshold: float | None, model: str | None):
    generator = dependencies.get_answer_generator()
    if model:
        from app.rag.generation import AnswerGenerator

        # Базовый провайдер, а не кольцо: кольцо само выбирает модель из
        # профиля и переопределение молча проигнорировало бы.
        generator = AnswerGenerator(dependencies.get_base_llm_provider(), model=model)
    return PipelineAnswerer(
        retriever=dependencies.get_retriever(),
        context_builder=dependencies.get_context_builder(),
        generator=generator,
        top_k=top_k,
        score_threshold=score_threshold,
    )


def format_report(report: RAGReport) -> str:
    parts = [
        f"{report.name:<10}",
        f"ответов={report.answer_rate:.3f}",
        f"цитаты P={report.citation_precision:.3f} R={report.citation_recall:.3f}",
        f"обоснованность={report.groundedness:.3f}",
        f"выдуманные числа={report.unsupported_numbers_rate:.3f}",
    ]
    if report.correct_abstention_rate is not None:
        parts.append(f"верный отказ={report.correct_abstention_rate:.3f}")
        parts.append(f"выдумки={report.hallucination_rate:.3f}")
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка качества ответов RAG")
    parser.add_argument("dataset", help="путь к JSON-датасету")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--model", default=None, help="переопределить модель генерации")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
        answerer = build_answerer(args.top_k, args.score_threshold, args.model)
        report = RAGEvaluator(answerer).evaluate(dataset, name=args.model or "rag")
    except AppError as exc:
        # Локальный inference недоступен — честная ошибка, а не «нулевое качество».
        print(f"Оценка не выполнена: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Примеров: {len(dataset)} (с ответом: {len(dataset.answerable)})")
        print(format_report(report))
        for scores in report.per_example:
            if scores.unsupported_numbers:
                print(
                    f"  ! {scores.example_id}: чисел нет в источнике — "
                    f"{', '.join(scores.unsupported_numbers)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
