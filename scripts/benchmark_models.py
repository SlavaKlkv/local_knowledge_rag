"""Сравнение локальных моделей на одном датасете: качество и скорость.

По умолчанию берутся модели кольца активного профиля железа — то есть ровно
те, между которыми система и так переключается в бою:

    uv run python -m scripts.benchmark_models docs/evaluation/example_dataset.json

Список можно задать явно: `--models qwen3:4b gemma3:4b`. Недоступные модели
не роняют прогон — они показываются отдельной строкой с причиной.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.api import dependencies
from app.core.errors import AppError
from app.evaluation.benchmark import BenchmarkResult, ModelBenchmark
from app.evaluation.dataset import load_dataset
from app.evaluation.rag import PipelineAnswerer
from app.hardware.profiles import get_profile_definition
from app.rag.generation import AnswerGenerator


def profile_models() -> list[str]:
    definition = get_profile_definition(dependencies.get_active_profile())
    return [entry.model for entry in definition.ring]


def make_factory(top_k: int, score_threshold: float | None):
    retriever = dependencies.get_retriever()
    context_builder = dependencies.get_context_builder()
    # Базовый провайдер, а не кольцо: кольцо само выбирает модель, и сравнение
    # моделей через него было бы сравнением одного и того же с самим собой.
    provider = dependencies.get_base_llm_provider()

    def factory(model: str) -> PipelineAnswerer:
        return PipelineAnswerer(
            retriever=retriever,
            context_builder=context_builder,
            generator=AnswerGenerator(provider, model=model),
            top_k=top_k,
            score_threshold=score_threshold,
        )

    return factory


def format_result(result: BenchmarkResult) -> str:
    lines = [
        f"{'модель':<22}{'обоснов.':>10}{'цитаты P':>10}{'цитаты R':>10}"
        f"{'отказ':>8}{'медиана':>10}{'p95':>9}"
    ]
    for row in result.ranked_by_quality():
        report = row.report
        abstention = (
            f"{report.correct_abstention_rate:.2f}"
            if report.correct_abstention_rate is not None
            else "—"
        )
        lines.append(
            f"{row.model:<22}{report.groundedness:>10.3f}"
            f"{report.citation_precision:>10.3f}{report.citation_recall:>10.3f}"
            f"{abstention:>8}{row.median_latency_ms:>9} мс{row.p95_latency_ms:>7} мс"
        )
    for row in result.rows:
        if row.failed:
            # Ошибки runtime бывают многострочными и ломают таблицу.
            reason = " ".join((row.error or "").split())
            lines.append(f"{row.model:<22}пропущена: {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Бенчмарк локальных моделей")
    parser.add_argument("dataset", help="путь к JSON-датасету")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
        models = args.models or profile_models()
        benchmark = ModelBenchmark(
            make_factory(args.top_k, args.score_threshold), models
        )
        result = benchmark.run(dataset)
    except AppError as exc:
        print(f"Бенчмарк не выполнен: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Примеров: {len(dataset)} (с ответом: {len(dataset.answerable)})")
        print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
