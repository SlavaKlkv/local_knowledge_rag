"""Сравнение профилей железа по фактическому качеству их колец.

Профили LIGHT/STANDARD/PERFORMANCE описаны через требования к памяти и состав
кольца, но само по себе это ничего не говорит о том, что пользователь получит
на своей машине. Здесь профиль оценивается по его лучшей реально доступной
модели: профиль, все модели которого не установлены, — это не «нулевое
качество», а недоступный профиль, и показывать его надо именно так.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.core.errors import ValidationError
from app.evaluation.benchmark import BenchmarkResult, BenchmarkRow, ModelBenchmark
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.rag import QuestionAnswerer
from app.hardware.profiles import HardwareProfile, get_profile_definition


@dataclass(slots=True, frozen=True)
class ProfileOutcome:
    profile: HardwareProfile
    result: BenchmarkResult

    @property
    def available(self) -> bool:
        return bool(self.result.succeeded)

    @property
    def best(self) -> BenchmarkRow | None:
        ranked = self.result.ranked_by_quality()
        return ranked[0] if ranked else None

    @property
    def unavailable_models(self) -> list[str]:
        return [row.model for row in self.result.rows if row.failed]

    def as_dict(self) -> dict:
        best = self.best
        return {
            "profile": str(self.profile),
            "available": self.available,
            "best_model": best.model if best else None,
            "groundedness": round(best.report.groundedness, 4) if best else None,
            "median_latency_ms": best.median_latency_ms if best else None,
            "unavailable_models": self.unavailable_models,
        }


@dataclass(slots=True)
class ProfileBenchmarkResult:
    outcomes: list[ProfileOutcome]

    @property
    def available(self) -> list[ProfileOutcome]:
        return [outcome for outcome in self.outcomes if outcome.available]

    def best_profile(self) -> ProfileOutcome | None:
        """Профиль с самой обоснованной моделью среди доступных.

        Сознательно не «самый тяжёлый из доступных»: смысл сравнения в том,
        что более тяжёлый профиль не обязан оказаться лучше на конкретных
        документах, и решать это должны измерения, а не таблица требований.
        """
        candidates = [
            outcome for outcome in self.available if outcome.best is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda outcome: outcome.best.report.groundedness)

    def as_dict(self) -> dict:
        return {"profiles": [outcome.as_dict() for outcome in self.outcomes]}


class ProfileBenchmark:
    def __init__(
        self,
        answerer_factory: Callable[[str], QuestionAnswerer],
        profiles: list[HardwareProfile] | None = None,
    ) -> None:
        self._factory = answerer_factory
        # None означает «все профили», пустой список — ошибку вызова:
        # `profiles or list(...)` смешивал эти два случая и молча сравнивал всё.
        if profiles is None:
            profiles = list(HardwareProfile)
        if not profiles:
            raise ValidationError("Не задано ни одного профиля для сравнения")
        self._factory = answerer_factory
        self._profiles = profiles

    def run(self, dataset: EvaluationDataset) -> ProfileBenchmarkResult:
        outcomes: list[ProfileOutcome] = []
        for profile in self._profiles:
            models = [entry.model for entry in get_profile_definition(profile).ring]
            result = ModelBenchmark(self._factory, models).run(dataset)
            outcomes.append(ProfileOutcome(profile=profile, result=result))
        return ProfileBenchmarkResult(outcomes=outcomes)
