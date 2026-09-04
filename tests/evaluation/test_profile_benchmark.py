"""Профиль оценивается по лучшей реально доступной модели своего кольца."""

import pytest

from app.core.errors import InferenceError, ValidationError
from app.evaluation.dataset import EvaluationDataset, EvaluationExample, RelevanceLabel
from app.evaluation.profile_benchmark import ProfileBenchmark
from app.evaluation.rag import AnsweredQuestion
from app.hardware.profiles import HardwareProfile, get_profile_definition
from app.rag.generation import Answer, Citation

SOURCE = "Отпуск составляет 28 календарных дней подряд."
DATASET = EvaluationDataset(
    [EvaluationExample("q1", "q1", "kb-1", (RelevanceLabel("doc"),))]
)


class StubAnswerer:
    def __init__(self, text: str, latency_ms: int = 100) -> None:
        self._text = text
        self._latency_ms = latency_ms

    def answer(self, question: str, knowledge_base_id: str) -> AnsweredQuestion:
        return AnsweredQuestion(
            Answer(
                text=self._text, has_answer=True,
                citations=[
                    Citation(1, "doc-chunk", "doc", None, None, None)
                ],
                latency_ms=self._latency_ms,
            ),
            [SOURCE],
        )


def models_of(profile: HardwareProfile) -> list[str]:
    return [entry.model for entry in get_profile_definition(profile).ring]


def only_available(available: set[str], text: str = "Отпуск составляет 28 дней."):
    def factory(model: str):
        if model not in available:
            raise_error = InferenceError(f"Модель '{model}' не установлена")
            class Failing:
                def answer(self, question, knowledge_base_id):
                    raise raise_error
            return Failing()
        return StubAnswerer(text)

    return factory


def test_profile_without_installed_models_is_unavailable_not_bad():
    light_model = models_of(HardwareProfile.LIGHT)[0]
    benchmark = ProfileBenchmark(
        only_available({light_model}),
        [HardwareProfile.LIGHT, HardwareProfile.PERFORMANCE],
    )

    result = benchmark.run(DATASET)

    light, performance = result.outcomes
    assert light.available is True
    assert performance.available is False
    # Недоступный профиль не получает нулевого качества — у него его просто нет.
    assert performance.best is None
    assert performance.as_dict()["groundedness"] is None
    assert performance.unavailable_models == models_of(HardwareProfile.PERFORMANCE)


def test_best_profile_is_chosen_by_measurement_not_by_weight():
    light_model = models_of(HardwareProfile.LIGHT)[0]
    heavy_model = models_of(HardwareProfile.PERFORMANCE)[0]

    def factory(model: str):
        if model == light_model:
            return StubAnswerer("Отпуск составляет 28 календарных дней.")
        if model == heavy_model:
            return StubAnswerer("Совершенно посторонний придуманный текст.")
        class Failing:
            def answer(self, question, knowledge_base_id):
                raise InferenceError("не установлена")
        return Failing()

    result = ProfileBenchmark(
        factory, [HardwareProfile.LIGHT, HardwareProfile.PERFORMANCE]
    ).run(DATASET)

    # Более тяжёлый профиль не обязан быть лучше на конкретных документах.
    assert result.best_profile().profile is HardwareProfile.LIGHT


def test_best_profile_is_none_when_nothing_is_installed():
    def factory(model: str):
        class Failing:
            def answer(self, question, knowledge_base_id):
                raise InferenceError("не установлена")
        return Failing()

    result = ProfileBenchmark(factory).run(DATASET)

    assert result.best_profile() is None
    assert result.available == []


def test_all_profiles_are_compared_by_default():
    result = ProfileBenchmark(only_available(set())).run(DATASET)

    assert [outcome.profile for outcome in result.outcomes] == list(HardwareProfile)


def test_empty_profile_list_is_rejected():
    with pytest.raises(ValidationError, match="ни одного профиля"):
        ProfileBenchmark(only_available(set()), [])


def test_outcome_serialization_names_the_best_model():
    light_model = models_of(HardwareProfile.LIGHT)[0]

    result = ProfileBenchmark(
        only_available({light_model}), [HardwareProfile.LIGHT]
    ).run(DATASET)

    payload = result.as_dict()["profiles"][0]
    assert payload["profile"] == "light"
    assert payload["best_model"] == light_model
    assert payload["median_latency_ms"] == 100
