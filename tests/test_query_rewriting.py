import json

import pytest

from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.rag.query_rewriting import QueryRewriter


class FakeProvider(LocalLLMProvider):
    name = "fake"

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        self.last_prompt = request.prompt
        return GenerationResult(text=self.response, model=model, provider=self.name, latency_ms=1)

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="qwen3:4b", provider=self.name)]


def test_question_without_history_is_returned_unchanged():
    provider = FakeProvider("не должен вызываться")
    rewriter = QueryRewriter(provider, "qwen3:4b")

    result = rewriter.rewrite("Как изменить паспорт сотрудника?", history=None)

    assert result == "Как изменить паспорт сотрудника?"
    assert provider.last_prompt is None


def test_contextual_question_is_rewritten_using_history():
    provider = FakeProvider(
        json.dumps({"rewritten": "Что делать, если паспорт сотрудника просрочен?"})
    )
    rewriter = QueryRewriter(provider, "qwen3:4b")

    result = rewriter.rewrite(
        "А если он просрочен?",
        history="Пользователь: Как изменить паспорт сотрудника?",
    )

    assert result == "Что делать, если паспорт сотрудника просрочен?"
    assert "А если он просрочен?" in provider.last_prompt


def test_json_wrapped_in_prose_is_still_parsed():
    provider = FakeProvider('Конечно: {"rewritten": "Переписанный вопрос."} Готово.')
    rewriter = QueryRewriter(provider, "qwen3:4b")

    result = rewriter.rewrite("вопрос", history="история")

    assert result == "Переписанный вопрос."


@pytest.mark.parametrize("response", ["не json", json.dumps({"rewritten": ""}), "{}"])
def test_unparsable_or_empty_response_falls_back_to_original_question(response):
    provider = FakeProvider(response)
    rewriter = QueryRewriter(provider, "qwen3:4b")

    result = rewriter.rewrite("исходный вопрос", history="история")

    assert result == "исходный вопрос"


def test_blank_history_is_treated_as_no_history():
    provider = FakeProvider("не должен вызываться")
    rewriter = QueryRewriter(provider, "qwen3:4b")

    result = rewriter.rewrite("вопрос", history="   ")

    assert result == "вопрос"
    assert provider.last_prompt is None
