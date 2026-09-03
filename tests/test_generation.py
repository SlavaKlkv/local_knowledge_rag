import json

import pytest

from app.core.errors import InferenceError
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.rag.context_builder import ContextBuilder
from app.rag.generation import AnswerGenerator
from app.rag.prompts import NO_ANSWER_TEXT
from app.rag.vector_store import RetrievedChunk


class FakeProvider(LocalLLMProvider):
    name = "fake"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.last_request: GenerationRequest | None = None

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        self.last_request = request
        if isinstance(self.response, Exception):
            raise self.response
        return GenerationResult(
            text=self.response, model=model, provider=self.name, latency_ms=12
        )

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="qwen3:4b", provider=self.name)]


def _context(count: int = 2):
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            document_name=f"doc-{i}.pdf",
            text=f"Фрагмент {i}.",
            score=0.9,
            chunk_index=i,
            page=i + 1,
        )
        for i in range(count)
    ]
    return ContextBuilder().build(chunks)


def test_answer_carries_resolved_citations():
    provider = FakeProvider(
        json.dumps({"answer": "Ответ [1].", "has_answer": True, "citations": [1]})
    )

    answer = AnswerGenerator(provider, "qwen3:4b").generate("вопрос", _context())

    assert answer.has_answer
    assert [c.chunk_id for c in answer.citations] == ["chunk-0"]
    assert answer.citations[0].page == 1
    assert answer.model == "qwen3:4b"


def test_citations_outside_context_are_discarded():
    provider = FakeProvider(
        json.dumps({"answer": "Ответ.", "has_answer": True, "citations": [1, 99, "x"]})
    )

    answer = AnswerGenerator(provider, "qwen3:4b").generate("вопрос", _context())

    assert [c.ref for c in answer.citations] == [1]


def test_empty_context_short_circuits_to_no_answer():
    provider = FakeProvider(RuntimeError("модель не должна вызываться"))

    answer = AnswerGenerator(provider, "qwen3:4b").generate("вопрос", ContextBuilder().build([]))

    assert answer.has_answer is False
    assert answer.text == NO_ANSWER_TEXT


def test_model_signalling_no_answer_is_respected():
    provider = FakeProvider(
        json.dumps({"answer": "", "has_answer": False, "citations": []})
    )

    answer = AnswerGenerator(provider, "qwen3:4b").generate("вопрос", _context())

    assert answer.has_answer is False
    assert answer.text == NO_ANSWER_TEXT


def test_json_wrapped_in_prose_is_still_parsed():
    provider = FakeProvider(
        'Вот ответ: {"answer": "Да [2].", "has_answer": true, "citations": [2]} Готово.'
    )

    answer = AnswerGenerator(provider, "qwen3:4b").generate("вопрос", _context())

    assert answer.text == "Да [2]."
    assert [c.ref for c in answer.citations] == [2]


def test_unparsable_response_becomes_no_answer():
    provider = FakeProvider("модель ответила свободным текстом")

    answer = AnswerGenerator(provider, "qwen3:4b").generate("вопрос", _context())

    assert answer.has_answer is False


def test_inference_error_is_not_swallowed():
    provider = FakeProvider(InferenceError("runtime недоступен"))

    with pytest.raises(InferenceError):
        AnswerGenerator(provider, "qwen3:4b").generate("вопрос", _context())
