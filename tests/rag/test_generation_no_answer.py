"""Генерация уважает политику отказа и объясняет, почему промолчала."""

from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.rag.context_builder import BuiltContext, ContextItem
from app.rag.generation import AnswerGenerator
from app.rag.no_answer import NoAnswerCode
from app.rag.vector_store import RetrievedChunk


class RecordingLLM(LocalLLMProvider):
    name = "fake"

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        self.calls += 1
        return GenerationResult(
            text=self._payload, model=model, provider=self.name, latency_ms=5
        )

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="fake-model", provider=self.name)]


def context(score: float = 0.9) -> BuiltContext:
    chunk = RetrievedChunk(
        chunk_id="c1", document_id="doc", document_name="doc.pdf",
        text="Отпуск составляет 28 дней.", score=score, chunk_index=0,
    )
    return BuiltContext(items=[ContextItem(ref=1, chunk=chunk)], text="[1] ...", token_count=5)


def test_nothing_above_the_threshold_does_not_reach_the_model():
    """Отказ до генерации не только честнее, но и дешевле на секунды инференса."""
    llm = RecordingLLM('{"answer": "Ответ", "has_answer": true, "citations": [1]}')
    generator = AnswerGenerator(llm, model="fake-model")
    empty = BuiltContext(items=[], text="", token_count=0)

    answer = generator.generate("вопрос", empty, threshold_applied=True)

    assert answer.has_answer is False
    assert answer.no_answer_code == NoAnswerCode.BELOW_THRESHOLD
    assert llm.calls == 0


def test_answer_whose_citations_are_all_invalid_becomes_a_refusal():
    """Ссылка на несуществующий фрагмент отбрасывается — и ответ остаётся без опоры."""
    llm = RecordingLLM('{"answer": "Ответ.", "has_answer": true, "citations": [99]}')
    generator = AnswerGenerator(llm, model="fake-model")

    answer = generator.generate("вопрос", context())

    assert answer.has_answer is False
    assert answer.no_answer_code == NoAnswerCode.NO_CITATIONS
    # Модель всё же вызывалась, поэтому её латентность известна и полезна.
    assert answer.latency_ms == 5


def test_valid_answer_passes_through_with_no_reason():
    llm = RecordingLLM('{"answer": "Отпуск 28 дней.", "has_answer": true, "citations": [1]}')

    answer = AnswerGenerator(llm, model="fake-model").generate("вопрос", context())

    assert answer.has_answer is True
    assert answer.no_answer_reason is None
    assert [c.ref for c in answer.citations] == [1]


def test_empty_context_reports_its_own_reason():
    llm = RecordingLLM("{}")
    empty = BuiltContext(items=[], text="", token_count=0)

    answer = AnswerGenerator(llm, model="fake-model").generate("вопрос", empty)

    assert answer.no_answer_code == NoAnswerCode.EMPTY_CONTEXT
    assert llm.calls == 0


def test_unparsable_model_output_is_a_refusal_not_a_crash():
    llm = RecordingLLM("это вообще не json")

    answer = AnswerGenerator(llm, model="fake-model").generate("вопрос", context())

    assert answer.has_answer is False
    assert answer.no_answer_code == NoAnswerCode.MODEL_DECLINED
