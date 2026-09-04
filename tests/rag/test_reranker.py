import pytest

from app.core.errors import InferenceError
from app.rag.reranker import CrossEncoderReranker, NoOpReranker
from app.rag.vector_store import RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, document_id="doc-1", document_name="doc.pdf",
        text=text, score=score, chunk_index=0,
    )


class FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def test_noop_reranker_preserves_order_and_uses_retrieval_score():
    candidates = [_chunk("a", "первый", 0.9), _chunk("b", "второй", 0.5)]

    reranked = NoOpReranker().rerank("вопрос", candidates)

    assert [r.chunk.chunk_id for r in reranked] == ["a", "b"]
    assert [r.rerank_score for r in reranked] == [0.9, 0.5]


def test_noop_reranker_respects_top_k():
    candidates = [_chunk(str(i), f"текст {i}") for i in range(5)]

    reranked = NoOpReranker().rerank("вопрос", candidates, top_k=2)

    assert len(reranked) == 2


def test_cross_encoder_reorders_by_model_score(monkeypatch):
    candidates = [_chunk("a", "слабое совпадение"), _chunk("b", "точное совпадение")]
    reranker = CrossEncoderReranker()
    fake_model = FakeModel(scores=[0.1, 0.9])
    monkeypatch.setattr(reranker, "_load", lambda: fake_model)

    reranked = reranker.rerank("вопрос", candidates)

    assert [r.chunk.chunk_id for r in reranked] == ["b", "a"]
    assert fake_model.pairs == [("вопрос", "слабое совпадение"), ("вопрос", "точное совпадение")]


def test_cross_encoder_respects_top_k(monkeypatch):
    candidates = [_chunk(str(i), f"текст {i}") for i in range(5)]
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load", lambda: FakeModel(scores=[0.1, 0.5, 0.9, 0.3, 0.7]))

    reranked = reranker.rerank("вопрос", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in reranked] == ["2", "4"]


def test_empty_candidates_short_circuit_without_loading_model(monkeypatch):
    reranker = CrossEncoderReranker()

    def fail_load():  # pragma: no cover - не должен вызываться
        raise AssertionError("модель не должна загружаться для пустого списка")

    monkeypatch.setattr(reranker, "_load", fail_load)

    assert reranker.rerank("вопрос", []) == []


def test_missing_dependency_raises_inference_error(monkeypatch):
    reranker = CrossEncoderReranker()

    def broken_import():
        raise ImportError("no module named sentence_transformers")

    monkeypatch.setattr(
        "app.rag.reranker.CrossEncoderReranker._load",
        lambda self: (_ for _ in ()).throw(InferenceError("sentence-transformers не установлен")),
    )

    with pytest.raises(InferenceError, match="не установлен"):
        reranker.rerank("вопрос", [_chunk("a", "текст")])


def test_health_check_reflects_model_availability(monkeypatch):
    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_load", lambda: FakeModel(scores=[]))
    assert reranker.health_check() is True

    def failing_load():
        raise InferenceError("модель недоступна")

    monkeypatch.setattr(reranker, "_load", failing_load)
    assert reranker.health_check() is False
