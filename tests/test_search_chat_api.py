import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.core.errors import InferenceError
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.main import create_app
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import RetrievedChunk


class FakeEmbeddings(EmbeddingProvider):
    @property
    def model(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def health_check(self) -> bool:
        return True


class FakeVectorStore:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits

    def search(self, **kwargs):
        return self.hits


class FakeLLM(LocalLLMProvider):
    name = "fake"

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        return GenerationResult(text=self.response, model=model, provider=self.name, latency_ms=5)

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="qwen3:4b", provider=self.name)]


def _hit(chunk_id="c1", text="Отпуск предоставляется ежегодно."):
    return RetrievedChunk(
        chunk_id=chunk_id, document_id="doc-1", document_name="policy.pdf",
        text=text, score=0.9, chunk_index=0, page=1,
    )


@pytest.fixture
def client():
    return TestClient(create_app())


def test_search_returns_hits_with_citation_metadata(client, monkeypatch):
    monkeypatch.setattr(
        dependencies, "get_embedding_provider", lambda: FakeEmbeddings()
    )
    monkeypatch.setattr(
        dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()])
    )
    client.app.dependency_overrides.clear()

    response = client.post(
        "/search",
        json={"query": "отпуск", "knowledge_base_id": str(uuid.uuid4())},
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits[0]["document_name"] == "policy.pdf"
    assert hits[0]["page"] == 1


def test_chat_returns_grounded_answer_with_citations(client, monkeypatch):
    import json

    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()]))
    monkeypatch.setattr(
        dependencies,
        "get_llm_provider",
        lambda: FakeLLM(
            json.dumps({"answer": "Ежегодно [1].", "has_answer": True, "citations": [1]})
        ),
    )

    response = client.post(
        "/chat",
        json={"question": "Когда предоставляется отпуск?", "knowledge_base_id": str(uuid.uuid4())},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_answer"] is True
    assert body["citations"][0]["document_name"] == "policy.pdf"


def test_chat_returns_no_answer_when_nothing_is_retrieved(client, monkeypatch):
    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([]))

    response = client.post(
        "/chat",
        json={"question": "Вопрос без ответа", "knowledge_base_id": str(uuid.uuid4())},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_answer"] is False
    assert body["citations"] == []


def test_inference_error_returns_503_not_silent_fallback(client, monkeypatch):
    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()]))

    class BrokenLLM(LocalLLMProvider):
        name = "broken"

        def generate(self, request, model):
            raise InferenceError("модель недоступна")

        def health_check(self) -> bool:
            return False

        def list_models(self):
            return []

    monkeypatch.setattr(dependencies, "get_llm_provider", lambda: BrokenLLM())

    response = client.post(
        "/chat",
        json={"question": "Вопрос", "knowledge_base_id": str(uuid.uuid4())},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "inference_error"


def test_chat_emits_a_structured_trace_event(client, monkeypatch, caplog):
    import json
    import logging

    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()]))
    monkeypatch.setattr(
        dependencies,
        "get_llm_provider",
        lambda: FakeLLM(
            json.dumps({"answer": "Ежегодно [1].", "has_answer": True, "citations": [1]})
        ),
    )
    caplog.set_level(logging.INFO, logger="rag.query")

    client.post(
        "/chat",
        json={"question": "Когда отпуск?", "knowledge_base_id": str(uuid.uuid4())},
    )

    record = next(r for r in caplog.records if r.name == "rag.query")
    assert record.rag_query["original_query"] == "Когда отпуск?"
    assert record.rag_query["has_answer"] is True
    assert record.rag_query["retrieved_chunk_ids"] == ["c1"]
