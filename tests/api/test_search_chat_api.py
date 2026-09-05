import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import dependencies
from app.core.errors import InferenceError
from app.db.base import Base
from app.db.session import get_db
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.main import create_app
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranker import NoOpReranker
from app.rag.vector_store import RetrievedChunk
from tests.conftest import authenticate


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

    def sparse_search(self, **kwargs):
        # Хватает для проверки маршрутизации и контрактов API — RRF-фьюжн
        # на реальных dense/sparse скорах покрыт отдельно на HybridRetriever.
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
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture
def client(db_session):
    app = create_app()
    # NoOpReranker по умолчанию: реальный cross-encoder качает веса из
    # сети и не нужен для проверки маршрутизации/контрактов API.
    app.dependency_overrides[dependencies.get_reranker] = lambda: NoOpReranker()
    # /chat трогает БД только когда передан conversation_id, но роутер
    # /conversations нужен всем сценариям с диалогом — держим одну сессию.
    app.dependency_overrides[get_db] = lambda: db_session
    client = TestClient(app)
    authenticate(client)
    return client


@pytest.fixture
def knowledge_base_id(client):
    """Реальная база знаний: retrieval теперь требует прав на неё."""
    return client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]


def test_search_returns_hits_with_citation_metadata(client, knowledge_base_id, monkeypatch):
    monkeypatch.setattr(
        dependencies, "get_embedding_provider", lambda: FakeEmbeddings()
    )
    monkeypatch.setattr(
        dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()])
    )
    response = client.post(
        "/search",
        json={"query": "отпуск", "knowledge_base_id": knowledge_base_id},
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits[0]["document_name"] == "policy.pdf"
    assert hits[0]["page"] == 1


def test_chat_returns_grounded_answer_with_citations(client, knowledge_base_id, monkeypatch):
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
        json={
            "question": "Когда предоставляется отпуск?",
            "knowledge_base_id": knowledge_base_id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_answer"] is True
    assert body["citations"][0]["document_name"] == "policy.pdf"


def test_chat_returns_no_answer_when_nothing_is_retrieved(client, knowledge_base_id, monkeypatch):
    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([]))

    response = client.post(
        "/chat",
        json={"question": "Вопрос без ответа", "knowledge_base_id": knowledge_base_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["has_answer"] is False
    assert body["citations"] == []


def test_inference_error_returns_503_not_silent_fallback(client, knowledge_base_id, monkeypatch):
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
        json={"question": "Вопрос", "knowledge_base_id": knowledge_base_id},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "inference_error"


def test_chat_emits_a_structured_trace_event(client, knowledge_base_id, monkeypatch, caplog):
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
        json={"question": "Когда отпуск?", "knowledge_base_id": knowledge_base_id},
    )

    record = next(r for r in caplog.records if r.name == "rag.query")
    assert record.rag_query["original_query"] == "Когда отпуск?"
    assert record.rag_query["has_answer"] is True
    assert record.rag_query["retrieved_chunk_ids"] == ["c1"]


def test_chat_uses_reranker_to_pick_the_final_chunks(client, knowledge_base_id, monkeypatch):
    import json

    from app.rag.reranker import RerankedChunk, Reranker

    weak = _hit(chunk_id="weak", text="слабое совпадение")
    strong = _hit(chunk_id="strong", text="точное совпадение")

    class ReverseReranker(Reranker):
        """Намеренно инвертирует порядок retrieval, чтобы тест доказывал,
        что финальные citations формируются по результату reranking,
        а не по исходному порядку dense-поиска."""

        name = "reverse"

        def rerank(self, query, candidates, top_k=8):
            reversed_candidates = list(reversed(candidates))
            scores = range(len(reversed_candidates), 0, -1)
            return [
                RerankedChunk(chunk=chunk, rerank_score=float(score))
                for chunk, score in zip(reversed_candidates, scores, strict=True)
            ][:top_k]

        def health_check(self) -> bool:
            return True

    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    # Dense-поиск ставит strong первым; реранкер должен это переопределить.
    monkeypatch.setattr(
        dependencies, "get_vector_store", lambda: FakeVectorStore([strong, weak])
    )
    client.app.dependency_overrides[dependencies.get_reranker] = lambda: ReverseReranker()
    monkeypatch.setattr(
        dependencies,
        "get_llm_provider",
        lambda: FakeLLM(
            json.dumps({"answer": "Ответ [1].", "has_answer": True, "citations": [1]})
        ),
    )

    response = client.post(
        "/chat",
        json={"question": "вопрос", "knowledge_base_id": knowledge_base_id},
    )

    assert response.status_code == 200
    assert response.json()["citations"][0]["chunk_id"] == "weak"


def test_chat_with_conversation_persists_messages(client, monkeypatch):
    import json

    kb_id = client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]
    conversation_id = client.post(
        "/conversations", json={"knowledge_base_id": kb_id}
    ).json()["id"]

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
        json={
            "question": "Когда отпуск?",
            "knowledge_base_id": kb_id,
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == conversation_id

    messages = client.get(f"/conversations/{conversation_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Когда отпуск?"
    assert messages[1]["citations"][0]["document_name"] == "policy.pdf"


def test_chat_rejects_conversation_from_a_different_knowledge_base(client, monkeypatch):
    kb_a = client.post("/knowledge-bases", json={"name": "A"}).json()["id"]
    kb_b = client.post("/knowledge-bases", json={"name": "B"}).json()["id"]
    conversation_id = client.post(
        "/conversations", json={"knowledge_base_id": kb_a}
    ).json()["id"]

    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()]))

    response = client.post(
        "/chat",
        json={
            "question": "вопрос",
            "knowledge_base_id": kb_b,
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 422


def test_chat_uses_history_to_rewrite_a_contextual_follow_up(client, monkeypatch, db_session):
    import json

    from app.db.models import Message
    from app.llm.base import GenerationRequest, GenerationResult

    kb_id = client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]
    conversation_id = client.post(
        "/conversations", json={"knowledge_base_id": kb_id}
    ).json()["id"]

    db_session.add(
        Message(conversation_id=uuid.UUID(conversation_id), role="user", content="Когда отпуск?")
    )
    db_session.add(
        Message(conversation_id=uuid.UUID(conversation_id), role="assistant", content="Ежегодно.")
    )
    db_session.commit()

    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: FakeVectorStore([_hit()]))

    rewritten_query = "Что если отпуск просрочен?"
    final_answer = json.dumps(
        {"answer": "Ответ на переписанный вопрос [1].", "has_answer": True, "citations": [1]}
    )

    class ConversationAwareLLM(LocalLLMProvider):
        """Различает запрос переписывания и запрос генерации по системному
        промпту — так же, как реально ведут себя два разных вызова модели."""

        name = "fake"

        def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
            if "переписываешь вопрос" in request.system:
                text = json.dumps({"rewritten": rewritten_query})
            else:
                text = final_answer
            return GenerationResult(text=text, model=model, provider=self.name, latency_ms=1)

        def health_check(self) -> bool:
            return True

        def list_models(self) -> list[ModelInfo]:
            return [ModelInfo(name="qwen3:4b", provider=self.name)]

    monkeypatch.setattr(dependencies, "get_llm_provider", lambda: ConversationAwareLLM())

    response = client.post(
        "/chat",
        json={
            "question": "А если он просрочен?",
            "knowledge_base_id": kb_id,
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rewritten_query"] == rewritten_query
    assert body["answer"] == "Ответ на переписанный вопрос [1]."
