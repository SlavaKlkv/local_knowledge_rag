"""Изоляция баз знаний между пользователями.

Главная проверка здесь — не «список не показывает чужое», а то, что чужие
фрагменты не попадают даже в retrieval и в контекст ответа.
"""

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import dependencies
from app.db.base import Base
from app.db.session import get_db
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.main import create_app
from app.rag.embeddings import EmbeddingProvider
from app.rag.reranker import NoOpReranker
from app.rag.vector_store import RetrievedChunk
from tests.conftest import register_user


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
    """Возвращает фрагменты всегда — фильтровать обязан слой прав."""

    def __init__(self) -> None:
        self.searches: list[dict] = []

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return [
            RetrievedChunk(
                chunk_id="secret-1",
                document_id="doc-1",
                document_name="secret.pdf",
                text="Секретные данные чужой базы знаний.",
                score=0.9,
                chunk_index=0,
            )
        ]

    def sparse_search(self, **kwargs):
        return self.search(**kwargs)


class FakeLLM(LocalLLMProvider):
    name = "fake"

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        return GenerationResult(
            text='{"answer": "Ответ [1].", "has_answer": true, "citations": [1]}',
            model=model,
            provider=self.name,
            latency_ms=1,
        )

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="qwen3:4b", provider=self.name)]


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/perm.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture
def vector_store(monkeypatch) -> FakeVectorStore:
    store = FakeVectorStore()
    monkeypatch.setattr(dependencies, "get_embedding_provider", lambda: FakeEmbeddings())
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: store)
    monkeypatch.setattr(dependencies, "get_llm_provider", lambda: FakeLLM())
    return store


@pytest.fixture
def app(db_session, tmp_path):
    from app.api.storage import DocumentStorage

    application = create_app()
    application.dependency_overrides[get_db] = lambda: db_session
    application.dependency_overrides[dependencies.get_reranker] = lambda: NoOpReranker()
    application.dependency_overrides[dependencies.get_document_storage] = (
        lambda: DocumentStorage(tmp_path / "uploads")
    )
    return application


@pytest.fixture
def owner(app) -> tuple[TestClient, dict]:
    client = TestClient(app)
    user = register_user(client, "owner@example.com")
    client.headers.update({"Authorization": f"Bearer {user['token']}"})
    return client, user


@pytest.fixture
def outsider(app) -> tuple[TestClient, dict]:
    client = TestClient(app)
    user = register_user(client, "outsider@example.com")
    client.headers.update({"Authorization": f"Bearer {user['token']}"})
    return client, user


@pytest.fixture
def private_kb(owner) -> str:
    client, _ = owner
    return client.post("/knowledge-bases", json={"name": "Секретная"}).json()["id"]


def test_unauthenticated_requests_are_rejected(app):
    anonymous = TestClient(app)

    assert anonymous.get("/knowledge-bases").status_code == 401
    assert anonymous.post("/knowledge-bases", json={"name": "X"}).status_code == 401


def test_outsider_does_not_see_someone_elses_knowledge_base(outsider, private_kb):
    client, _ = outsider

    assert client.get("/knowledge-bases").json() == []
    assert client.get(f"/knowledge-bases/{private_kb}").status_code == 403


def test_outsider_cannot_search_in_someone_elses_knowledge_base(
    outsider, private_kb, vector_store
):
    client, _ = outsider

    response = client.post(
        "/search", json={"query": "секрет", "knowledge_base_id": private_kb}
    )

    assert response.status_code == 403
    # Ключевое: до векторного индекса запрос вообще не дошёл.
    assert vector_store.searches == []


def test_outsider_cannot_get_an_answer_from_someone_elses_knowledge_base(
    outsider, private_kb, vector_store
):
    client, _ = outsider

    response = client.post(
        "/chat", json={"question": "что там секретного?", "knowledge_base_id": private_kb}
    )

    assert response.status_code == 403
    assert vector_store.searches == []


def test_outsider_cannot_upload_documents_into_someone_elses_knowledge_base(
    outsider, private_kb
):
    client, _ = outsider

    response = client.post(
        "/documents",
        params={"knowledge_base_id": private_kb},
        files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
    )

    assert response.status_code == 403


def test_viewer_can_search_but_not_upload(owner, outsider, private_kb, vector_store):
    owner_client, _ = owner
    outsider_client, outsider_user = outsider

    granted = owner_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": outsider_user["id"], "role": "viewer"},
    )
    assert granted.status_code == 201

    search = outsider_client.post(
        "/search", json={"query": "секрет", "knowledge_base_id": private_kb}
    )
    assert search.status_code == 200

    upload = outsider_client.post(
        "/documents",
        params={"knowledge_base_id": private_kb},
        files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert upload.status_code == 403


def test_editor_can_upload_but_not_delete_the_knowledge_base(
    owner, outsider, private_kb, monkeypatch
):
    monkeypatch.setattr(
        "app.api.routers.documents.index_document_task.delay",
        lambda document_id, job_id: None,
    )
    owner_client, _ = owner
    outsider_client, outsider_user = outsider
    owner_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": outsider_user["id"], "role": "editor"},
    )

    upload = outsider_client.post(
        "/documents",
        params={"knowledge_base_id": private_kb},
        files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert upload.status_code == 202

    assert outsider_client.delete(f"/knowledge-bases/{private_kb}").status_code == 403


def test_revoked_access_stops_further_retrieval(owner, outsider, private_kb, vector_store):
    owner_client, _ = owner
    outsider_client, outsider_user = outsider
    owner_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": outsider_user["id"], "role": "viewer"},
    )
    assert (
        outsider_client.post(
            "/search", json={"query": "секрет", "knowledge_base_id": private_kb}
        ).status_code
        == 200
    )

    owner_client.delete(f"/knowledge-bases/{private_kb}/permissions/{outsider_user['id']}")

    assert (
        outsider_client.post(
            "/search", json={"query": "секрет", "knowledge_base_id": private_kb}
        ).status_code
        == 403
    )


def test_only_the_owner_manages_permissions(owner, outsider, private_kb):
    owner_client, owner_user = owner
    outsider_client, outsider_user = outsider
    owner_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": outsider_user["id"], "role": "editor"},
    )

    # Даже редактор не может раздавать доступы дальше.
    response = outsider_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": str(uuid.uuid4()), "role": "viewer"},
    )

    assert response.status_code == 403
    assert outsider_client.get(f"/knowledge-bases/{private_kb}/permissions").status_code == 403


def test_granting_the_same_user_twice_updates_the_role(owner, outsider, private_kb):
    owner_client, _ = owner
    _, outsider_user = outsider

    owner_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": outsider_user["id"], "role": "viewer"},
    )
    updated = owner_client.post(
        f"/knowledge-bases/{private_kb}/permissions",
        json={"user_id": outsider_user["id"], "role": "editor"},
    )

    assert updated.json()["role"] == "editor"
    permissions = owner_client.get(f"/knowledge-bases/{private_kb}/permissions").json()
    assert len(permissions) == 1


def test_outsider_cannot_read_someone_elses_conversation(owner, outsider, private_kb):
    owner_client, _ = owner
    outsider_client, _ = outsider
    conversation_id = owner_client.post(
        "/conversations", json={"knowledge_base_id": private_kb}
    ).json()["id"]

    assert outsider_client.get(f"/conversations/{conversation_id}").status_code == 403
    assert (
        outsider_client.get(f"/conversations/{conversation_id}/messages").status_code == 403
    )
