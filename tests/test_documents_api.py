import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.core.errors import InferenceError
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.rag.indexer import IndexingResult


class FakeIndexer:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.removed: list[str] = []

    def index(self, path, document_id, knowledge_base_id, **kwargs):
        if self.fail:
            raise InferenceError("модель недоступна")
        return IndexingResult(
            document_id=document_id,
            version=1,
            chunk_count=3,
            embedding_version="fake:3",
            chunking_strategy="recursive",
        )

    def remove(self, document_id: str) -> None:
        self.removed.append(document_id)


@pytest.fixture
def db_session(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture
def client(db_session, tmp_path, monkeypatch):
    from app.api.storage import DocumentStorage

    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[dependencies.get_document_storage] = (
        lambda: DocumentStorage(tmp_path / "uploads")
    )
    app.dependency_overrides[dependencies.get_indexer] = lambda: FakeIndexer()
    return TestClient(app)


@pytest.fixture
def knowledge_base_id(client):
    response = client.post("/knowledge-bases", json={"name": "HR"})
    return response.json()["id"]


def test_uploading_a_supported_file_indexes_and_marks_it_ready(client, knowledge_base_id):
    response = client.post(
        "/documents",
        params={"knowledge_base_id": knowledge_base_id},
        files={"file": ("policy.txt", io.BytesIO(b"content"), "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["filename"] == "policy.txt"


def test_uploading_to_unknown_knowledge_base_is_rejected(client):
    response = client.post(
        "/documents",
        params={"knowledge_base_id": str(uuid.uuid4())},
        files={"file": ("policy.txt", io.BytesIO(b"content"), "text/plain")},
    )

    assert response.status_code == 404


def test_unsupported_extension_is_rejected(client, knowledge_base_id):
    response = client.post(
        "/documents",
        params={"knowledge_base_id": knowledge_base_id},
        files={"file": ("archive.zip", io.BytesIO(b"content"), "application/zip")},
    )

    assert response.status_code == 422


def test_indexing_failure_marks_document_failed_instead_of_500(
    client, db_session, tmp_path, knowledge_base_id
):

    client.app.dependency_overrides[dependencies.get_indexer] = lambda: FakeIndexer(fail=True)

    response = client.post(
        "/documents",
        params={"knowledge_base_id": knowledge_base_id},
        files={"file": ("policy.txt", io.BytesIO(b"content"), "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "failed"
    assert "недоступна" in body["error"]


def test_deleting_a_document_removes_its_vectors(client, knowledge_base_id):
    upload = client.post(
        "/documents",
        params={"knowledge_base_id": knowledge_base_id},
        files={"file": ("policy.txt", io.BytesIO(b"content"), "text/plain")},
    )
    document_id = upload.json()["id"]

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 204
    assert client.get(f"/documents/{document_id}").status_code == 404
