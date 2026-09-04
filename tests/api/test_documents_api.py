import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import dependencies
from app.api.storage import DocumentStorage
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from tests.conftest import authenticate


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        yield session


class FakeIndexer:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove(self, document_id: str) -> None:
        self.removed.append(document_id)


@pytest.fixture
def indexer() -> FakeIndexer:
    return FakeIndexer()


@pytest.fixture
def dispatched(monkeypatch) -> list[tuple[str, str]]:
    """Перехватывает постановку задачи: сам воркер здесь не проверяется."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.api.routers.documents.index_document_task.delay",
        lambda document_id, job_id: calls.append((document_id, job_id)),
    )
    return calls


@pytest.fixture
def client(db_session, tmp_path, indexer, dispatched):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[dependencies.get_document_storage] = (
        lambda: DocumentStorage(tmp_path / "uploads")
    )
    app.dependency_overrides[dependencies.get_indexer] = lambda: indexer
    client = TestClient(app)
    authenticate(client)
    return client


@pytest.fixture
def knowledge_base_id(client):
    return client.post("/knowledge-bases", json={"name": "HR"}).json()["id"]


def _upload(client, knowledge_base_id, name="policy.txt", content=b"content"):
    return client.post(
        "/documents",
        params={"knowledge_base_id": knowledge_base_id},
        files={"file": (name, io.BytesIO(content), "text/plain")},
    )


def test_upload_is_accepted_and_indexing_is_deferred(client, knowledge_base_id, dispatched):
    response = _upload(client, knowledge_base_id)

    assert response.status_code == 202
    body = response.json()
    # Документ ещё не проиндексирован — статус отражает именно это.
    assert body["document"]["status"] == "uploaded"
    assert body["document"]["filename"] == "policy.txt"
    assert dispatched == [(body["document"]["id"], body["job_id"])]


def test_upload_creates_a_pending_indexing_job(client, knowledge_base_id):
    body = _upload(client, knowledge_base_id).json()

    job = client.get(f"/indexing-jobs/{body['job_id']}").json()

    assert job["status"] == "pending"
    assert job["document_id"] == body["document"]["id"]


def test_jobs_can_be_listed_for_a_document(client, knowledge_base_id):
    body = _upload(client, knowledge_base_id).json()
    document_id = body["document"]["id"]

    jobs = client.get("/indexing-jobs", params={"document_id": document_id}).json()

    assert [j["id"] for j in jobs] == [body["job_id"]]


def test_uploading_to_an_inaccessible_knowledge_base_is_rejected(client, dispatched):
    response = _upload(client, str(uuid.uuid4()))

    assert response.status_code == 403
    assert dispatched == []


def test_unsupported_extension_is_rejected(client, knowledge_base_id, dispatched):
    response = client.post(
        "/documents",
        params={"knowledge_base_id": knowledge_base_id},
        files={"file": ("archive.zip", io.BytesIO(b"content"), "application/zip")},
    )

    assert response.status_code == 422
    assert dispatched == []


def test_stored_file_path_is_persisted_before_indexing(client, knowledge_base_id, db_session):
    """Путь сохраняется сразу: иначе после сбоя переиндексировать нечего."""
    from app.db.models import DocumentVersion

    body = _upload(client, knowledge_base_id).json()

    version = db_session.query(DocumentVersion).one()
    assert str(version.document_id) == body["document"]["id"]
    assert version.storage_path
    assert version.chunk_count == 0


def test_deleting_a_document_removes_its_vectors(client, knowledge_base_id, indexer):
    body = _upload(client, knowledge_base_id).json()
    document_id = body["document"]["id"]

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 204
    assert indexer.removed == [document_id]
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_unknown_indexing_job_returns_404(client):
    assert client.get(f"/indexing-jobs/{uuid.uuid4()}").status_code == 404


def test_reindex_starts_a_new_job_without_creating_a_version(
    client, knowledge_base_id, db_session, dispatched
):
    from app.db.models import DocumentVersion

    document_id = _upload(client, knowledge_base_id).json()["document"]["id"]
    dispatched.clear()

    response = client.post(f"/documents/{document_id}/reindex")

    assert response.status_code == 202
    assert response.json()["document"]["current_version"] == 1
    assert db_session.query(DocumentVersion).count() == 1
    assert len(dispatched) == 1


def test_reindex_resets_a_failed_document(client, knowledge_base_id, db_session):
    import uuid as _uuid

    from app.db.models import Document, DocumentStatus

    document_id = _upload(client, knowledge_base_id).json()["document"]["id"]
    document = db_session.get(Document, _uuid.UUID(document_id))
    document.status = DocumentStatus.FAILED
    document.error = "модель недоступна"
    db_session.commit()

    body = client.post(f"/documents/{document_id}/reindex").json()

    assert body["document"]["status"] == "uploaded"
    assert body["document"]["error"] is None


def test_reindex_of_unknown_document_returns_404(client):
    assert client.post(f"/documents/{uuid.uuid4()}/reindex").status_code == 404


def test_new_version_increments_the_counter_and_stores_the_new_file(
    client, knowledge_base_id, db_session, dispatched
):
    from app.db.models import DocumentVersion

    document_id = _upload(client, knowledge_base_id).json()["document"]["id"]
    dispatched.clear()

    response = client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("policy_v2.txt", io.BytesIO(b"updated content"), "text/plain")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["document"]["current_version"] == 2
    assert body["document"]["filename"] == "policy_v2.txt"
    assert body["document"]["status"] == "uploaded"

    versions = db_session.query(DocumentVersion).order_by(DocumentVersion.version).all()
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].storage_path != versions[1].storage_path
    assert len(dispatched) == 1


def test_versions_can_be_listed(client, knowledge_base_id):
    document_id = _upload(client, knowledge_base_id).json()["document"]["id"]
    client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("v2.txt", io.BytesIO(b"v2"), "text/plain")},
    )

    versions = client.get(f"/documents/{document_id}/versions").json()

    assert [v["version"] for v in versions] == [1, 2]


def test_new_version_of_unknown_document_returns_404(client):
    response = client.post(
        f"/documents/{uuid.uuid4()}/versions",
        files={"file": ("v2.txt", io.BytesIO(b"v2"), "text/plain")},
    )

    assert response.status_code == 404


def test_new_version_rejects_unsupported_format(client, knowledge_base_id, dispatched):
    document_id = _upload(client, knowledge_base_id).json()["document"]["id"]
    dispatched.clear()

    response = client.post(
        f"/documents/{document_id}/versions",
        files={"file": ("v2.zip", io.BytesIO(b"v2"), "application/zip")},
    )

    assert response.status_code == 422
    assert dispatched == []
