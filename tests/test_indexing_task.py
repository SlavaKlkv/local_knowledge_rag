import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.errors import InferenceError
from app.db.base import Base
from app.db.models import (
    Document,
    DocumentStatus,
    DocumentVersion,
    IndexingJob,
    JobStatus,
    KnowledgeBase,
)
from app.rag.indexer import IndexingResult
from app.workers import tasks


class FakeIndexer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    def index(self, path, document_id, knowledge_base_id, **kwargs):
        self.calls.append(
            {"path": path, "document_id": document_id, "kwargs": kwargs}
        )
        if self.error:
            raise self.error
        return IndexingResult(
            document_id=document_id,
            version=kwargs.get("version", 1),
            chunk_count=7,
            embedding_version="fake:3",
            chunking_strategy="recursive",
        )


@pytest.fixture
def session_factory(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path}/task.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(tasks, "get_session_factory", lambda: factory)
    return factory


@pytest.fixture
def prepared(session_factory, tmp_path):
    """Документ с версией и pending-job, как после загрузки через API."""
    source = tmp_path / "policy.txt"
    source.write_text("Отпуск предоставляется ежегодно.", encoding="utf-8")

    with session_factory() as db:
        kb = KnowledgeBase(name="HR")
        db.add(kb)
        db.flush()
        document = Document(
            knowledge_base_id=kb.id,
            filename="policy.txt",
            content_type="text/plain",
            size_bytes=32,
            checksum="x" * 64,
            status=DocumentStatus.UPLOADED,
        )
        db.add(document)
        db.flush()
        db.add(
            DocumentVersion(
                document_id=document.id,
                version=1,
                checksum="x" * 64,
                storage_path=str(source),
            )
        )
        job = IndexingJob(document_id=document.id, status=JobStatus.PENDING)
        db.add(job)
        db.commit()
        return str(document.id), str(job.id)


def _run(monkeypatch, indexer: FakeIndexer, document_id: str, job_id: str) -> str:
    monkeypatch.setattr(tasks, "_build_indexer", lambda: indexer)
    return tasks.index_document_task(document_id, job_id)


def test_successful_indexing_marks_document_ready(monkeypatch, session_factory, prepared):
    document_id, job_id = prepared
    indexer = FakeIndexer()

    result = _run(monkeypatch, indexer, document_id, job_id)

    assert result == JobStatus.SUCCEEDED
    with session_factory() as db:
        document = db.get(Document, uuid.UUID(document_id))
        job = db.get(IndexingJob, uuid.UUID(job_id))
        assert document.status == DocumentStatus.READY
        assert document.error is None
        assert job.status == JobStatus.SUCCEEDED
        assert job.finished_at is not None


def test_successful_indexing_fills_version_metadata(monkeypatch, session_factory, prepared):
    document_id, job_id = prepared

    _run(monkeypatch, FakeIndexer(), document_id, job_id)

    with session_factory() as db:
        version = db.query(DocumentVersion).one()
        assert version.chunk_count == 7
        assert version.embedding_model == "fake:3"
        assert version.chunking_strategy == "recursive"


def test_indexer_receives_the_stored_file_path(monkeypatch, session_factory, prepared):
    document_id, job_id = prepared
    indexer = FakeIndexer()

    _run(monkeypatch, indexer, document_id, job_id)

    assert indexer.calls[0]["path"].name == "policy.txt"
    assert indexer.calls[0]["document_id"] == document_id


def test_first_version_does_not_delete_a_previous_one(monkeypatch, session_factory, prepared):
    document_id, job_id = prepared
    indexer = FakeIndexer()

    _run(monkeypatch, indexer, document_id, job_id)

    assert indexer.calls[0]["kwargs"]["previous_version"] is None


def test_failed_indexing_records_the_error_on_document_and_job(
    monkeypatch, session_factory, prepared
):
    document_id, job_id = prepared
    indexer = FakeIndexer(error=InferenceError("модель недоступна"))

    result = _run(monkeypatch, indexer, document_id, job_id)

    assert result == JobStatus.FAILED
    with session_factory() as db:
        document = db.get(Document, uuid.UUID(document_id))
        job = db.get(IndexingJob, uuid.UUID(job_id))
        assert document.status == DocumentStatus.FAILED
        assert "модель недоступна" in document.error
        assert job.status == JobStatus.FAILED
        assert "модель недоступна" in job.error


def test_missing_document_does_not_raise(monkeypatch, session_factory, prepared):
    _, job_id = prepared

    result = _run(monkeypatch, FakeIndexer(), str(uuid.uuid4()), job_id)

    assert result == JobStatus.FAILED


def test_document_without_a_version_fails_with_a_clear_error(
    monkeypatch, session_factory
):
    with session_factory() as db:
        kb = KnowledgeBase(name="HR")
        db.add(kb)
        db.flush()
        document = Document(
            knowledge_base_id=kb.id,
            filename="orphan.txt",
            content_type="text/plain",
            size_bytes=1,
            checksum="y" * 64,
            status=DocumentStatus.UPLOADED,
        )
        db.add(document)
        db.flush()
        job = IndexingJob(document_id=document.id, status=JobStatus.PENDING)
        db.add(job)
        db.commit()
        document_id, job_id = str(document.id), str(job.id)

    result = _run(monkeypatch, FakeIndexer(), document_id, job_id)

    assert result == JobStatus.FAILED
    with session_factory() as db:
        assert "версия документа" in db.get(Document, uuid.UUID(document_id)).error
