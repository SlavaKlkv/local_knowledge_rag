from app.db.base import Base
from app.db.models import Document, DocumentStatus, JobStatus


def test_schema_covers_core_entities():
    tables = set(Base.metadata.tables)
    assert {
        "users",
        "knowledge_bases",
        "documents",
        "document_versions",
        "indexing_jobs",
        "conversations",
        "messages",
    } <= tables


def test_document_lifecycle_statuses():
    assert DocumentStatus.UPLOADED == "uploaded"
    assert DocumentStatus.FAILED == "failed"
    assert set(JobStatus) == {
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
    }


def test_documents_are_isolated_by_knowledge_base():
    fk = next(iter(Document.__table__.c.knowledge_base_id.foreign_keys))
    assert fk.column.table.name == "knowledge_bases"
    assert Document.__table__.c.knowledge_base_id.nullable is False
