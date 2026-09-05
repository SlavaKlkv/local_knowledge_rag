"""Фоновые задачи индексации документов.

Задача ведёт документ по его жизненному циклу и фиксирует каждый переход в
БД: пользователь видит, на каком этапе документ находится, а после сбоя —
на каком этапе он сломался и почему.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentStatus, DocumentVersion, IndexingJob, JobStatus
from app.db.session import get_session_factory
from app.observability.metrics import record_indexing
from app.workers.celery_app import celery_app

logger = logging.getLogger("rag.indexing")


def _build_indexer():
    # Импорт внутри функции: воркер не должен поднимать HTTP-зависимости
    # приложения при импорте модуля задач.
    from app.api.dependencies import get_indexer

    return get_indexer()


@celery_app.task(name="index_document", bind=True, max_retries=0)
def index_document_task(self, document_id: str, job_id: str) -> str:
    """Индексирует документ, отражая прогресс в статусах документа и job'а."""
    session_factory = get_session_factory()
    with session_factory() as db:
        job = db.get(IndexingJob, uuid.UUID(job_id))
        document = db.get(Document, uuid.UUID(document_id))
        if document is None or job is None:
            logger.warning(
                "indexing_target_missing",
                extra={"document_id": document_id, "job_id": job_id},
            )
            return JobStatus.FAILED

        version = _current_version(db, document)
        if version is None:
            _fail(db, document, job, "Не найдена версия документа с путём к файлу")
            return JobStatus.FAILED

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        _set_stage(db, document, job, DocumentStatus.PARSING)

        try:
            indexer = _build_indexer()
            # Индексатор внутри проходит parsing → chunking → embeddings →
            # запись в Qdrant; статусы выставляем вокруг него, чтобы не
            # протаскивать колбэки через весь RAG-слой.
            _set_stage(db, document, job, DocumentStatus.EMBEDDING)
            result = indexer.index(
                Path(version.storage_path),
                document_id=str(document.id),
                knowledge_base_id=str(document.knowledge_base_id),
                version=version.version,
                document_name=document.filename,
                previous_version=(
                    version.version - 1 if version.version > 1 else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - любая ошибка индексации фиксируется
            _fail(db, document, job, str(exc))
            record_indexing("failed")
            logger.warning(
                "indexing_failed",
                extra={"document_id": document_id, "error": str(exc)},
            )
            return JobStatus.FAILED

        version.chunk_count = result.chunk_count
        version.embedding_model = result.embedding_version
        version.chunking_strategy = result.chunking_strategy
        document.status = DocumentStatus.READY
        document.error = None
        job.status = JobStatus.SUCCEEDED
        job.stage = DocumentStatus.READY
        job.finished_at = datetime.now(UTC)
        db.commit()

        record_indexing("succeeded", chunk_count=result.chunk_count)
        logger.info(
            "indexing_succeeded",
            extra={"document_id": document_id, "chunks": result.chunk_count},
        )
        return JobStatus.SUCCEEDED


def _current_version(db: Session, document: Document) -> DocumentVersion | None:
    stmt = (
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id == document.id,
            DocumentVersion.version == document.current_version,
        )
        .limit(1)
    )
    return db.scalars(stmt).first()


def _set_stage(
    db: Session, document: Document, job: IndexingJob, stage: DocumentStatus
) -> None:
    document.status = stage
    job.stage = stage
    db.commit()


def _fail(db: Session, document: Document, job: IndexingJob, error: str) -> None:
    document.status = DocumentStatus.FAILED
    document.error = error
    job.status = JobStatus.FAILED
    job.error = error
    job.finished_at = datetime.now(UTC)
    db.commit()
