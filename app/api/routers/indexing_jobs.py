"""Статус фоновых задач индексации."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, require_role
from app.api.schemas import IndexingJobRead
from app.core.errors import NotFoundError
from app.db.models import Document, IndexingJob, PermissionRole, User
from app.db.session import get_db

router = APIRouter(prefix="/indexing-jobs", tags=["indexing-jobs"])


@router.get("/{job_id}", response_model=IndexingJobRead)
def get_indexing_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> IndexingJob:
    job = db.get(IndexingJob, job_id)
    if job is None:
        raise NotFoundError(f"Задача индексации {job_id} не найдена")
    _require_document_access(db, user, job.document_id)
    return job


@router.get("", response_model=list[IndexingJobRead])
def list_indexing_jobs(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[IndexingJob]:
    _require_document_access(db, user, document_id)
    stmt = (
        select(IndexingJob)
        .where(IndexingJob.document_id == document_id)
        .order_by(IndexingJob.created_at.desc())
    )
    return list(db.scalars(stmt))


def _require_document_access(db: Session, user: User, document_id: uuid.UUID) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Документ {document_id} не найден")
    require_role(db, user, document.knowledge_base_id, PermissionRole.VIEWER)
