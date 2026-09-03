"""Статус фоновых задач индексации."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import IndexingJobRead
from app.core.errors import NotFoundError
from app.db.models import IndexingJob
from app.db.session import get_db

router = APIRouter(prefix="/indexing-jobs", tags=["indexing-jobs"])


@router.get("/{job_id}", response_model=IndexingJobRead)
def get_indexing_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> IndexingJob:
    job = db.get(IndexingJob, job_id)
    if job is None:
        raise NotFoundError(f"Задача индексации {job_id} не найдена")
    return job


@router.get("", response_model=list[IndexingJobRead])
def list_indexing_jobs(
    document_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[IndexingJob]:
    stmt = (
        select(IndexingJob)
        .where(IndexingJob.document_id == document_id)
        .order_by(IndexingJob.created_at.desc())
    )
    return list(db.scalars(stmt))
