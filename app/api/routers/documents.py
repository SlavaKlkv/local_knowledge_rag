"""Загрузка и жизненный цикл документов.

Индексация уходит в Celery: парсинг и эмбеддинги занимают минуты, держать
на них открытым HTTP-запрос нельзя. Загрузка отвечает 202 и идентификатором
задачи, а прогресс виден через статус документа и /indexing-jobs.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import (
    ALLOWED_UPLOAD_EXTENSIONS,
    get_document_storage,
    get_indexer,
)
from app.api.schemas import DocumentRead, DocumentUploadResponse
from app.api.storage import DocumentStorage, validate_upload
from app.core.errors import NotFoundError
from app.db.models import (
    Document,
    DocumentStatus,
    DocumentVersion,
    IndexingJob,
    JobStatus,
    KnowledgeBase,
)
from app.db.session import get_db
from app.rag.indexer import DocumentIndexer
from app.workers.tasks import index_document_task

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=202)
def upload_document(
    knowledge_base_id: uuid.UUID,
    file: UploadFile,
    db: Session = Depends(get_db),
    storage: DocumentStorage = Depends(get_document_storage),
) -> DocumentUploadResponse:
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None:
        raise NotFoundError(f"База знаний {knowledge_base_id} не найдена")

    content = file.file.read()
    safe_name = validate_upload(
        file.filename or "", len(content), ALLOWED_UPLOAD_EXTENSIONS
    )
    path, checksum = storage.save(content, safe_name)

    document = Document(
        knowledge_base_id=knowledge_base_id,
        filename=safe_name,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        checksum=checksum,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    db.flush()

    # Версия с путём к файлу создаётся сразу, до индексации: иначе после
    # сбоя путь к загруженному файлу потерян и переиндексировать нечего.
    db.add(
        DocumentVersion(
            document_id=document.id,
            version=1,
            checksum=checksum,
            storage_path=str(path),
        )
    )
    job = IndexingJob(document_id=document.id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()
    db.refresh(document)
    db.refresh(job)

    index_document_task.delay(str(document.id), str(job.id))
    return DocumentUploadResponse(document=DocumentRead.model_validate(document), job_id=job.id)


@router.get("", response_model=list[DocumentRead])
def list_documents(
    knowledge_base_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.created_at)
    )
    return list(db.scalars(stmt))


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Документ {document_id} не найден")
    return document


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
    indexer: DocumentIndexer = Depends(get_indexer),
) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Документ {document_id} не найден")
    # Векторы удаляются раньше строки БД: если упадёт на середине, лучше
    # оставить "осиротевшую" запись документа, чем stale-векторы в Qdrant.
    indexer.remove(str(document_id))
    db.delete(document)
    db.commit()
