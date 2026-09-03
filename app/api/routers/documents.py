"""Загрузка и жизненный цикл документов.

Индексация выполняется синхронно в V1 — фоновые задачи (Celery) появляются
в V4. Роутер уже отделён от механизма индексации, чтобы замена была локальной.
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
from app.api.schemas import DocumentRead
from app.api.storage import DocumentStorage, validate_upload
from app.core.errors import NotFoundError
from app.db.models import Document, DocumentStatus, DocumentVersion, KnowledgeBase
from app.db.session import get_db
from app.rag.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=201)
def upload_document(
    knowledge_base_id: uuid.UUID,
    file: UploadFile,
    db: Session = Depends(get_db),
    storage: DocumentStorage = Depends(get_document_storage),
    indexer: DocumentIndexer = Depends(get_indexer),
) -> Document:
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
        status=DocumentStatus.PARSING,
    )
    db.add(document)
    db.flush()

    try:
        result = indexer.index(
            path,
            document_id=str(document.id),
            knowledge_base_id=str(knowledge_base_id),
            document_name=safe_name,
        )
    except Exception as exc:  # noqa: BLE001 - ошибка индексации фиксируется в статусе
        document.status = DocumentStatus.FAILED
        document.error = str(exc)
        db.commit()
        db.refresh(document)
        return document

    document.status = DocumentStatus.READY
    db.add(
        DocumentVersion(
            document_id=document.id,
            version=1,
            checksum=checksum,
            storage_path=str(path),
            chunk_count=result.chunk_count,
            embedding_model=result.embedding_version,
            chunking_strategy=result.chunking_strategy,
        )
    )
    db.commit()
    db.refresh(document)
    return document


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
