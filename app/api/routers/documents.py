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
from app.api.schemas import DocumentRead, DocumentUploadResponse, DocumentVersionRead
from app.api.storage import DocumentStorage, validate_upload
from app.core.errors import NotFoundError, ValidationError
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


@router.post(
    "/{document_id}/reindex", response_model=DocumentUploadResponse, status_code=202
)
def reindex_document(
    document_id: uuid.UUID, db: Session = Depends(get_db)
) -> DocumentUploadResponse:
    """Переиндексирует документ в текущей версии.

    Нужен, когда сменилась embedding-модель или стратегия чанкинга:
    содержимое документа то же, поэтому новая версия не создаётся.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Документ {document_id} не найден")

    if _current_version(db, document) is None:
        raise ValidationError(
            "У документа нет сохранённой версии с файлом — переиндексировать нечего"
        )

    document.status = DocumentStatus.UPLOADED
    document.error = None
    job = IndexingJob(document_id=document.id, status=JobStatus.PENDING)
    db.add(job)
    db.commit()
    db.refresh(document)
    db.refresh(job)

    index_document_task.delay(str(document.id), str(job.id))
    return DocumentUploadResponse(
        document=DocumentRead.model_validate(document), job_id=job.id
    )


@router.post(
    "/{document_id}/versions", response_model=DocumentUploadResponse, status_code=202
)
def upload_new_version(
    document_id: uuid.UUID,
    file: UploadFile,
    db: Session = Depends(get_db),
    storage: DocumentStorage = Depends(get_document_storage),
) -> DocumentUploadResponse:
    """Загружает новую версию документа.

    Векторы предыдущей версии удаляет задача индексации — уже после
    успешной записи новых, иначе сбой оставил бы документ без индекса.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Документ {document_id} не найден")

    content = file.file.read()
    safe_name = validate_upload(
        file.filename or "", len(content), ALLOWED_UPLOAD_EXTENSIONS
    )
    path, checksum = storage.save(content, safe_name)

    document.current_version += 1
    document.filename = safe_name
    document.content_type = file.content_type or document.content_type
    document.size_bytes = len(content)
    document.checksum = checksum
    document.status = DocumentStatus.UPLOADED
    document.error = None

    db.add(
        DocumentVersion(
            document_id=document.id,
            version=document.current_version,
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
    return DocumentUploadResponse(
        document=DocumentRead.model_validate(document), job_id=job.id
    )


@router.get("/{document_id}/versions", response_model=list[DocumentVersionRead])
def list_versions(
    document_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[DocumentVersion]:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Документ {document_id} не найден")
    stmt = (
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version)
    )
    return list(db.scalars(stmt))


def _current_version(db: Session, document: Document) -> DocumentVersion | None:
    stmt = select(DocumentVersion).where(
        DocumentVersion.document_id == document.id,
        DocumentVersion.version == document.current_version,
    )
    return db.scalars(stmt).first()


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
