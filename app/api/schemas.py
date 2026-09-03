"""Схемы запросов и ответов API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.db.models import DocumentStatus, JobStatus


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class KnowledgeBaseRead(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentRead(BaseModel):
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    current_version: int
    error: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class IndexingJobRead(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    status: JobStatus
    stage: str | None = None
    error: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentVersionRead(BaseModel):
    id: uuid.UUID
    version: int
    checksum: str
    chunk_count: int
    embedding_model: str | None = None
    chunking_strategy: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Загрузка принята: документ создан, индексация идёт в фоне."""

    document: DocumentRead
    job_id: uuid.UUID


class CitationRead(BaseModel):
    ref: int
    chunk_id: str
    document_id: str
    document_name: str | None = None
    page: int | None = None
    section: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    knowledge_base_id: uuid.UUID
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str | None
    text: str
    score: float
    page: int | None = None
    section: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    knowledge_base_id: uuid.UUID
    top_k: int = Field(default=10, ge=1, le=100)
    # Диалог опционален: без него /chat остаётся одиночным stateless-запросом.
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    answer: str
    has_answer: bool
    citations: list[CitationRead]
    model: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
    conversation_id: uuid.UUID | None = None
    rewritten_query: str | None = None


class ConversationCreate(BaseModel):
    knowledge_base_id: uuid.UUID
    title: str | None = Field(default=None, max_length=512)


class ConversationRead(BaseModel):
    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageRead(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[dict] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
