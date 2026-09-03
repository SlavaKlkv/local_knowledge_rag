"""Схемы запросов и ответов API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.db.models import DocumentStatus


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


class ChatResponse(BaseModel):
    answer: str
    has_answer: bool
    citations: list[CitationRead]
    model: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
