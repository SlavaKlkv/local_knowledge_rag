"""Диалоги: создание, просмотр истории сообщений."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import ConversationCreate, ConversationRead, MessageRead
from app.core.errors import NotFoundError
from app.db.models import Conversation, KnowledgeBase, Message
from app.db.session import get_db

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=201)
def create_conversation(
    payload: ConversationCreate, db: Session = Depends(get_db)
) -> Conversation:
    kb = db.get(KnowledgeBase, payload.knowledge_base_id)
    if kb is None:
        raise NotFoundError(f"База знаний {payload.knowledge_base_id} не найдена")

    conversation = Conversation(
        knowledge_base_id=payload.knowledge_base_id, title=payload.title
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("/{conversation_id}", response_model=ConversationRead)
def get_conversation(
    conversation_id: uuid.UUID, db: Session = Depends(get_db)
) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Диалог {conversation_id} не найден")
    return conversation


@router.get("/{conversation_id}/messages", response_model=list[MessageRead])
def list_messages(
    conversation_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[Message]:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Диалог {conversation_id} не найден")
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    return list(db.scalars(stmt))
