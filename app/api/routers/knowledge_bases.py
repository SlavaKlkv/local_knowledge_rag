"""CRUD баз знаний."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import KnowledgeBaseCreate, KnowledgeBaseRead
from app.core.errors import NotFoundError
from app.db.models import KnowledgeBase
from app.db.session import get_db

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBaseRead, status_code=201)
def create_knowledge_base(
    payload: KnowledgeBaseCreate, db: Session = Depends(get_db)
) -> KnowledgeBase:
    kb = KnowledgeBase(name=payload.name, description=payload.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(db: Session = Depends(get_db)) -> list[KnowledgeBase]:
    return list(db.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at)))


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(
    knowledge_base_id: uuid.UUID, db: Session = Depends(get_db)
) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None:
        raise NotFoundError(f"База знаний {knowledge_base_id} не найдена")
    return kb


@router.delete("/{knowledge_base_id}", status_code=204)
def delete_knowledge_base(
    knowledge_base_id: uuid.UUID, db: Session = Depends(get_db)
) -> None:
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None:
        raise NotFoundError(f"База знаний {knowledge_base_id} не найдена")
    db.delete(kb)
    db.commit()
