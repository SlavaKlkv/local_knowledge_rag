"""CRUD баз знаний."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import (
    accessible_knowledge_base_ids,
    get_current_user,
    require_role,
)
from app.api.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    PermissionGrant,
    PermissionRead,
)
from app.core.errors import NotFoundError, ValidationError
from app.db.models import (
    KnowledgeBase,
    KnowledgeBasePermission,
    PermissionRole,
    User,
)
from app.db.session import get_db

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBaseRead, status_code=201)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> KnowledgeBase:
    kb = KnowledgeBase(
        name=payload.name, description=payload.description, owner_id=user.id
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[KnowledgeBase]:
    # Пользователь видит только свои и явно выданные базы знаний.
    accessible = accessible_knowledge_base_ids(db, user)
    stmt = (
        select(KnowledgeBase)
        .where(KnowledgeBase.id.in_(accessible))
        .order_by(KnowledgeBase.created_at)
    )
    return list(db.scalars(stmt))


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(
    knowledge_base_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> KnowledgeBase:
    require_role(db, user, knowledge_base_id, PermissionRole.VIEWER)
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None:
        raise NotFoundError(f"База знаний {knowledge_base_id} не найдена")
    return kb


@router.delete("/{knowledge_base_id}", status_code=204)
def delete_knowledge_base(
    knowledge_base_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    # Удаление — только владельцу: редактор может менять содержимое,
    # но не сносить базу знаний целиком.
    require_role(db, user, knowledge_base_id, PermissionRole.OWNER)
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None:
        raise NotFoundError(f"База знаний {knowledge_base_id} не найдена")
    db.delete(kb)
    db.commit()


@router.post(
    "/{knowledge_base_id}/permissions", response_model=PermissionRead, status_code=201
)
def grant_permission(
    knowledge_base_id: uuid.UUID,
    payload: PermissionGrant,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> KnowledgeBasePermission:
    """Выдаёт доступ другому пользователю. Только для владельца."""
    require_role(db, user, knowledge_base_id, PermissionRole.OWNER)
    if payload.user_id == user.id:
        raise ValidationError("Владелец уже имеет полный доступ к своей базе знаний")

    existing = db.scalars(
        select(KnowledgeBasePermission).where(
            KnowledgeBasePermission.knowledge_base_id == knowledge_base_id,
            KnowledgeBasePermission.user_id == payload.user_id,
        )
    ).first()
    if existing is not None:
        existing.role = payload.role
        db.commit()
        db.refresh(existing)
        return existing

    permission = KnowledgeBasePermission(
        knowledge_base_id=knowledge_base_id, user_id=payload.user_id, role=payload.role
    )
    db.add(permission)
    db.commit()
    db.refresh(permission)
    return permission


@router.get("/{knowledge_base_id}/permissions", response_model=list[PermissionRead])
def list_permissions(
    knowledge_base_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[KnowledgeBasePermission]:
    require_role(db, user, knowledge_base_id, PermissionRole.OWNER)
    stmt = select(KnowledgeBasePermission).where(
        KnowledgeBasePermission.knowledge_base_id == knowledge_base_id
    )
    return list(db.scalars(stmt))


@router.delete(
    "/{knowledge_base_id}/permissions/{user_id}", status_code=204
)
def revoke_permission(
    knowledge_base_id: uuid.UUID,
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    require_role(db, user, knowledge_base_id, PermissionRole.OWNER)
    permission = db.scalars(
        select(KnowledgeBasePermission).where(
            KnowledgeBasePermission.knowledge_base_id == knowledge_base_id,
            KnowledgeBasePermission.user_id == user_id,
        )
    ).first()
    if permission is None:
        raise NotFoundError("У пользователя нет доступа к этой базе знаний")
    db.delete(permission)
    db.commit()
