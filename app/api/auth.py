"""Зависимости аутентификации и проверки прав.

Права проверяются на входе в эндпоинт и повторно применяются как фильтр
retrieval: недостаточно скрыть базу знаний в списке — фрагменты из неё не
должны попадать даже в контекст ответа.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import AuthError, ForbiddenError, decode_access_token
from app.db.models import KnowledgeBase, KnowledgeBasePermission, PermissionRole, User
from app.db.session import get_db

# OWNER умеет всё, что EDITOR, а EDITOR — всё, что VIEWER.
_ROLE_RANK = {
    PermissionRole.VIEWER: 0,
    PermissionRole.EDITOR: 1,
    PermissionRole.OWNER: 2,
}


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("Требуется заголовок Authorization: Bearer <token>")

    user_id = decode_access_token(token)
    try:
        user = db.get(User, uuid.UUID(user_id))
    except ValueError as exc:
        raise AuthError("Некорректный идентификатор пользователя в токене") from exc

    if user is None:
        raise AuthError("Пользователь из токена не найден")
    if not user.is_active:
        # Токен ещё не истёк, но пользователь отключён — доступа нет.
        raise AuthError("Учётная запись отключена")
    return user


def user_role(
    db: Session, user: User, knowledge_base_id: uuid.UUID
) -> PermissionRole | None:
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None:
        return None
    if kb.owner_id == user.id:
        # Владелец не нуждается в явной записи прав на свою базу знаний.
        return PermissionRole.OWNER

    stmt = select(KnowledgeBasePermission).where(
        KnowledgeBasePermission.knowledge_base_id == knowledge_base_id,
        KnowledgeBasePermission.user_id == user.id,
    )
    permission = db.scalars(stmt).first()
    return permission.role if permission else None


def require_role(
    db: Session,
    user: User,
    knowledge_base_id: uuid.UUID,
    minimum: PermissionRole = PermissionRole.VIEWER,
) -> PermissionRole:
    role = user_role(db, user, knowledge_base_id)
    if role is None or _ROLE_RANK[role] < _ROLE_RANK[minimum]:
        # Одна и та же ошибка и для «нет базы знаний», и для «нет прав»:
        # иначе по коду ответа можно перебором узнать, какие базы существуют.
        raise ForbiddenError("Нет доступа к этой базе знаний")
    return role


def accessible_knowledge_base_ids(db: Session, user: User) -> list[uuid.UUID]:
    """Базы знаний, доступные пользователю: свои плюс выданные явно."""
    owned = select(KnowledgeBase.id).where(KnowledgeBase.owner_id == user.id)
    granted = select(KnowledgeBasePermission.knowledge_base_id).where(
        KnowledgeBasePermission.user_id == user.id
    )
    return list(db.scalars(owned)) + list(db.scalars(granted))
