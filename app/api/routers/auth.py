"""Регистрация, вход и сведения о текущем пользователе."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.api.schemas import TokenResponse, UserCreate, UserLogin, UserRead
from app.core.errors import ValidationError
from app.core.security import (
    AuthError,
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    email = payload.email.strip().lower()
    existing = db.scalars(select(User).where(User.email == email)).first()
    if existing is not None:
        raise ValidationError("Пользователь с таким email уже зарегистрирован")

    user = User(email=email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    email = payload.email.strip().lower()
    user = db.scalars(select(User).where(User.email == email)).first()

    # Одинаковая ошибка для несуществующего пользователя и неверного пароля:
    # иначе по ответу можно перебором выяснить, кто зарегистрирован.
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise AuthError("Неверный email или пароль")
    if not user.is_active:
        raise AuthError("Учётная запись отключена")

    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)) -> User:
    return user
