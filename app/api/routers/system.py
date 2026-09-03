"""Служебные эндпоинты: состояние приложения и окружения."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/system", tags=["system"])


class HealthResponse(BaseModel):
    status: str
    app_env: str


class InfoResponse(BaseModel):
    app_env: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    qdrant_collection: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings: Settings = get_settings()
    return HealthResponse(status="ok", app_env=settings.app_env)


@router.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    settings = get_settings()
    return InfoResponse(
        app_env=settings.app_env,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        qdrant_collection=settings.qdrant_collection,
    )
