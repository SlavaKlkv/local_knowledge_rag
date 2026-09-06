"""Точка входа FastAPI-приложения."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routers import (
    auth,
    chat,
    conversations,
    documents,
    indexing_jobs,
    inference,
    knowledge_bases,
    metrics,
    search,
    system,
)
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Local Knowledge RAG Platform",
        description="Локальный поиск и ответы по внутренним документам организации",
        version="0.1.0",
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    app.include_router(auth.router)
    app.include_router(system.router)
    app.include_router(metrics.router)
    app.include_router(inference.router)
    app.include_router(knowledge_bases.router)
    app.include_router(documents.router)
    app.include_router(indexing_jobs.router)
    app.include_router(search.router)
    app.include_router(conversations.router)
    app.include_router(chat.router)
    _mount_web_ui(app)
    return app


def _mount_web_ui(app: FastAPI) -> None:
    """Отдаёт веб-интерфейс тем же приложением, что и API.

    Интерфейс — статика без сборки, поэтому его достаточно смонтировать:
    отдельный процесс, node-тулчейн и настройка CORS не нужны, а запросы
    из браузера идут на тот же origin, что и страница.
    """
    static_dir = Path(__file__).parent / "web" / "static"
    if not static_dir.is_dir():  # pragma: no cover - только битая сборка
        return

    app.mount("/ui", StaticFiles(directory=static_dir), name="ui")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")


app = create_app()
