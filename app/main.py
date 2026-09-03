"""Точка входа FastAPI-приложения."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import system
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

    app.include_router(system.router)
    return app


app = create_app()
