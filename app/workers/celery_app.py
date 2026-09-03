"""Celery-приложение для фоновой обработки документов.

Парсинг, эмбеддинги и индексация — операции на минуты, держать на них
открытым HTTP-запрос нельзя. Redis выступает и брокером, и хранилищем
результатов: отдельная инфраструктура ради этого не нужна.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "local_knowledge_rag",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.workers.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Задача подтверждается после выполнения: если воркер умрёт на
        # середине индексации, документ не потеряется, а будет переобработан.
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        # Eager-режим выполняет задачу прямо в вызывающем процессе — нужен
        # для тестов и локального запуска без отдельного воркера.
        task_always_eager=settings.celery_task_always_eager,
        task_eager_propagates=False,
    )
    return app


celery_app = create_celery_app()
