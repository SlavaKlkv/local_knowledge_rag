"""Эндпоинт метрик для Prometheus.

Живёт на `/metrics` — это соглашение, от которого Prometheus отталкивается по
умолчанию, и отступать от него без причины значит усложнять всем настройку.

Аутентификации здесь нет сознательно: scrape выполняет не человек, а Prometheus
внутри сети, и токен в его конфиге ничего не защищал бы, зато усложнял бы
эксплуатацию. Раскрывать при этом нечего — в метриках нет ни текстов запросов,
ни идентификаторов баз знаний, документов и пользователей (см. app/observability/
metrics.py). Порт приложения при этом наружу выставлять всё равно не следует.
"""

from fastapi import APIRouter, Response

from app.observability.metrics import render

router = APIRouter(tags=["system"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    payload, content_type = render()
    return Response(content=payload, media_type=content_type)
