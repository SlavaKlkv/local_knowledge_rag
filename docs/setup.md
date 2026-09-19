# Установка без Docker и работа с моделями

Быстрый запуск целиком в контейнерах описан в
[README](../README.md#запуск). Здесь — то, что нужно реже: установка из
исходников для разработки и управление локальными моделями через API.

## Локальная установка

Хранилища всё равно удобнее держать в контейнерах, из исходников поднимается
только приложение и воркер.

```bash
git clone https://github.com/SlavaKlkv/local_knowledge_rag.git && cd local_knowledge_rag
cp .env.example .env
docker compose up -d postgres qdrant redis
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Индексация документов идёт в фоне, поэтому нужен ещё воркер — в отдельном
терминале:

```bash
uv run celery -A app.workers.celery_app:celery_app worker --loglevel=info
```

Интерфейс — на [localhost:8000](http://localhost:8000), схема API —
[Swagger UI](http://localhost:8000/docs).

Reranking по умолчанию не устанавливается: cross-encoder тянет torch
(~2.5 ГБ). Если он нужен — `uv sync --extra reranking`, иначе выключите его
через `RERANK_ENABLED=false`.

Нужна embedding-модель: `ollama pull nomic-embed-text`.

## Что обнаружено на машине

| Что показывает | Запрос |
|---|---|
| CPU/RAM/GPU/VRAM и рекомендуемый профиль | `curl http://localhost:8000/system/hardware` |
| Найденные runtime'ы Ollama / vLLM | `curl http://localhost:8000/inference/runtimes` |
| Кольцо моделей и недостающие веса | `curl http://localhost:8000/inference/status` |

## Загрузка весов

Загрузка запускается **только явным запросом** — приложение не тянет
многогигабайтные веса по своей инициативе:

```bash
curl -X POST http://localhost:8000/inference/models/qwen3:4b/download
curl http://localhost:8000/inference/downloads/qwen3:4b
```

## Состав кольца моделей

Состав и порядок задаёт [профиль железа](../README.md#профили-железа), но их
можно переопределить — порядок в списке и есть порядок обхода при отказе
модели. Настройка живёт в `storage/ring_config.json` и применяется без
перезапуска:

```bash
curl http://localhost:8000/inference/ring
curl -X PUT http://localhost:8000/inference/ring \
  -H 'Content-Type: application/json' \
  -d '{"models": ["llama3.1:8b", "qwen3:4b"]}'
curl -X DELETE http://localhost:8000/inference/ring   # вернуть состав профиля
```

Добавить в кольцо можно модель любого профиля или уже установленную в
runtime; неизвестная модель отклоняется, чтобы кольцо не деградировало до
отказа отвечать. То же доступно в интерфейсе — кнопка «Настроить» над
кольцом моделей.
