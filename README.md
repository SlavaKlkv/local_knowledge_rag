# Local Knowledge RAG Platform

Полностью локальная RAG-платформа для интеллектуального поиска и ответов
по внутренним документам организации.

> **найди → отфильтруй → проверь → объясни → покажи источник — полностью локально.**

## Ключевой инвариант

Документы, chunks, embeddings, retrieved context и пользовательские запросы
**не покидают локальную инфраструктуру**. Скрытого cloud fallback не существует:
если локальный inference недоступен, API возвращает `503 inference_error`,
а не отправляет корпоративный контекст во внешний AI-провайдер.

## Особенности

- **RAG-пайплайн написан самостоятельно** — без LangChain, LlamaIndex и прочих
  orchestration-фреймворков. Каждый слой (chunking, retrieval, fusion, reranking,
  context building) можно прочитать и понять целиком.
- **Hybrid retrieval**: dense-эмбеддинги ловят смысловую близость, sparse-векторы —
  точные лексические совпадения (номера статей, коды); объединяются через
  Reciprocal Rank Fusion.
- **Локальный cross-encoder reranker** сужает 20–30 кандидатов до лучших 5–10
  (опциональный extra: базовая установка остаётся лёгкой).
- **Grounded-ответы с citations**: модель обязана ссылаться на номера фрагментов,
  ссылки на несуществующие фрагменты отбрасываются, при нехватке данных
  возвращается честное «нет ответа», а не догадка.
- **Hardware-aware выбор моделей**: детекция CPU/RAM/GPU/VRAM, рекомендация
  профиля LIGHT/STANDARD/PERFORMANCE (с возможностью переопределить вручную).
- **Кольцевой fallback Qwen → Gemma → Llama** с health-состояниями, cooldown
  и ограничением попыток на один запрос.

## Pipeline

```text
documents → parsing → normalization → chunking → local embeddings
→ Qdrant indexing → retrieval (dense + sparse → RRF) → reranking
→ context building → local LLM → answer + citations
```

## Стек

Python 3.13, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, Qdrant,
Redis, Ollama / vLLM, sentence-transformers, Docker Compose, pytest, ruff.

## Быстрый старт

### Через Docker Compose

Поднимает всё сразу — приложение, воркер, PostgreSQL, Qdrant и Redis:

```bash
cp .env.example .env
SECRET_KEY=$(openssl rand -hex 32) docker compose up -d
```

Ollama остаётся на хосте: в контейнере нет доступа к Metal на macOS и к GPU
на Linux без отдельной настройки.

### Локально

```bash
cp .env.example .env
docker compose up -d postgres qdrant redis
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Reranking по умолчанию не устанавливается: cross-encoder тянет torch
(~2.5 ГБ). Если он нужен — `uv sync --extra reranking`, иначе выключите
его через `RERANK_ENABLED=false`.

Индексация документов идёт в фоне, поэтому нужен ещё воркер — в отдельном
терминале:

```bash
uv run celery -A app.workers.celery_app:celery_app worker --loglevel=info
```

Документация API: http://localhost:8000/docs

### Локальные модели

Проверить, что обнаружено и чего не хватает:

```bash
curl http://localhost:8000/system/hardware      # железо и рекомендуемый профиль
curl http://localhost:8000/inference/runtimes   # найденные Ollama / vLLM
curl http://localhost:8000/inference/status     # кольцо моделей и что нужно скачать
```

Загрузка модели запускается **только явным запросом** — приложение не тянет
многогигабайтные веса по своей инициативе:

```bash
curl -X POST http://localhost:8000/inference/models/qwen3:4b/download
curl http://localhost:8000/inference/downloads/qwen3:4b   # прогресс
```

Нужна ещё embedding-модель: `ollama pull nomic-embed-text`.

## API

| Группа | Назначение |
|---|---|
| `/system` | Состояние приложения, железо, рекомендуемый профиль |
| `/inference` | Runtime'ы, кольцо моделей, их установка и загрузка |
| `/knowledge-bases` | Базы знаний — единица изоляции документов |
| `/documents` | Загрузка, статус и удаление документов |
| `/indexing-jobs` | Прогресс фоновой индексации |
| `/search` | Поиск по базе знаний без генерации |
| `/conversations` | Диалоги и история сообщений |
| `/chat` | Вопрос-ответ с citations |

Поддерживаемые форматы документов: PDF, DOCX, HTML, Markdown, TXT.

## Конфигурация

Все параметры — через переменные окружения, см. [.env.example](.env.example):
подключения к PostgreSQL / Qdrant / Redis, выбор inference-провайдера
(`INFERENCE_PROVIDER=ollama|vllm`), переопределение профиля
(`HARDWARE_PROFILE_OVERRIDE`), параметры кольца моделей, reranking и
hybrid retrieval.

## Тесты

```bash
uv run pytest
uv run ruff check .
```

Тесты Qdrant — интеграционные, против реально поднятого сервиса; при его
отсутствии они пропускаются.

## Оценка качества

Качество retrieval измеряется на размеченном датасете: Recall@K, Precision@K,
MRR, nDCG@K и доля ложных срабатываний на вопросах без ответа.

```bash
uv run python -m scripts.evaluate_retrieval docs/evaluation/example_dataset.json --k 1 3 5
```

Качество ответов измеряется отдельно — обоснованность, точность цитат и
честный отказ на вопросах без ответа:

```bash
uv run python -m scripts.evaluate_rag docs/evaluation/example_dataset.json --top-k 5
```

Формат датасета, смысл каждой метрики и результаты прогонов —
в [docs/evaluation/README.md](docs/evaluation/README.md).

## Статус

Реализованы Stage 1–4 (ядро RAG, продвинутый retrieval, локальная
inference-платформа, production-бэкенд); Stage 5 начат — метрики retrieval
уже считаются. Дальнейшие шаги — в [docs/roadmap.md](docs/roadmap.md).

## Лицензия

[MIT](LICENSE)
