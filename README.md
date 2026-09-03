# Local Knowledge RAG Platform

Полностью локальная RAG-платформа для интеллектуального поиска и ответов
по внутренним документам организации.

> **найди → отфильтруй → проверь → объясни → покажи источник — полностью локально.**

## Ключевой инвариант

Документы, chunks, embeddings, retrieved context и пользовательские запросы
**не покидают локальную инфраструктуру**. Скрытого cloud fallback не существует:
если локальный inference недоступен — возвращается ошибка, а не запрос во внешний API.

## Особенности

- RAG/retrieval pipeline реализован самостоятельно, без LangChain / LlamaIndex
  и прочих orchestration-фреймворков.
- Локальные LLM, embedding-модели и reranker (Ollama / vLLM).
- Dense → hybrid retrieval поверх Qdrant, локальный reranking, citations.
- Hardware-aware выбор профиля моделей и кольцевой fallback Qwen → Gemma → Llama.

## Pipeline

```text
documents → parsing → normalization → chunking → local embeddings
→ Qdrant indexing → retrieval → hybrid search → reranking
→ context building → local LLM → answer + citations → evaluation
```

## Стек

Python 3.13+, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, Qdrant,
Redis, Celery, Ollama/vLLM, Docker Compose, Prometheus/Grafana, pytest.

## Статус

Проект разрабатывается поэтапно, см. [docs/roadmap.md](docs/roadmap.md).

## Быстрый старт

```bash
cp .env.example .env
docker compose up -d postgres qdrant redis
uv sync
uv run uvicorn app.main:app --reload
```

Документация API: http://localhost:8000/docs
