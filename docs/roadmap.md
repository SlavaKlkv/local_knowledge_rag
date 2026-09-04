# Roadmap

## Stage 1 — Core Local RAG ✅
FastAPI, PostgreSQL, Qdrant, Ollama, local LLM, local embeddings, PDF/TXT,
chunking, dense retrieval, generation, citations.

## Stage 2 — Advanced Retrieval ✅
Остальные форматы документов, metadata filters, hybrid retrieval,
локальный reranker, query rewriting, configurable chunking.

## Stage 3 — Local Inference Platform ✅
LocalLLMProvider, OllamaProvider, VLLMProvider, HardwareDetector,
RuntimeDetector, ProfileRecommender, профили LIGHT/STANDARD/PERFORMANCE,
ручное переопределение профиля, предложение установки Ollama,
проверка доступности и загрузка моделей, persistent model storage,
кольцевой fallback Qwen/Gemma/Llama, health checks, cooldown, failover.

## Stage 4 — Production Backend ✅
Redis, Celery, асинхронная индексация, document lifecycle, versioning,
reindex, auth, permissions, conversations, Docker Compose.

## Stage 5 — AI Quality
Evaluation dataset ✅, retrieval metrics ✅, RAG evaluation ✅, LLM benchmark,
profile benchmark, no-answer, Prometheus/Grafana, observability.
