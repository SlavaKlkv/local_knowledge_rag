# Многоступенчатая сборка: зависимости ставятся отдельным слоем и не
# пересобираются при каждом изменении кода.
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

WORKDIR /app

# Сначала только манифесты: слой с зависимостями переиспользуется, пока
# pyproject.toml и uv.lock не менялись.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Приложение работает не от root: контейнер имеет доступ к загруженным
# документам, и лишние привилегии ему ни к чему.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app app ./app

# Директория для загруженных файлов создаётся заранее и с нужным
# владельцем: при первом монтировании Docker копирует права из образа,
# иначе именованный том достаётся root, и приложение под непривилегированным
# пользователем не может в него писать.
RUN mkdir -p /app/storage/documents && chown -R app:app /app/storage

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
