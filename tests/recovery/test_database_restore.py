"""Восстановление PostgreSQL из дампа: бэкап, потеря, restore в чистую базу.

Проверяется не «запускается ли pg_dump», а что дамп платформы полностью
восстанавливает и схему, и данные. Схема берётся не из моделей, а из
миграций Alembic — именно её встретит реальный сервер, — поэтому в неё входят
сгенерированные enum-типы и таблица версий. Бэкап, который молча теряет
таблицу или тип, выглядит успешным до первого настоящего сбоя, поэтому после
restore сверяются конкретные строки и JSONB-поля, а не код возврата команд.

Сценарий — стандартная аварийная процедура: снять дамп с рабочей базы и
развернуть его в пустой. Потеря представлена тем, что целевая база изначально
пуста, так что восстановление идёт только из дампа, без остатков от источника.

Требует настоящий PostgreSQL и бинари pg_dump/psql на PATH; иначе тест
пропускается — как интеграционные тесты Qdrant.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import (
    Conversation,
    Document,
    DocumentStatus,
    DocumentVersion,
    IndexingJob,
    JobStatus,
    KnowledgeBase,
    KnowledgeBasePermission,
    Message,
    PermissionRole,
    User,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("pg_dump") is None or shutil.which("psql") is None,
    reason="pg_dump/psql не установлены — восстановление БД проверить нечем",
)


class RecoveryDatabases(NamedTuple):
    settings: Settings
    source_name: str
    target_name: str
    source: Engine
    target: Engine


def _sqlalchemy_dsn(settings: Settings, database: str) -> str:
    return (
        f"postgresql+psycopg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{database}"
    )


def _libpq_dsn(settings: Settings, database: str) -> str:
    # libpq-URL для pg_dump/psql: без суффикса драйвера.
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{database}"
    )


def _tool_env(settings: Settings) -> dict[str, str]:
    return {**os.environ, "PGPASSWORD": settings.postgres_password}


def _run_alembic_upgrade(settings: Settings, database: str) -> None:
    """Собирает схему миграциями — ту же, что увидит реальный сервер."""
    venv_alembic = Path(sys.executable).with_name("alembic")
    alembic = str(venv_alembic) if venv_alembic.exists() else shutil.which("alembic")
    assert alembic, "alembic не найден — нечем собрать схему"
    result = subprocess.run(
        [alembic, "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env={
            **_tool_env(settings),
            "POSTGRES_HOST": settings.postgres_host,
            "POSTGRES_PORT": str(settings.postgres_port),
            "POSTGRES_DB": database,
            "POSTGRES_USER": settings.postgres_user,
            "POSTGRES_PASSWORD": settings.postgres_password,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def databases() -> Iterator[RecoveryDatabases]:
    settings = get_settings()
    admin = create_engine(
        _sqlalchemy_dsn(settings, "postgres"),
        isolation_level="AUTOCOMMIT",
        pool_pre_ping=True,
    )
    try:
        with admin.connect():
            pass
    except Exception as exc:  # noqa: BLE001 - любой отказ соединения значит "нет PostgreSQL"
        admin.dispose()
        pytest.skip(f"PostgreSQL недоступен: {exc}")

    suffix = uuid.uuid4().hex[:12]
    source_name = f"lkr_recovery_src_{suffix}"
    target_name = f"lkr_recovery_dst_{suffix}"
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{source_name}"'))
        connection.execute(text(f'CREATE DATABASE "{target_name}"'))

    source = create_engine(_sqlalchemy_dsn(settings, source_name))
    target = create_engine(_sqlalchemy_dsn(settings, target_name))
    try:
        _run_alembic_upgrade(settings, source_name)
        yield RecoveryDatabases(settings, source_name, target_name, source, target)
    finally:
        source.dispose()
        target.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{source_name}" WITH (FORCE)'))
            connection.execute(text(f'DROP DATABASE IF EXISTS "{target_name}" WITH (FORCE)'))
        admin.dispose()


def _seed(engine: Engine) -> dict[str, uuid.UUID]:
    """Кладёт в источник строки, задевающие enum, JSONB и внешние ключи."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        user = User(email="recovery@example.com", hashed_password="pbkdf2_sha256$1$aa$bb")
        session.add(user)
        session.flush()

        knowledge_base = KnowledgeBase(
            name="Восстановление", description="проверка restore", owner_id=user.id
        )
        session.add(knowledge_base)
        session.flush()

        document = Document(
            knowledge_base_id=knowledge_base.id,
            filename="policy.md",
            content_type="text/markdown",
            size_bytes=123,
            checksum="a" * 64,
            status=DocumentStatus.READY,
            current_version=2,
        )
        session.add(document)
        session.flush()

        session.add_all(
            [
                DocumentVersion(
                    document_id=document.id,
                    version=2,
                    checksum="a" * 64,
                    storage_path="/app/storage/documents/policy.md",
                    chunk_count=7,
                    embedding_model="fake-embed:3",
                    chunking_strategy="recursive",
                ),
                KnowledgeBasePermission(
                    knowledge_base_id=knowledge_base.id,
                    user_id=user.id,
                    role=PermissionRole.EDITOR,
                ),
                IndexingJob(document_id=document.id, status=JobStatus.SUCCEEDED, stage="ready"),
            ]
        )

        conversation = Conversation(
            knowledge_base_id=knowledge_base.id, user_id=user.id, title="Бэкап"
        )
        session.add(conversation)
        session.flush()

        message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="Ответ [1].",
            citations=[{"index": 1, "document_id": str(document.id)}],
            meta={"model": "qwen3:4b", "latency_ms": 42},
        )
        session.add(message)
        session.commit()

        return {
            "user": user.id,
            "knowledge_base": knowledge_base.id,
            "document": document.id,
            "conversation": conversation.id,
            "message": message.id,
        }


def _dump(databases: RecoveryDatabases, destination: Path) -> None:
    result = subprocess.run(
        [
            "pg_dump",
            "--no-owner",
            "--no-privileges",
            "--format=plain",
            "--file",
            str(destination),
            _libpq_dsn(databases.settings, databases.source_name),
        ],
        env=_tool_env(databases.settings),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert destination.stat().st_size > 0


def _restore(databases: RecoveryDatabases, source: Path) -> None:
    result = subprocess.run(
        [
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-q",
            "-d",
            _libpq_dsn(databases.settings, databases.target_name),
            "-f",
            str(source),
        ],
        env=_tool_env(databases.settings),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _assert_restored(engine: Engine, ids: dict[str, uuid.UUID]) -> None:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        user = session.get(User, ids["user"])
        assert user is not None and user.email == "recovery@example.com"

        knowledge_base = session.get(KnowledgeBase, ids["knowledge_base"])
        assert knowledge_base is not None and knowledge_base.owner_id == user.id

        document = session.get(Document, ids["document"])
        assert document is not None
        assert document.status == DocumentStatus.READY
        assert document.current_version == 2

        version = session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document.id, DocumentVersion.version == 2
            )
        )
        assert version is not None and version.chunk_count == 7
        assert version.storage_path == "/app/storage/documents/policy.md"

        permission = session.scalar(select(KnowledgeBasePermission))
        assert permission is not None and permission.role == PermissionRole.EDITOR

        assert session.query(IndexingJob).count() == 1

        message = session.get(Message, ids["message"])
        assert message is not None and message.content == "Ответ [1]."
        # JSONB восстанавливается в структуру, а не в строку.
        assert message.citations == [{"index": 1, "document_id": str(document.id)}]
        assert message.meta == {"model": "qwen3:4b", "latency_ms": 42}

        # Схема применена целиком: версия миграций тоже пережила restore.
        assert session.execute(text("select version_num from alembic_version")).scalar()


def test_dump_restores_schema_and_data(databases: RecoveryDatabases, tmp_path: Path) -> None:
    ids = _seed(databases.source)

    # До restore цели пусты: всё, что появится дальше, пришло из дампа.
    assert not inspect(databases.target).has_table("documents")

    dump_path = tmp_path / "dump.sql"
    _dump(databases, dump_path)
    _restore(databases, dump_path)

    _assert_restored(databases.target, ids)
