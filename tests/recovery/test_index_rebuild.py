"""Восстановление векторного индекса Qdrant из источника истины.

Qdrant в этой платформе — производные данные: он хранит только чанки текущих
версий, а истина лежит в PostgreSQL и на диске (`storage_path`). Бэкапить
индекс поэтому не нужно — при полной потере коллекции его пересобирают
переиндексацией. Тест фиксирует именно это свойство: после исчезновения
коллекции индекс восстанавливается из исходных документов и совпадает с
источником, а не остаётся пустым.

Пропускается, если локальный Qdrant не поднят, — как остальные интеграционные
тесты Qdrant.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from app.core.config import get_settings  # noqa: E402
from app.rag.embeddings import EmbeddingProvider  # noqa: E402
from app.rag.indexer import DocumentIndexer  # noqa: E402
from app.rag.vector_store import QdrantVectorStore  # noqa: E402


class FakeEmbeddings(EmbeddingProvider):
    @property
    def model(self) -> str:
        return "fake-embed"

    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def health_check(self) -> bool:
        return True


@pytest.fixture
def client():
    client = qdrant_client.QdrantClient(url=get_settings().qdrant_url, timeout=3)
    try:
        client.get_collections()
    except Exception:  # noqa: BLE001 - любой отказ соединения значит "нет Qdrant"
        pytest.skip("Локальный Qdrant недоступен")
    return client


@pytest.fixture
def store(client):
    collection = f"recovery_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(client=client, collection=collection, dimension=3)
    store.ensure_collection()
    yield store
    client.delete_collection(collection)


def _document(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _indexed_documents(client, collection: str) -> set[tuple[str, int]]:
    """Что реально лежит в индексе: пары (документ, версия) без дублей."""
    points, _ = client.scroll(collection_name=collection, limit=1000, with_payload=True)
    return {(point.payload["document_id"], point.payload["version"]) for point in points}


def test_collection_loss_is_recovered_by_reindexing(store, client, tmp_path):
    indexer = DocumentIndexer(FakeEmbeddings(), store)
    first = _document(tmp_path, "policy.md", "# Отпуска\n\nОтпуск предоставляется ежегодно.")
    second = _document(tmp_path, "duty.md", "# Дежурства\n\nСмена длится с 10 до 19 часов.")
    indexer.index(first, document_id="doc-1", knowledge_base_id="kb-1")
    indexer.index(second, document_id="doc-2", knowledge_base_id="kb-1")

    expected_count = store.count("kb-1")
    expected_documents = {("doc-1", 1), ("doc-2", 1)}
    assert expected_count > 0
    assert _indexed_documents(client, store.collection) == expected_documents

    # Полная потеря индекса: коллекция исчезла целиком.
    client.delete_collection(store.collection)
    assert not client.collection_exists(store.collection)

    # Восстановление — пересборка из исходников, а не из бэкапа векторов.
    store.ensure_collection()
    indexer.index(first, document_id="doc-1", knowledge_base_id="kb-1")
    indexer.index(second, document_id="doc-2", knowledge_base_id="kb-1")

    assert store.count("kb-1") == expected_count
    assert _indexed_documents(client, store.collection) == expected_documents
    hits = store.search([1.0, 0.0, 0.0], knowledge_base_id="kb-1", top_k=10)
    assert {hit.document_id for hit in hits} == {"doc-1", "doc-2"}


def test_rebuilt_index_holds_only_the_current_version(store, client, tmp_path):
    indexer = DocumentIndexer(FakeEmbeddings(), store)
    path = _document(tmp_path, "policy.md", "# Отпуска\n\nОтпуск предоставляется ежегодно.")
    indexer.index(path, document_id="doc-1", knowledge_base_id="kb-1", version=1)

    client.delete_collection(store.collection)
    store.ensure_collection()

    # Источник истины после переиндексации — только версия 2; старая в
    # восстановленном индексе остаться не должна.
    indexer.index(
        path,
        document_id="doc-1",
        knowledge_base_id="kb-1",
        version=2,
        previous_version=1,
    )

    assert _indexed_documents(client, store.collection) == {("doc-1", 2)}
