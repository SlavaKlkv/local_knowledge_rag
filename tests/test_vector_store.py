"""Интеграционные тесты Qdrant.

Пропускаются, если локальный Qdrant не поднят: тесты retrieval проверяют
реальное поведение фильтров и удаления, мок здесь бесполезен.
"""

import uuid

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from app.core.config import get_settings  # noqa: E402
from app.rag.vector_store import (  # noqa: E402
    ChunkPoint,
    QdrantVectorStore,
    build_chunk_id,
)


@pytest.fixture
def store():
    client = qdrant_client.QdrantClient(url=get_settings().qdrant_url, timeout=3)
    try:
        client.get_collections()
    except Exception:  # noqa: BLE001 - любой отказ соединения означает "нет Qdrant"
        pytest.skip("Локальный Qdrant недоступен")
    collection = f"test_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(client=client, collection=collection, dimension=3)
    store.ensure_collection()
    yield store
    client.delete_collection(collection)


def _point(kb: str, doc: str, index: int, vector: list[float], version: int = 1) -> ChunkPoint:
    return ChunkPoint(
        chunk_id=build_chunk_id(doc, version, index),
        document_id=doc,
        knowledge_base_id=kb,
        chunk_index=index,
        text=f"фрагмент {index} документа {doc}",
        vector=vector,
        version=version,
        page=index + 1,
        document_name=f"{doc}.pdf",
    )


def test_search_returns_indexed_chunk_with_metadata(store):
    store.upsert_chunks([_point("kb-1", "doc-1", 0, [1.0, 0.0, 0.0])])

    results = store.search([1.0, 0.0, 0.0], knowledge_base_id="kb-1")

    assert len(results) == 1
    assert results[0].document_name == "doc-1.pdf"
    assert results[0].page == 1
    assert results[0].score > 0.99


def test_search_is_isolated_by_knowledge_base(store):
    store.upsert_chunks(
        [
            _point("kb-1", "doc-1", 0, [1.0, 0.0, 0.0]),
            _point("kb-2", "doc-2", 0, [1.0, 0.0, 0.0]),
        ]
    )

    results = store.search([1.0, 0.0, 0.0], knowledge_base_id="kb-1", top_k=10)

    assert {r.document_id for r in results} == {"doc-1"}


def test_document_filter_narrows_results(store):
    store.upsert_chunks(
        [
            _point("kb-1", "doc-1", 0, [1.0, 0.0, 0.0]),
            _point("kb-1", "doc-2", 0, [0.9, 0.1, 0.0]),
        ]
    )

    results = store.search(
        [1.0, 0.0, 0.0], knowledge_base_id="kb-1", document_ids=["doc-2"]
    )

    assert {r.document_id for r in results} == {"doc-2"}


def test_score_threshold_filters_weak_matches(store):
    store.upsert_chunks([_point("kb-1", "doc-1", 0, [0.0, 1.0, 0.0])])

    results = store.search(
        [1.0, 0.0, 0.0], knowledge_base_id="kb-1", score_threshold=0.5
    )

    assert results == []


def test_reindexing_same_version_does_not_duplicate_points(store):
    point = _point("kb-1", "doc-1", 0, [1.0, 0.0, 0.0])
    store.upsert_chunks([point])
    store.upsert_chunks([point])

    assert store.count("kb-1") == 1


def test_deleting_a_document_removes_its_vectors(store):
    store.upsert_chunks(
        [
            _point("kb-1", "doc-1", 0, [1.0, 0.0, 0.0]),
            _point("kb-1", "doc-2", 0, [0.0, 1.0, 0.0]),
        ]
    )

    store.delete_document("doc-1")

    assert store.count("kb-1") == 1
    remaining = store.search([1.0, 0.0, 0.0], knowledge_base_id="kb-1", top_k=10)
    assert remaining[0].document_id == "doc-2"


def test_old_version_vectors_are_removed_on_reindex(store):
    store.upsert_chunks([_point("kb-1", "doc-1", 0, [1.0, 0.0, 0.0], version=1)])
    store.upsert_chunks([_point("kb-1", "doc-1", 0, [0.0, 1.0, 0.0], version=2)])

    store.delete_document("doc-1", version=1)

    remaining = store.search([0.0, 1.0, 0.0], knowledge_base_id="kb-1", top_k=10)
    assert store.count("kb-1") == 1
    assert remaining[0].chunk_id == build_chunk_id("doc-1", 2, 0)
