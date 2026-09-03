import pytest

from app.rag.embeddings import EmbeddingProvider
from app.rag.indexer import DocumentIndexer
from app.rag.vector_store import build_chunk_id


class FakeEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @property
    def model(self) -> str:
        return "fake-embed"

    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(self, texts):
        self.batches.append(texts)
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def health_check(self) -> bool:
        return True


class FakeStore:
    def __init__(self) -> None:
        self.points = []
        self.deleted: list[tuple[str, int | None]] = []
        self.ensured = 0

    def ensure_collection(self):
        self.ensured += 1

    def upsert_chunks(self, chunks):
        self.points.extend(chunks)
        return len(chunks)

    def delete_document(self, document_id, version=None):
        self.deleted.append((document_id, version))


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(
        "# Отпуска\n\nОтпуск предоставляется ежегодно.\n\n"
        "## Перенос\n\nПеренос согласуется с руководителем.",
        encoding="utf-8",
    )
    return path


def test_indexing_produces_points_with_source_metadata(document):
    store = FakeStore()
    indexer = DocumentIndexer(FakeEmbeddings(), store)

    result = indexer.index(document, document_id="doc-1", knowledge_base_id="kb-1")

    assert result.chunk_count == len(store.points) > 0
    assert {point.knowledge_base_id for point in store.points} == {"kb-1"}
    assert store.points[0].section == "Отпуска"
    assert store.points[0].document_name == "policy.md"
    assert store.points[0].embedding_version == "fake-embed:3"


def test_chunk_ids_are_deterministic(document):
    store = FakeStore()

    DocumentIndexer(FakeEmbeddings(), store).index(
        document, document_id="doc-1", knowledge_base_id="kb-1", version=2
    )

    assert store.points[0].chunk_id == build_chunk_id("doc-1", 2, 0)


def test_reindexing_deletes_previous_version_after_writing_new_one(document):
    store = FakeStore()

    DocumentIndexer(FakeEmbeddings(), store).index(
        document,
        document_id="doc-1",
        knowledge_base_id="kb-1",
        version=2,
        previous_version=1,
    )

    assert store.deleted == [("doc-1", 1)]
    assert all(point.version == 2 for point in store.points)


def test_first_indexing_does_not_delete_anything(document):
    store = FakeStore()

    DocumentIndexer(FakeEmbeddings(), store).index(
        document, document_id="doc-1", knowledge_base_id="kb-1"
    )

    assert store.deleted == []


def test_embeddings_are_computed_in_batches(document):
    embeddings = FakeEmbeddings()

    DocumentIndexer(embeddings, FakeStore(), batch_size=1).index(
        document, document_id="doc-1", knowledge_base_id="kb-1"
    )

    assert all(len(batch) == 1 for batch in embeddings.batches)
    assert len(embeddings.batches) >= 2


def test_removing_a_document_clears_all_its_vectors(document):
    store = FakeStore()

    DocumentIndexer(FakeEmbeddings(), store).remove("doc-1")

    assert store.deleted == [("doc-1", None)]
