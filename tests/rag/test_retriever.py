import pytest

from app.core.errors import ValidationError
from app.rag.embeddings import EmbeddingProvider
from app.rag.retriever import DenseRetriever, RetrievalQuery
from app.rag.vector_store import RetrievedChunk


class FakeEmbeddings(EmbeddingProvider):
    @property
    def model(self) -> str:
        return "fake-embed"

    @property
    def dimension(self) -> int:
        return 3

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def health_check(self) -> bool:
        return True


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                document_name="doc.pdf",
                text="фрагмент",
                score=0.8,
                chunk_index=0,
            )
        ]


@pytest.fixture
def retriever_and_store():
    store = FakeStore()
    return DenseRetriever(FakeEmbeddings(), store), store


def test_retrieval_passes_knowledge_base_filter_to_the_store(retriever_and_store):
    retriever, store = retriever_and_store

    retriever.retrieve(RetrievalQuery(text="вопрос", knowledge_base_id="kb-1", top_k=5))

    assert store.calls[0]["knowledge_base_id"] == "kb-1"
    assert store.calls[0]["top_k"] == 5


def test_score_threshold_and_document_filter_are_forwarded(retriever_and_store):
    retriever, store = retriever_and_store

    retriever.retrieve(
        RetrievalQuery(
            text="вопрос",
            knowledge_base_id="kb-1",
            score_threshold=0.5,
            document_ids=["d1"],
        )
    )

    assert store.calls[0]["score_threshold"] == 0.5
    assert store.calls[0]["document_ids"] == ["d1"]


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(retriever_and_store, query):
    retriever, _ = retriever_and_store

    with pytest.raises(ValidationError):
        retriever.retrieve(RetrievalQuery(text=query, knowledge_base_id="kb-1"))


def test_non_positive_top_k_is_rejected(retriever_and_store):
    retriever, _ = retriever_and_store

    with pytest.raises(ValidationError):
        retriever.retrieve(
            RetrievalQuery(text="вопрос", knowledge_base_id="kb-1", top_k=0)
        )


def _chunk(chunk_id: str, score: float = 0.5, **kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=kwargs.pop("document_id", "d1"),
        document_name=kwargs.pop("document_name", "doc.pdf"),
        text=kwargs.pop("text", "фрагмент"),
        score=score,
        chunk_index=kwargs.pop("chunk_index", 0),
        **kwargs,
    )


class FakeHybridStore:
    """Раздельные списки для dense и sparse — реальные ранги двух разных
    ретриверов не совпадают, это и проверяет RRF."""

    def __init__(
        self, dense_hits: list[RetrievedChunk], sparse_hits: list[RetrievedChunk]
    ) -> None:
        self.dense_hits = dense_hits
        self.sparse_hits = sparse_hits
        self.sparse_calls: list[dict] = []

    def search(self, **kwargs):
        return self.dense_hits

    def sparse_search(self, **kwargs):
        self.sparse_calls.append(kwargs)
        return self.sparse_hits


def test_sparse_retriever_delegates_to_sparse_search():
    from app.rag.retriever import SparseRetriever

    store = FakeHybridStore(dense_hits=[], sparse_hits=[_chunk("a")])

    results = SparseRetriever(store).retrieve(
        RetrievalQuery(text="152-ФЗ", knowledge_base_id="kb-1")
    )

    assert [r.chunk_id for r in results] == ["a"]
    assert store.sparse_calls[0]["knowledge_base_id"] == "kb-1"


def test_hybrid_retriever_boosts_chunks_found_by_both_lists():
    from app.rag.retriever import DenseRetriever, HybridRetriever, SparseRetriever

    only_dense = _chunk("only-dense")
    only_sparse = _chunk("only-sparse")
    both = _chunk("both")

    store = FakeHybridStore(
        dense_hits=[both, only_dense],
        sparse_hits=[only_sparse, both],
    )
    hybrid = HybridRetriever(
        DenseRetriever(FakeEmbeddings(), store), SparseRetriever(store)
    )

    results = hybrid.retrieve(RetrievalQuery(text="вопрос", knowledge_base_id="kb-1"))

    # Чанк, найденный обоими списками, получает вклад от каждого ранга
    # и обязан оказаться выше того, что нашёл только один из ретриверов.
    assert results[0].chunk_id == "both"
    assert {r.chunk_id for r in results} == {"only-dense", "only-sparse", "both"}


def test_hybrid_retriever_respects_top_k():
    from app.rag.retriever import DenseRetriever, HybridRetriever, SparseRetriever

    dense_hits = [_chunk(f"d{i}") for i in range(5)]
    store = FakeHybridStore(dense_hits=dense_hits, sparse_hits=[])
    hybrid = HybridRetriever(
        DenseRetriever(FakeEmbeddings(), store), SparseRetriever(store)
    )

    results = hybrid.retrieve(
        RetrievalQuery(text="вопрос", knowledge_base_id="kb-1", top_k=2)
    )

    assert len(results) == 2


def test_hybrid_retriever_deduplicates_across_lists():
    from app.rag.retriever import DenseRetriever, HybridRetriever, SparseRetriever

    shared = _chunk("shared")
    store = FakeHybridStore(dense_hits=[shared], sparse_hits=[shared])
    hybrid = HybridRetriever(
        DenseRetriever(FakeEmbeddings(), store), SparseRetriever(store)
    )

    results = hybrid.retrieve(RetrievalQuery(text="вопрос", knowledge_base_id="kb-1"))

    assert [r.chunk_id for r in results] == ["shared"]


def test_hybrid_retriever_rejects_invalid_query():
    from app.rag.retriever import DenseRetriever, HybridRetriever, SparseRetriever

    store = FakeHybridStore(dense_hits=[], sparse_hits=[])
    hybrid = HybridRetriever(
        DenseRetriever(FakeEmbeddings(), store), SparseRetriever(store)
    )

    with pytest.raises(ValidationError):
        hybrid.retrieve(RetrievalQuery(text="", knowledge_base_id="kb-1"))
