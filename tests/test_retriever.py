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
