from app.rag.context_builder import ContextBuilder
from app.rag.vector_store import RetrievedChunk


def _chunk(chunk_id: str, text: str, **kwargs) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=kwargs.pop("document_id", "doc-1"),
        document_name=kwargs.pop("document_name", "policy.pdf"),
        text=text,
        score=kwargs.pop("score", 0.9),
        chunk_index=kwargs.pop("chunk_index", 0),
        **kwargs,
    )


def test_duplicate_chunks_are_dropped():
    chunks = [_chunk("a", "Один и тот же текст."), _chunk("b", "один и тот же   текст.")]

    context = ContextBuilder().build(chunks)

    assert len(context.items) == 1


def test_token_budget_is_respected():
    long_text = " ".join(["слово"] * 100)
    chunks = [_chunk(str(i), f"{i} {long_text}") for i in range(5)]

    context = ContextBuilder(token_budget=250).build(chunks)

    assert context.token_count <= 250
    assert 0 < len(context.items) < 5


def test_first_chunk_is_kept_even_if_it_exceeds_the_budget():
    chunks = [_chunk("a", " ".join(["слово"] * 500))]

    context = ContextBuilder(token_budget=10).build(chunks)

    assert len(context.items) == 1


def test_max_chunks_limits_context():
    chunks = [_chunk(str(i), f"фрагмент {i}") for i in range(20)]

    context = ContextBuilder(max_chunks=3).build(chunks)

    assert len(context.items) == 3


def test_rendered_context_carries_source_metadata():
    chunks = [_chunk("a", "Текст.", page=4, section="Отпуска")]

    context = ContextBuilder().build(chunks)

    assert "[1] policy.pdf | стр. 4 | Отпуска" in context.text


def test_refs_are_sequential_from_one():
    chunks = [_chunk(str(i), f"фрагмент {i}") for i in range(3)]

    context = ContextBuilder().build(chunks)

    assert [item.ref for item in context.items] == [1, 2, 3]


def test_empty_input_produces_empty_context():
    context = ContextBuilder().build([])

    assert context.is_empty
    assert context.text == ""
