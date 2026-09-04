import pytest

from app.core.errors import ValidationError
from app.ingestion.chunking import get_chunking_strategy
from app.ingestion.models import ParsedBlock, ParsedDocument


def _document(text: str, **kwargs) -> ParsedDocument:
    return ParsedDocument(blocks=[ParsedBlock(text=text, **kwargs)])


def test_fixed_chunking_respects_size_and_overlap():
    document = _document(" ".join(f"w{i}" for i in range(25)))

    chunks = get_chunking_strategy("fixed", chunk_size=10, chunk_overlap=2).split(document)

    assert [chunk.token_count for chunk in chunks] == [10, 10, 9]
    # Перекрытие: последние два слова чанка повторяются в начале следующего.
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_fixed_chunking_indexes_are_sequential():
    document = _document(" ".join(f"w{i}" for i in range(50)))

    chunks = get_chunking_strategy("fixed", chunk_size=10, chunk_overlap=0).split(document)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_recursive_chunking_does_not_split_inside_a_sentence():
    text = "Первое предложение тут. Второе предложение тут. Третье предложение тут."

    chunks = get_chunking_strategy(
        "recursive", chunk_size=8, chunk_overlap=0
    ).split(_document(text))

    assert all(chunk.text.endswith(".") for chunk in chunks)
    assert len(chunks) > 1


def test_oversized_sentence_is_force_split():
    text = " ".join(f"w{i}" for i in range(30)) + "."

    chunks = get_chunking_strategy(
        "recursive", chunk_size=10, chunk_overlap=0
    ).split(_document(text))

    assert max(chunk.token_count for chunk in chunks) <= 10


def test_chunks_keep_page_and_section_for_citations():
    document = _document("Текст раздела.", page=7, section="Отпуска")

    chunk = get_chunking_strategy("recursive").split(document)[0]

    assert (chunk.page, chunk.section) == (7, "Отпуска")


@pytest.mark.parametrize(
    ("size", "overlap"), [(0, 0), (10, 10), (10, 15), (10, -1)]
)
def test_invalid_chunking_parameters_are_rejected(size, overlap):
    with pytest.raises(ValidationError):
        get_chunking_strategy("fixed", chunk_size=size, chunk_overlap=overlap)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValidationError, match="Неизвестная стратегия"):
        get_chunking_strategy("magic")
