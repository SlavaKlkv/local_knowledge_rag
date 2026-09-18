"""Выбор режима выдачи /search.

Cross-encoder надёжен на запросах-вопросах и разваливается на запросах из
одного-двух слов: там его скоры почти случайны, и порядок надо строить по
retrieval и лексическому совпадению.
"""

import pytest

from app.rag.reranker import RerankedChunk
from app.rag.search_ranking import (
    RankingMode,
    lexical_coverage,
    rank_search_hits,
)
from app.rag.vector_store import RetrievedChunk

TRUST = 0.3
MIN_SCORE = 0.2


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        document_name="doc.md",
        text=text,
        score=0.03,
        chunk_index=0,
    )


def _rank(query, candidates, scores, top_k=10):
    reranked = sorted(
        (RerankedChunk(chunk=c, rerank_score=scores[c.chunk_id]) for c in candidates),
        key=lambda item: item.rerank_score,
        reverse=True,
    )
    return rank_search_hits(
        query,
        candidates,
        reranked,
        trust_min_score=TRUST,
        min_score=MIN_SCORE,
        top_k=top_k,
    )


class TestLexicalCoverage:
    def test_counts_words_regardless_of_word_form(self):
        assert lexical_coverage("командировка", "Политика командировок") == 1.0

    def test_does_not_match_a_different_word_with_the_same_prefix(self):
        # «командировк» и «командн» начинаются одинаково, но это разные слова.
        assert lexical_coverage("командировка", "Дни командных встреч") == 0.0

    def test_ignores_function_words(self):
        # Иначе «как» находилось бы в каждом втором фрагменте базы.
        assert lexical_coverage("Как оформить ипотеку?", "Как выдают ноутбук") == 0.0

    def test_partial_match_scores_below_full_match(self):
        text = "Лимиты расходов: проживание до 6000 рублей"
        assert lexical_coverage("лимиты расходов", text) == 1.0
        assert lexical_coverage("лимиты командировки", text) == pytest.approx(0.5)


class TestConfidentReranker:
    def test_reranker_decides_the_order(self):
        candidates = [_chunk("c1", "Первый"), _chunk("c2", "Второй")]
        ranked = _rank("Как компенсируется дежурство?", candidates, {"c1": 0.4, "c2": 0.9})

        assert ranked.mode is RankingMode.CROSS_ENCODER
        assert [hit.chunk.chunk_id for hit in ranked.hits] == ["c2", "c1"]
        assert ranked.hits[0].score == pytest.approx(0.9)

    def test_low_scoring_chunks_are_cut_off(self):
        candidates = [_chunk("c1", "Первый"), _chunk("c2", "Второй")]
        ranked = _rank("Как компенсируется дежурство?", candidates, {"c1": 0.05, "c2": 0.9})

        assert [hit.chunk.chunk_id for hit in ranked.hits] == ["c2"]


class TestUnsureReranker:
    """Скоры около нуля у всех кандидатов: модель вне своей области."""

    def test_falls_back_to_lexical_match(self):
        # Cross-encoder ставит первым фрагмент, где слова запроса нет вовсе.
        relevant = _chunk("c1", "Командировка согласовывается руководителем")
        noise = _chunk("c2", "Дежурство в выходной компенсируется отгулом")
        ranked = _rank("командировка", [relevant, noise], {"c1": 0.09, "c2": 0.27})

        assert ranked.mode is RankingMode.LEXICAL
        assert [hit.chunk.chunk_id for hit in ranked.hits] == ["c1"]
        assert ranked.hits[0].score == pytest.approx(1.0)

    def test_returns_nothing_when_the_words_are_absent(self):
        candidates = [_chunk("c1", "Дежурство в выходной компенсируется отгулом")]
        ranked = _rank("отпуск", candidates, {"c1": 0.28})

        assert ranked.hits == []

    def test_full_match_outranks_partial_one(self):
        full = _chunk("c1", "Лимиты расходов в командировке")
        partial = _chunk("c2", "Лимиты времени на ответ")
        ranked = _rank("лимиты расходов", [full, partial], {"c1": 0.02, "c2": 0.03})

        assert [hit.chunk.chunk_id for hit in ranked.hits] == ["c1", "c2"]
        assert ranked.hits[0].score > ranked.hits[1].score

    def test_retrieval_order_breaks_the_tie(self):
        first = _chunk("c1", "Суточные 1500 рублей")
        second = _chunk("c2", "Кроме суточных, расходы подтверждаются")
        ranked = _rank("суточные", [first, second], {"c1": 0.03, "c2": 0.04})

        assert [hit.chunk.chunk_id for hit in ranked.hits] == ["c1", "c2"]

    def test_top_k_limits_the_output(self):
        candidates = [_chunk(f"c{i}", "Командировка согласовывается") for i in range(5)]
        scores = dict.fromkeys([f"c{i}" for i in range(5)], 0.05)
        ranked = _rank("командировка", candidates, scores, top_k=2)

        assert len(ranked.hits) == 2
