"""Порядок выдачи /search: чей ранг считать итоговым.

Cross-encoder разбирает пару «запрос-фрагмент» точнее, чем retrieval, но
только там, где он обучен: на запросах-вопросах. На запросе из одного-двух
слов («командировка», «суточные») модель уходит за пределы своего
распределения и раздаёт близкие к нулю скоры почти случайно — в выдаче
всплывает фрагмент, к запросу отношения не имеющий, а точное лексическое
попадание отсекается порогом. Retrieval в этих же запросах не ошибается:
лексическая нога находит слово там, где оно действительно есть.

Поэтому режим выбирается не заранее, а по уверенности самого reranker:
насколько высоко он оценил своего лидера. Уверен — ранжирует он; не уверен —
выдача строится по retrieval и сужается до фрагментов, где слова запроса
реально встречаются. Второе заодно даёт честный ответ «ничего нет»: если
слова запроса не встречаются нигде, показывать нечего.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.rag.reranker import RerankedChunk
from app.rag.sparse import content_stems, stem_matches
from app.rag.vector_store import RetrievedChunk


class RankingMode(StrEnum):
    CROSS_ENCODER = "cross-encoder"
    LEXICAL = "lexical"


@dataclass(slots=True)
class RankedHit:
    chunk: RetrievedChunk
    score: float


@dataclass(slots=True)
class RankedSearch:
    hits: list[RankedHit]
    mode: RankingMode


def lexical_coverage(query: str, text: str) -> float:
    """Доля слов запроса, которые встречаются во фрагменте.

    Сравниваются основы, а не словоформы: «командировка» в запросе и
    «командировок» в тексте — одно и то же слово, и пользователь ждёт, что
    поиск это увидит.
    """
    query_stems = content_stems(query)
    if not query_stems:
        return 0.0
    text_stems = set(content_stems(text))
    matched = sum(
        1
        for query_stem in query_stems
        if any(stem_matches(query_stem, text_stem) for text_stem in text_stems)
    )
    return matched / len(query_stems)


def rank_search_hits(
    query: str,
    candidates: list[RetrievedChunk],
    reranked: list[RerankedChunk],
    *,
    trust_min_score: float,
    min_score: float | None,
    top_k: int,
) -> RankedSearch:
    if reranked and reranked[0].rerank_score >= trust_min_score:
        hits = [
            RankedHit(chunk=item.chunk, score=item.rerank_score)
            for item in reranked
            if min_score is None or item.rerank_score >= min_score
        ]
        return RankedSearch(hits=hits[:top_k], mode=RankingMode.CROSS_ENCODER)

    # Порядок retrieval уже отсортирован по релевантности, поэтому позиция
    # кандидата и служит тай-брейком при равном покрытии.
    scored = [
        (rank, chunk, lexical_coverage(query, chunk.text))
        for rank, chunk in enumerate(candidates)
    ]
    matched = sorted(
        (item for item in scored if item[2] > 0.0),
        key=lambda item: (-item[2], item[0]),
    )
    return RankedSearch(
        hits=[RankedHit(chunk=chunk, score=coverage) for _, chunk, coverage in matched][
            :top_k
        ],
        mode=RankingMode.LEXICAL,
    )
