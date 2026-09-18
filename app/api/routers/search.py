"""Поиск по базе знаний без генерации ответа."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, require_role
from app.api.dependencies import get_reranker, get_retriever
from app.api.schemas import SearchHit, SearchRequest, SearchResponse
from app.core.config import get_settings
from app.db.models import PermissionRole, User
from app.db.session import get_db
from app.rag.reranker import NoOpReranker, Reranker
from app.rag.retriever import RetrievalQuery, Retriever
from app.rag.search_ranking import rank_search_hits

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SearchResponse:
    # Права проверяются до обращения к векторному индексу: пользователь не
    # должен получать фрагменты из чужой базы знаний ни при каких условиях.
    require_role(db, user, payload.knowledge_base_id, PermissionRole.VIEWER)
    settings = get_settings()

    # Кандидатов берём с запасом, как и в /chat: reranker точнее retrieval
    # именно на широком наборе, а до пользователя дойдёт только top_k.
    candidates = retriever.retrieve(
        RetrievalQuery(
            text=payload.query,
            knowledge_base_id=str(payload.knowledge_base_id),
            top_k=max(payload.top_k, settings.rerank_candidates),
            score_threshold=payload.score_threshold,
        )
    )
    reranked = reranker.rerank(payload.query, candidates, top_k=len(candidates))

    # Retrieval всегда возвращает top_k ближайших, даже когда в базе про
    # запрос нет ничего: без отсечки выдача заполняется случайными
    # фрагментами со скором, по которому этого не видно. Порог применим
    # только к настоящему reranker — у NoOp скор приходит из retrieval
    # (RRF или косинус) и с порогом cross-encoder несопоставим.
    threshold = payload.min_score
    if threshold is None and not isinstance(reranker, NoOpReranker):
        threshold = settings.search_min_rerank_score

    # Порядок выдачи выбирается по уверенности reranker: на запросе-вопросе
    # ранжирует он, на коротком запросе — retrieval и лексика. У NoOp своя
    # шкала, доверять ей как cross-encoder нельзя, поэтому порог доверия
    # снимается вместе с отсечкой.
    ranked = rank_search_hits(
        payload.query,
        candidates,
        reranked,
        trust_min_score=(
            float("-inf")
            if isinstance(reranker, NoOpReranker)
            else settings.rerank_trust_min_score
        ),
        min_score=threshold,
        top_k=payload.top_k,
    )

    return SearchResponse(
        query=payload.query,
        ranking=ranked.mode,
        hits=[
            SearchHit(
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                document_name=item.chunk.document_name,
                text=item.chunk.text,
                # Скор — из той шкалы, которая и определила порядок: оценка
                # cross-encoder либо доля слов запроса, найденных во
                # фрагменте. Какая именно, сказано в поле ranking.
                score=item.score,
                page=item.chunk.page,
                section=item.chunk.section,
            )
            for item in ranked.hits
        ],
    )
