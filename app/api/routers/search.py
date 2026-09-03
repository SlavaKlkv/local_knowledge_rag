"""Поиск по базе знаний без генерации ответа."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_retriever
from app.api.schemas import SearchHit, SearchRequest, SearchResponse
from app.rag.retriever import RetrievalQuery, Retriever

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest, retriever: Retriever = Depends(get_retriever)
) -> SearchResponse:
    hits = retriever.retrieve(
        RetrievalQuery(
            text=payload.query,
            knowledge_base_id=str(payload.knowledge_base_id),
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
    )
    return SearchResponse(
        query=payload.query,
        hits=[
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_name=hit.document_name,
                text=hit.text,
                score=hit.score,
                page=hit.page,
                section=hit.section,
            )
            for hit in hits
        ],
    )
