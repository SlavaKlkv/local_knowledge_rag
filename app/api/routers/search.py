"""Поиск по базе знаний без генерации ответа."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, require_role
from app.api.dependencies import get_retriever
from app.api.schemas import SearchHit, SearchRequest, SearchResponse
from app.db.models import PermissionRole, User
from app.db.session import get_db
from app.rag.retriever import RetrievalQuery, Retriever

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SearchResponse:
    # Права проверяются до обращения к векторному индексу: пользователь не
    # должен получать фрагменты из чужой базы знаний ни при каких условиях.
    require_role(db, user, payload.knowledge_base_id, PermissionRole.VIEWER)
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
