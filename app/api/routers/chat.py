"""Вопрос-ответ по базе знаний: retrieval + grounded generation с citations.

Диалог опционален: без conversation_id запрос работает как раньше —
одиночный stateless вопрос. С conversation_id подключается история
для query rewriting, а вопрос и ответ сохраняются как сообщения диалога.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, require_role
from app.api.dependencies import (
    get_answer_generator,
    get_context_builder,
    get_query_rewriter,
    get_reranker,
    get_retriever,
)
from app.api.schemas import ChatRequest, ChatResponse, CitationRead
from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.db.models import Conversation, Message, PermissionRole, User
from app.db.session import get_db
from app.observability.events import traced_query
from app.rag.context_builder import ContextBuilder
from app.rag.conversation_history import render_history
from app.rag.generation import AnswerGenerator
from app.rag.query_rewriting import QueryRewriter
from app.rag.reranker import Reranker
from app.rag.retriever import RetrievalQuery, Retriever

router = APIRouter(prefix="/chat", tags=["chat"])


def _load_conversation(payload: ChatRequest, db: Session) -> Conversation | None:
    if payload.conversation_id is None:
        return None
    conversation = db.get(Conversation, payload.conversation_id)
    if conversation is None:
        raise NotFoundError(f"Диалог {payload.conversation_id} не найден")
    if str(conversation.knowledge_base_id) != str(payload.knowledge_base_id):
        raise ValidationError(
            "Диалог принадлежит другой базе знаний — изоляция баз знаний "
            "распространяется и на conversation_id"
        )
    return conversation


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    retriever: Retriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
    context_builder: ContextBuilder = Depends(get_context_builder),
    generator: AnswerGenerator = Depends(get_answer_generator),
    query_rewriter: QueryRewriter = Depends(get_query_rewriter),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    # Тот же фильтр, что и в /search: недостаточно скрыть базу знаний в
    # списке — её фрагменты не должны попадать даже в контекст ответа.
    require_role(db, user, payload.knowledge_base_id, PermissionRole.VIEWER)
    settings = get_settings()
    conversation = _load_conversation(payload, db)

    history = None
    if conversation is not None:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at)
        )
        history = render_history(list(db.scalars(stmt)))

    with traced_query(str(payload.knowledge_base_id), payload.question) as trace:
        # Переписывание запроса опирается только на историю диалога:
        # без conversation_id она отсутствует, и запрос уходит как есть.
        retrieval_query = query_rewriter.rewrite(payload.question, history)
        trace.rewritten_query = (
            retrieval_query if retrieval_query != payload.question else None
        )

        # Кандидатов берём с запасом относительно итогового top_k: reranker
        # работает точнее dense-поиска именно на более широком наборе.
        retrieval_started = time.perf_counter()
        # Порог отсекает нерелевантное там, где скор ещё в своей шкале:
        # после RRF-фьюжна сравнивать его с косинусным порогом бессмысленно.
        candidates = retriever.retrieve(
            RetrievalQuery(
                text=retrieval_query,
                knowledge_base_id=str(payload.knowledge_base_id),
                top_k=max(payload.top_k, settings.rerank_candidates),
                score_threshold=settings.no_answer_min_score,
            )
        )
        trace.retrieved_chunk_ids = [hit.chunk_id for hit in candidates]
        trace.retrieval_scores = [hit.score for hit in candidates]

        reranked = reranker.rerank(
            retrieval_query, candidates, top_k=min(payload.top_k, settings.rerank_top_k)
        )
        trace.reranking_scores = [item.rerank_score for item in reranked]
        # Замер включает reranking: с точки зрения «куда уходит время» поиск
        # кандидатов и их пересортировка — одна ступень, и отделять её нужно
        # не друг от друга, а от генерации.
        trace.retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        hits = [item.chunk for item in reranked]

        context = context_builder.build(hits)
        # Генерация отвечает на исходный вопрос пользователя, а не на
        # переписанный retrieval-запрос — переписывание нужно только поиску.
        answer = generator.generate(
            payload.question,
            context,
            history=history,
            threshold_applied=settings.no_answer_min_score is not None,
        )

        trace.llm_model = answer.model
        trace.llm_provider = answer.provider
        trace.has_answer = answer.has_answer
        trace.no_answer_reason = answer.no_answer_reason
        trace.no_answer_code = answer.no_answer_code
        trace.llm_latency_ms = answer.latency_ms

    citations = [
        CitationRead(
            ref=c.ref,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_name=c.document_name,
            page=c.page,
            section=c.section,
        )
        for c in answer.citations
    ]

    if conversation is not None:
        db.add(Message(conversation_id=conversation.id, role="user", content=payload.question))
        db.add(
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content=answer.text,
                citations=[c.model_dump() for c in citations],
                meta={"model": answer.model, "provider": answer.provider},
            )
        )
        db.commit()

    return ChatResponse(
        answer=answer.text,
        has_answer=answer.has_answer,
        citations=citations,
        model=answer.model,
        provider=answer.provider,
        latency_ms=answer.latency_ms,
        conversation_id=conversation.id if conversation is not None else None,
        rewritten_query=trace.rewritten_query,
        no_answer_reason=answer.no_answer_reason,
    )
