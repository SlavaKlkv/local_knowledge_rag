"""Вопрос-ответ по базе знаний: retrieval + grounded generation с citations."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_answer_generator,
    get_context_builder,
    get_retriever,
)
from app.api.schemas import ChatRequest, ChatResponse, CitationRead
from app.rag.context_builder import ContextBuilder
from app.rag.generation import AnswerGenerator
from app.rag.retriever import DenseRetriever, RetrievalQuery

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    retriever: DenseRetriever = Depends(get_retriever),
    context_builder: ContextBuilder = Depends(get_context_builder),
    generator: AnswerGenerator = Depends(get_answer_generator),
) -> ChatResponse:
    hits = retriever.retrieve(
        RetrievalQuery(
            text=payload.question,
            knowledge_base_id=str(payload.knowledge_base_id),
            top_k=payload.top_k,
        )
    )
    context = context_builder.build(hits)
    answer = generator.generate(payload.question, context)

    return ChatResponse(
        answer=answer.text,
        has_answer=answer.has_answer,
        citations=[
            CitationRead(
                ref=c.ref,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=c.document_name,
                page=c.page,
                section=c.section,
            )
            for c in answer.citations
        ],
        model=answer.model,
        provider=answer.provider,
        latency_ms=answer.latency_ms,
    )
