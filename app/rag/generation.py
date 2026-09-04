"""Генерация grounded-ответа с цитатами."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.llm.base import GenerationRequest, LocalLLMProvider
from app.rag.context_builder import BuiltContext
from app.rag.no_answer import NoAnswerPolicy
from app.rag.prompts import (
    ANSWER_SCHEMA,
    NO_ANSWER_TEXT,
    SYSTEM_PROMPT,
    build_user_prompt,
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class Citation:
    ref: int
    chunk_id: str
    document_id: str
    document_name: str | None
    page: int | None
    section: str | None


@dataclass(slots=True)
class Answer:
    text: str
    has_answer: bool
    citations: list[Citation] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
    # Почему именно система промолчала: без этого отказ неотличим от сбоя
    # ни в логах, ни при разборе жалобы «он ничего не ответил».
    no_answer_reason: str | None = None
    no_answer_code: str | None = None


class AnswerGenerator:
    def __init__(
        self,
        provider: LocalLLMProvider,
        model: str,
        policy: NoAnswerPolicy | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._policy = policy or NoAnswerPolicy()

    def generate(
        self,
        question: str,
        context: BuiltContext,
        history: str | None = None,
        threshold_applied: bool = False,
    ) -> Answer:
        decision = self._policy.before_generation(context, threshold_applied)
        if decision.refuse:
            # Без пригодного контекста запрос к модели бессмыслен: любой ответ
            # будет неподтверждённым, а инференс — потраченным впустую.
            return Answer(
                text=NO_ANSWER_TEXT,
                has_answer=False,
                no_answer_reason=decision.reason,
                no_answer_code=decision.code,
            )

        result = self._provider.generate(
            GenerationRequest(
                system=SYSTEM_PROMPT,
                prompt=build_user_prompt(question, context.text, history),
                json_schema=ANSWER_SCHEMA,
            ),
            model=self._model,
        )
        payload = _parse_payload(result.text)
        answer_text = (payload.get("answer") or "").strip()
        citations = resolve_citations(payload.get("citations") or [], context)
        decision = self._policy.after_generation(
            has_answer=bool(payload.get("has_answer")) and bool(answer_text),
            citation_count=len(citations),
        )
        if decision.refuse:
            return Answer(
                text=NO_ANSWER_TEXT,
                has_answer=False,
                model=result.model,
                provider=result.provider,
                latency_ms=result.latency_ms,
                no_answer_reason=decision.reason,
                no_answer_code=decision.code,
            )

        return Answer(
            text=answer_text,
            has_answer=True,
            citations=citations,
            model=result.model,
            provider=result.provider,
            latency_ms=result.latency_ms,
        )


def resolve_citations(refs: list, context: BuiltContext) -> list[Citation]:
    """Сопоставляет номера ссылок из ответа с реальными фрагментами.

    Номера, которых не было в контексте, отбрасываются: цитата должна вести
    к проверяемому источнику, а не к галлюцинации модели.
    """
    by_ref = {item.ref: item.chunk for item in context.items}
    citations: list[Citation] = []
    seen: set[int] = set()
    for raw in refs:
        try:
            ref = int(raw)
        except (TypeError, ValueError):
            continue
        chunk = by_ref.get(ref)
        if chunk is None or ref in seen:
            continue
        seen.add(ref)
        citations.append(
            Citation(
                ref=ref,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page=chunk.page,
                section=chunk.section,
            )
        )
    return citations


def _parse_payload(text: str) -> dict:
    """Разбор структурированного ответа.

    Модели периодически добавляют пояснения вокруг JSON, поэтому сначала
    пробуем строгий разбор, затем — первый JSON-объект в тексте.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Ответ не разобран — трактуем как отсутствие подтверждённого ответа.
    return {"answer": "", "has_answer": False, "citations": []}
