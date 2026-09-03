"""Сборка текстовой истории диалога для query rewriting и generation.

Conversation memory ≠ RAG: история нужна только для связности диалога
(local reference resolution в query rewriting) и не является источником
фактов — LLM отвечает по retrieved context, а не по прошлым сообщениям.
"""

from __future__ import annotations

from app.db.models import Message

_ROLE_LABELS = {"user": "Пользователь", "assistant": "Ассистент"}


def render_history(messages: list[Message], max_messages: int = 6) -> str | None:
    """Последние сообщения диалога в виде плоского текста.

    Порядок сохраняется хронологическим (старые -> новые): именно так
    контекст читается моделью естественнее всего.
    """
    recent = messages[-max_messages:]
    lines = [
        f"{_ROLE_LABELS.get(message.role, message.role)}: {message.content}"
        for message in recent
        if message.content.strip()
    ]
    return "\n".join(lines) if lines else None
