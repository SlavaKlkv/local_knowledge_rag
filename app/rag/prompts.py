"""Промпты генерации.

Инструкция требует опираться только на контекст и явно разрешает ответ
"нет данных": выдуманный ответ хуже отсутствия ответа, потому что его
нельзя проверить по источникам.
"""

SYSTEM_PROMPT = """Ты — помощник по внутренней базе знаний организации.

Правила:
1. Отвечай только на основании фрагментов из раздела КОНТЕКСТ.
2. Не используй внешние знания и не додумывай факты.
3. Каждое утверждение подкрепляй ссылкой на номер фрагмента в формате [1], [2].
4. Если в контексте недостаточно данных для ответа, верни has_answer=false
   и пустой список citations. Не пытайся ответить приблизительно.
5. Отвечай на языке вопроса.

Верни строго JSON вида:
{"answer": "...", "has_answer": true|false, "citations": [1, 2]}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "has_answer": {"type": "boolean"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "has_answer", "citations"],
}

NO_ANSWER_TEXT = (
    "В базе знаний недостаточно данных, чтобы ответить на этот вопрос."
)


def build_user_prompt(question: str, context: str, history: str | None = None) -> str:
    parts = []
    if history:
        # История нужна для связности диалога, но источником фактов не является.
        parts.append(f"ИСТОРИЯ ДИАЛОГА (справочно, не источник фактов):\n{history}")
    parts.append(f"КОНТЕКСТ:\n{context}" if context else "КОНТЕКСТ:\n(пусто)")
    parts.append(f"ВОПРОС:\n{question}")
    return "\n\n".join(parts)
