"""Query rewriting: превращает вопрос пользователя в самостоятельный
retrieval-запрос с учётом истории диалога.

Пример из техзадания: "Как изменить паспорт сотрудника?" → "А если он
просрочен?" — второй вопрос без контекста первого нерелевантен для dense
retrieval. Переписывание выполняется локальной LLM тем же LocalLLMProvider,
чтобы не заводить отдельный inference-путь.
"""

from __future__ import annotations

import json
import re

from app.llm.base import GenerationRequest, LocalLLMProvider

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """Ты переписываешь вопрос пользователя для поиска по базе знаний.

Правила:
1. Если вопрос уже самостоятелен и понятен без истории — верни его как есть.
2. Если вопрос ссылается на контекст истории ("а если", "а он", "когда именно"
   и т.п.) — перепиши его в самостоятельный вопрос, вставив недостающий
   контекст из истории.
3. Не отвечай на вопрос и не добавляй ничего, кроме переписанного текста.
4. Сохраняй язык оригинального вопроса.

Верни строго JSON: {"rewritten": "..."}"""

_SCHEMA = {
    "type": "object",
    "properties": {"rewritten": {"type": "string"}},
    "required": ["rewritten"],
}


class QueryRewriter:
    def __init__(self, provider: LocalLLMProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    def rewrite(self, question: str, history: str | None) -> str:
        """Возвращает самостоятельный retrieval-запрос.

        Без истории переписывать нечего — простой вопрос не должен зависеть
        от лишнего обращения к LLM.
        """
        if not history or not history.strip():
            return question

        prompt = f"ИСТОРИЯ:\n{history}\n\nВОПРОС:\n{question}"
        result = self._provider.generate(
            GenerationRequest(system=_SYSTEM_PROMPT, prompt=prompt, json_schema=_SCHEMA),
            model=self._model,
        )
        rewritten = _parse_rewritten(result.text)
        # Пустой или полностью нечитаемый ответ модели не должен ломать
        # retrieval — используем исходный вопрос как безопасный fallback.
        return rewritten or question


def _parse_rewritten(text: str) -> str | None:
    for candidate in (text, _extract_json_block(text)):
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        rewritten = (payload.get("rewritten") or "").strip()
        if rewritten:
            return rewritten
    return None


def _extract_json_block(text: str) -> str | None:
    match = _JSON_BLOCK.search(text)
    return match.group(0) if match else None
