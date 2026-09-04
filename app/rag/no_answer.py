"""Политика честного отказа.

Измерения показали, что retrieval сам по себе молчать не умеет: косинусная
близость не бывает нулевой, поэтому на вопрос, ответа на который в базе нет,
он всё равно возвращает документы. Значит, решение «ответа нет» принимается
здесь — по собранному контексту и по тому, что вернула модель.

Политика намеренно вынесена из AnswerGenerator отдельным объектом: её условия
настраиваются, участвуют в evaluation и должны быть проверяемы без запуска
модели.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from app.rag.context_builder import BuiltContext


class NoAnswerCode(enum.StrEnum):
    """Причина отказа в виде ограниченного набора значений.

    Нужна отдельно от текста: в тексте есть числа (скор, порог), и в метке
    Prometheus он дал бы неограниченную кардинальность. Код — для метрик и
    ветвлений, текст — для человека.
    """

    EMPTY_CONTEXT = "empty_context"
    BELOW_THRESHOLD = "below_threshold"
    MODEL_DECLINED = "model_declined"
    NO_CITATIONS = "no_citations"


@dataclass(slots=True, frozen=True)
class NoAnswerDecision:
    refuse: bool
    code: NoAnswerCode | None = None
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class NoAnswerPolicy:
    """Условия, при которых система обязана промолчать.

    Порога здесь сознательно нет. Отсечение по скору выполняет retrieval, а не
    политика: скор чанка живёт в разных шкалах у разных стратегий — у dense это
    косинусная близость, а у hybrid уже RRF, где типичное значение около 0.016.
    Один и тот же порог значил бы в них совершенно разное, и сравнение с
    `chunk.score` после фьюжна молча врало бы. Порог задаётся в запросе к
    retrieval (`RetrievalQuery.score_threshold`), где он попадает в dense-ногу
    в своей родной шкале.

    `require_citations` по умолчанию включён: ответ, не сославшийся ни на один
    реальный фрагмент, проверить нечем, а непроверяемый ответ — ровно то, чего
    RAG должен избегать.
    """

    require_citations: bool = True

    def before_generation(
        self, context: BuiltContext, threshold_applied: bool = False
    ) -> NoAnswerDecision:
        """Проверка до обращения к модели.

        Отказ на этой стадии не только честнее, но и дешевле: генерация по
        заведомо непригодному контексту тратит секунды локального инференса.

        `threshold_applied` говорит, искали ли с порогом: пустой результат
        поиска с порогом и без него — разные ситуации, и в отчёте их надо
        различать, иначе «настроен слишком строгий порог» не отличить от
        «в базе действительно ничего нет».
        """
        if context.is_empty:
            if threshold_applied:
                return NoAnswerDecision(
                    True,
                    NoAnswerCode.BELOW_THRESHOLD,
                    "ничего не найдено выше порога релевантности",
                )
            return NoAnswerDecision(
                True, NoAnswerCode.EMPTY_CONTEXT, "в базе знаний ничего не найдено"
            )
        return NoAnswerDecision(False)

    def after_generation(self, has_answer: bool, citation_count: int) -> NoAnswerDecision:
        """Проверка того, что вернула модель."""
        if not has_answer:
            return NoAnswerDecision(
                True, NoAnswerCode.MODEL_DECLINED, "модель сообщила, что ответа нет"
            )
        if self.require_citations and citation_count == 0:
            return NoAnswerDecision(
                True,
                NoAnswerCode.NO_CITATIONS,
                "ответ не сослался ни на один действительный фрагмент",
            )
        return NoAnswerDecision(False)
