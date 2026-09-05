"""Метрики качества ответа: обоснованность, цитаты и честный отказ.

Retrieval-метрики отвечают на вопрос «нашли ли нужное». Здесь измеряется то,
что происходит дальше: опирается ли ответ на найденное, ведут ли цитаты к
нужным документам и умеет ли система молчать, когда ответа нет.

Все проверки здесь детерминированные и не требуют модели. Это сознательное
ограничение: они не понимают смысл и не заменяют оценку судьёй, зато дают
воспроизводимый сигнал, который не плавает от запуска к запуску и не стоит
секунд инференса на каждый пример. Смысловая оценка — отдельная ступень.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.evaluation.dataset import EvaluationExample
from app.rag.generation import Answer

# Числа — самый опасный вид выдумки: «28 дней» вместо «14» выглядит столь же
# уверенно, но меняет смысл целиком. Поэтому они проверяются отдельно от слов.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Короткие слова — предлоги, союзы и окончания служебных конструкций; их
# совпадение ничего не говорит об опоре ответа на источник.
_MIN_WORD_LENGTH = 4


def _content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in _WORD.findall(text)
        if len(word) >= _MIN_WORD_LENGTH
    }


def _numbers(text: str) -> set[str]:
    return {match.replace(",", ".") for match in _NUMBER.findall(text)}


def lexical_groundedness(answer_text: str, cited_texts: list[str]) -> float:
    """Доля значимых слов ответа, встречающихся в процитированных фрагментах.

    Это нижняя оценка обоснованности, а не мера истинности: перефразирование
    и словоизменение снижают её, не будучи выдумкой. Смысл метрики —
    в сравнении конфигураций и в отлове резких провалов, когда модель
    начинает сочинять поверх контекста.
    """
    words = _content_words(answer_text)
    if not words:
        return 0.0
    source = _content_words(" ".join(cited_texts))
    return len(words & source) / len(words)


def unsupported_numbers(answer_text: str, cited_texts: list[str]) -> set[str]:
    """Числа из ответа, которых нет в процитированных фрагментах."""
    return _numbers(answer_text) - _numbers(" ".join(cited_texts))


@dataclass(slots=True, frozen=True)
class AnswerScores:
    example_id: str
    answered: bool
    # Для примеров без ответа проверяется только корректность отказа,
    # поэтому остальные метрики там не определены.
    correct_abstention: bool | None
    hallucinated_answer: bool
    citation_precision: float | None
    citation_recall: float | None
    groundedness: float | None
    unsupported_numbers: tuple[str, ...]


def score_answer(
    answer: Answer, example: EvaluationExample, cited_texts: list[str] | None = None
) -> AnswerScores:
    """Считает метрики одного ответа относительно эталонной разметки."""
    cited_documents = {citation.document_id for citation in answer.citations}
    texts = cited_texts or []

    if not example.has_answer:
        # Вопрос без ответа: любой ответ по существу — выдумка, и никакие
        # метрики цитат для него не осмысленны.
        return AnswerScores(
            example_id=example.id,
            answered=answer.has_answer,
            correct_abstention=not answer.has_answer,
            hallucinated_answer=answer.has_answer,
            citation_precision=None,
            citation_recall=None,
            groundedness=None,
            unsupported_numbers=(),
        )

    if not answer.has_answer:
        # Отказ там, где ответ есть, — не выдумка, но и не успех:
        # цитаты отсутствуют, значит их точность равна нулю, а не не определена.
        return AnswerScores(
            example_id=example.id,
            answered=False,
            correct_abstention=None,
            hallucinated_answer=False,
            citation_precision=0.0,
            citation_recall=0.0,
            groundedness=0.0,
            unsupported_numbers=(),
        )

    relevant = example.relevant_ids
    return AnswerScores(
        example_id=example.id,
        answered=True,
        correct_abstention=None,
        hallucinated_answer=False,
        citation_precision=(
            len(cited_documents & relevant) / len(cited_documents) if cited_documents else 0.0
        ),
        citation_recall=len(cited_documents & relevant) / len(relevant),
        groundedness=lexical_groundedness(answer.text, texts),
        unsupported_numbers=tuple(sorted(unsupported_numbers(answer.text, texts))),
    )
