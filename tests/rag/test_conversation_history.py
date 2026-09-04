from app.db.models import Message
from app.rag.conversation_history import render_history


def _message(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def test_empty_history_returns_none():
    assert render_history([]) is None


def test_messages_are_rendered_in_chronological_order():
    messages = [
        _message("user", "Как изменить паспорт сотрудника?"),
        _message("assistant", "Через кадровый отдел."),
    ]

    history = render_history(messages)

    assert history == (
        "Пользователь: Как изменить паспорт сотрудника?\n"
        "Ассистент: Через кадровый отдел."
    )


def test_only_the_last_max_messages_are_kept():
    messages = [_message("user", f"вопрос {i}") for i in range(10)]

    history = render_history(messages, max_messages=3)

    assert history == "Пользователь: вопрос 7\nПользователь: вопрос 8\nПользователь: вопрос 9"


def test_blank_messages_are_skipped():
    messages = [_message("user", "  "), _message("assistant", "ответ")]

    history = render_history(messages)

    assert history == "Ассистент: ответ"
