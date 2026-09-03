"""Доменные ошибки приложения."""


class AppError(Exception):
    """Базовая ошибка приложения."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class InferenceError(AppError):
    """Локальный inference недоступен или вернул некорректный ответ.

    Осознанно завершает запрос ошибкой: обращение к внешнему AI-провайдеру
    в качестве fallback недопустимо.
    """

    status_code = 503
    code = "inference_error"
