"""Локальные embeddings.

Абстракция намеренно своя: pipeline не должен зависеть от конкретного runtime,
а смена модели обязана приводить к переиндексации (меняются размерность и
пространство векторов).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from app.core.config import get_settings
from app.core.errors import InferenceError


class EmbeddingProvider(ABC):
    """Контракт провайдера эмбеддингов."""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Батчевое кодирование фрагментов документа."""

    def embed_query(self, text: str) -> list[float]:
        """Кодирование запроса. Отделено от документов: у части моделей
        различаются префиксы/инструкции для query и passage."""
        return self.embed_texts([text])[0]

    @abstractmethod
    def health_check(self) -> bool: ...

    @property
    def version(self) -> str:
        """Версия индексации: попадает в payload и позволяет находить
        векторы, устаревшие после смены модели."""
        return f"{self.model}:{self.dimension}"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Эмбеддинги через локальный Ollama."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._model = model or settings.embedding_model
        self._dimension = dimension or settings.embedding_dim
        self._timeout_s = timeout_s or settings.inference_timeout_s

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": texts},
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings")
        except httpx.HTTPError as exc:
            # Наверх уходит ошибка, а не запрос во внешний API.
            raise InferenceError(
                f"Локальная embedding-модель '{self._model}' недоступна: {exc}"
            ) from exc

        if not vectors or len(vectors) != len(texts):
            raise InferenceError(
                f"Некорректный ответ embedding-модели '{self._model}': "
                f"ожидалось {len(texts)} векторов, получено {len(vectors or [])}"
            )
        if len(vectors[0]) != self._dimension:
            raise InferenceError(
                f"Размерность вектора {len(vectors[0])} не совпадает с настроенной "
                f"{self._dimension}: требуется переиндексация или правка EMBEDDING_DIM"
            )
        return vectors

    def health_check(self) -> bool:
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        models = {item.get("name", "") for item in response.json().get("models", [])}
        return any(name.split(":")[0] == self._model.split(":")[0] for name in models)
