"""Провайдер локального inference через Ollama."""

from __future__ import annotations

import time

import httpx

from app.core.config import get_settings
from app.core.errors import InferenceError
from app.llm.base import (
    GenerationRequest,
    GenerationResult,
    LocalLLMProvider,
    ModelInfo,
)


class OllamaProvider(LocalLLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_url).rstrip("/")
        self._timeout_s = timeout_s or settings.inference_timeout_s

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        payload: dict = {
            "model": model,
            "system": request.system,
            "prompt": request.prompt,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens
        if request.json_schema:
            # Structured output: модель обязана вернуть валидный JSON,
            # иначе разбор citations становится ненадёжным.
            payload["format"] = request.json_schema

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate", json=payload, timeout=self._timeout_s
            )
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise InferenceError(
                f"Модель '{model}' не ответила за {self._timeout_s} с"
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError(f"Ошибка Ollama при генерации моделью '{model}': {exc}") from exc

        text = (body.get("response") or "").strip()
        if not text:
            raise InferenceError(f"Модель '{model}' вернула пустой ответ")

        return GenerationResult(
            text=text,
            model=model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            meta={"done_reason": body.get("done_reason")},
        )

    def health_check(self) -> bool:
        try:
            httpx.get(f"{self._base_url}/api/version", timeout=5.0).raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def list_models(self) -> list[ModelInfo]:
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Ollama недоступна по адресу {self._base_url}: {exc}") from exc

        models = []
        for item in response.json().get("models", []):
            details = item.get("details") or {}
            models.append(
                ModelInfo(
                    name=item.get("name", ""),
                    provider=self.name,
                    size_bytes=item.get("size"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                )
            )
        return models
