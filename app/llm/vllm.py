"""Провайдер локального inference через vLLM (OpenAI-совместимый API).

vLLM — альтернативный runtime прежде всего для более производительного
server/GPU deployment. RAG pipeline не меняется от выбора провайдера —
единый интерфейс LocalLLMProvider отвечает за это.
"""

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


class VLLMProvider(LocalLLMProvider):
    name = "vllm"

    def __init__(self, base_url: str | None = None, timeout_s: float | None = None) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.vllm_url).rstrip("/")
        self._timeout_s = timeout_s or settings.inference_timeout_s

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.json_schema:
            # OpenAI-совместимый structured output: guided_json — расширение
            # vLLM поверх стандартного chat completions API.
            payload["extra_body"] = {"guided_json": request.json_schema}

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/v1/chat/completions", json=payload, timeout=self._timeout_s
            )
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as exc:
            raise InferenceError(
                f"Модель '{model}' не ответила за {self._timeout_s} с"
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError(f"Ошибка vLLM при генерации моделью '{model}': {exc}") from exc

        choices = body.get("choices") or []
        text = (choices[0].get("message", {}).get("content") or "").strip() if choices else ""
        if not text:
            raise InferenceError(f"Модель '{model}' вернула пустой ответ")

        usage = body.get("usage") or {}
        return GenerationResult(
            text=text,
            model=model,
            provider=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            meta={"finish_reason": choices[0].get("finish_reason") if choices else None},
        )

    def health_check(self) -> bool:
        try:
            httpx.get(f"{self._base_url}/health", timeout=5.0).raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def list_models(self) -> list[ModelInfo]:
        try:
            response = httpx.get(f"{self._base_url}/v1/models", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"vLLM недоступен по адресу {self._base_url}: {exc}") from exc

        return [
            ModelInfo(name=item.get("id", ""), provider=self.name)
            for item in response.json().get("data", [])
        ]
