"""Определение доступных inference runtime'ов.

Ollama — основной рекомендуемый runtime для локальной установки; если она
не найдена, приложение только предлагает установку — не делает этого молча.
vLLM — альтернатива для более производительного server/GPU deployment.
Если доступны оба, выбор всё равно остаётся за пользователем: детектор
только рекомендует.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import httpx


class InferenceRuntime(enum.StrEnum):
    OLLAMA = "ollama"
    VLLM = "vllm"


@dataclass(slots=True)
class RuntimeAvailability:
    runtime: InferenceRuntime
    available: bool
    base_url: str
    detail: str | None = None


@dataclass(slots=True)
class RuntimeDetectionResult:
    runtimes: list[RuntimeAvailability]

    @property
    def recommended(self) -> InferenceRuntime | None:
        """Ollama рекомендуется первой при наличии — она проще в установке
        и остаётся дефолтным путём для локальной установки."""
        available = {r.runtime: r for r in self.runtimes if r.available}
        if InferenceRuntime.OLLAMA in available:
            return InferenceRuntime.OLLAMA
        if InferenceRuntime.VLLM in available:
            return InferenceRuntime.VLLM
        return None

    def is_available(self, runtime: InferenceRuntime) -> bool:
        return any(r.runtime == runtime and r.available for r in self.runtimes)


class RuntimeDetector:
    def __init__(
        self,
        ollama_url: str,
        vllm_url: str,
        timeout_s: float = 3.0,
        http_get=httpx.get,
    ) -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._vllm_url = vllm_url.rstrip("/")
        self._timeout_s = timeout_s
        self._http_get = http_get

    def detect(self) -> RuntimeDetectionResult:
        return RuntimeDetectionResult(
            runtimes=[
                self._check(InferenceRuntime.OLLAMA, f"{self._ollama_url}/api/version"),
                self._check(InferenceRuntime.VLLM, f"{self._vllm_url}/health"),
            ]
        )

    def _check(self, runtime: InferenceRuntime, health_url: str) -> RuntimeAvailability:
        base_url = self._ollama_url if runtime == InferenceRuntime.OLLAMA else self._vllm_url
        try:
            response = self._http_get(health_url, timeout=self._timeout_s)
        except httpx.HTTPError as exc:
            return RuntimeAvailability(
                runtime=runtime, available=False, base_url=base_url, detail=str(exc)
            )
        if response.status_code >= 400:
            return RuntimeAvailability(
                runtime=runtime,
                available=False,
                base_url=base_url,
                detail=f"HTTP {response.status_code}",
            )
        return RuntimeAvailability(runtime=runtime, available=True, base_url=base_url)
