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
        except httpx.TimeoutException:
            return RuntimeAvailability(
                runtime=runtime,
                available=False,
                base_url=base_url,
                detail=(
                    f"Не ответил за {self._timeout_s:.0f} с по адресу {base_url} — "
                    "запущен, но перегружен или завис"
                ),
            )
        except httpx.HTTPError:
            return RuntimeAvailability(
                runtime=runtime,
                available=False,
                base_url=base_url,
                detail=f"Не отвечает по адресу {base_url} — похоже, не запущен",
            )
        if response.status_code >= 400:
            return RuntimeAvailability(
                runtime=runtime,
                available=False,
                base_url=base_url,
                detail=self._bad_status_detail(runtime, base_url, response.status_code),
            )
        return RuntimeAvailability(runtime=runtime, available=True, base_url=base_url)

    @staticmethod
    def _bad_status_detail(runtime: InferenceRuntime, base_url: str, status: int) -> str:
        """Ответ пришёл, но не тот: по адресу кто-то слушает.

        Код состояния остаётся в тексте — без него не понять, чинить ли
        сам сервис или искать, кто занял порт, — но ведущей становится
        причина, а не номер.
        """
        if status == 404:
            return (
                f"По адресу {base_url} отвечает не {runtime} — порт занят другим "
                f"сервисом или адрес указан неверно (HTTP {status})"
            )
        if status >= 500:
            return (
                f"{runtime} по адресу {base_url} отвечает с ошибкой — запущен, "
                f"но не готов принимать запросы (HTTP {status})"
            )
        return (
            f"{runtime} по адресу {base_url} отклонил проверку доступности "
            f"(HTTP {status})"
        )
