"""Предложение установки inference runtime'а.

Приложение не устанавливает системный компонент незаметно: сначала оно
показывает, что именно предлагается поставить и какой командой, и только
после явного подтверждения пользователя запускает установку.

Запуск установочного скрипта из веб-приложения — это выполнение
произвольной системной команды по HTTP-запросу, поэтому механизм по
умолчанию выключен (RUNTIME_INSTALL_ENABLED=false): в этом режиме
приложение отдаёт готовую команду, а выполняет её пользователь сам.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.errors import ValidationError
from app.hardware.runtime_detector import InferenceRuntime

logger = logging.getLogger("rag.runtime_installer")

_CommandRunner = Callable[[list[str]], tuple[int, str]]


def _run_command(args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(  # noqa: S603 - фиксированные аргументы, не пользовательский ввод
            args, capture_output=True, text=True, timeout=900, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


@dataclass(slots=True)
class InstallationOffer:
    """Что предлагается установить и как это сделать вручную."""

    runtime: InferenceRuntime
    supported: bool
    command: list[str] | None
    manual_command: str | None
    documentation_url: str
    note: str
    automatic_install_enabled: bool


@dataclass(slots=True)
class InstallationResult:
    runtime: InferenceRuntime
    succeeded: bool
    output: str


class RuntimeInstaller:
    """Готовит предложение установки и выполняет её только по подтверждению."""

    def __init__(
        self,
        run_command: _CommandRunner = _run_command,
        system: str | None = None,
        has_binary: Callable[[str], bool] | None = None,
    ) -> None:
        self._run_command = run_command
        self._system = system or platform.system()
        self._has_binary = has_binary or (lambda name: shutil.which(name) is not None)

    def offer(self, runtime: InferenceRuntime) -> InstallationOffer:
        automatic = get_settings().runtime_install_enabled
        if runtime == InferenceRuntime.OLLAMA:
            return self._ollama_offer(automatic)
        return InstallationOffer(
            runtime=runtime,
            supported=False,
            command=None,
            manual_command="pip install vllm",
            documentation_url="https://docs.vllm.ai/en/latest/getting_started/installation.html",
            note=(
                "vLLM ставится в окружение под конкретный GPU и версию CUDA — "
                "автоматическая установка не предлагается, выберите сборку под своё железо."
            ),
            automatic_install_enabled=False,
        )

    def _ollama_offer(self, automatic: bool) -> InstallationOffer:
        if self._system == "Darwin":
            # brew ставит ту же сборку, что и загрузка с сайта, но без
            # ручного перетаскивания приложения.
            supported = self._has_binary("brew")
            return InstallationOffer(
                runtime=InferenceRuntime.OLLAMA,
                supported=supported,
                command=["brew", "install", "ollama"] if supported else None,
                manual_command="brew install ollama",
                documentation_url="https://ollama.com/download",
                note=(
                    "Ollama — рекомендуемый runtime для локальной установки."
                    if supported
                    else "Homebrew не найден — скачайте установщик с сайта Ollama."
                ),
                automatic_install_enabled=automatic and supported,
            )
        if self._system == "Linux":
            supported = self._has_binary("sh") and self._has_binary("curl")
            return InstallationOffer(
                runtime=InferenceRuntime.OLLAMA,
                supported=supported,
                command=(
                    ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
                    if supported
                    else None
                ),
                manual_command="curl -fsSL https://ollama.com/install.sh | sh",
                documentation_url="https://ollama.com/download",
                note="Официальный установочный скрипт Ollama.",
                automatic_install_enabled=automatic and supported,
            )
        return InstallationOffer(
            runtime=InferenceRuntime.OLLAMA,
            supported=False,
            command=None,
            manual_command=None,
            documentation_url="https://ollama.com/download",
            note=f"Автоматическая установка для {self._system} не поддерживается.",
            automatic_install_enabled=False,
        )

    def install(self, runtime: InferenceRuntime, confirmed: bool) -> InstallationResult:
        """Ставит runtime. Без confirmed=True не делает ничего."""
        if not confirmed:
            raise ValidationError(
                "Установка системного компонента требует явного подтверждения "
                "(confirm=true)"
            )

        offer = self.offer(runtime)
        if not offer.automatic_install_enabled or offer.command is None:
            raise ValidationError(
                f"Автоматическая установка '{runtime}' недоступна. "
                f"{offer.note} Команда для ручной установки: {offer.manual_command}"
            )

        logger.info("runtime_install_started", extra={"runtime": str(runtime)})
        code, output = self._run_command(offer.command)
        succeeded = code == 0
        logger.info(
            "runtime_install_finished",
            extra={"runtime": str(runtime), "succeeded": succeeded},
        )
        return InstallationResult(runtime=runtime, succeeded=succeeded, output=output.strip())
