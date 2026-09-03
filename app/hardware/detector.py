"""Определение доступного оборудования.

Приложение только определяет и рекомендует — установку системных компонентов
без согласия пользователя не выполняет (это касается runtime, не детекции).
Обнаружение GPU через nvidia-smi/платформенные API — best-effort: недоступный
инструмент или неожиданный вывод не должны приводить к падению запуска,
а должны понижать оценку до CPU-only.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Callable

import psutil

from app.hardware.models import GpuInfo, GpuVendor, HardwareInfo

_CommandRunner = Callable[[list[str]], str | None]


def _run_command(args: list[str]) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - фиксированные аргументы, не пользовательский ввод
            args, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


class HardwareDetector:
    def __init__(
        self,
        run_command: _CommandRunner = _run_command,
        system: str | None = None,
        machine: str | None = None,
    ) -> None:
        self._run_command = run_command
        self._system = system or platform.system()
        self._machine = machine or platform.machine()

    def detect(self) -> HardwareInfo:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return HardwareInfo(
            cpu_count=psutil.cpu_count(logical=True) or 1,
            architecture=self._machine,
            total_ram_mb=memory.total // (1024 * 1024),
            available_ram_mb=memory.available // (1024 * 1024),
            free_disk_mb=disk.free // (1024 * 1024),
            gpu=self._detect_gpu(),
        )

    def _detect_gpu(self) -> GpuInfo:
        nvidia = self._detect_nvidia()
        if nvidia is not None:
            return nvidia
        if self._is_apple_silicon():
            return GpuInfo(vendor=GpuVendor.APPLE, name="Apple Silicon", metal_available=True)
        return GpuInfo(vendor=GpuVendor.NONE)

    def _detect_nvidia(self) -> GpuInfo | None:
        output = self._run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if not output:
            return None
        first_line = output.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) != 2:
            return None
        name, vram_raw = parts
        try:
            vram_total_mb = int(float(vram_raw))
        except ValueError:
            vram_total_mb = None
        return GpuInfo(
            vendor=GpuVendor.NVIDIA,
            name=name,
            vram_total_mb=vram_total_mb,
            cuda_available=True,
        )

    def _is_apple_silicon(self) -> bool:
        return self._system == "Darwin" and self._machine == "arm64"
