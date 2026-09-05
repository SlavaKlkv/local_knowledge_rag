"""Структуры данных аппаратного профиля."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class GpuVendor(enum.StrEnum):
    NVIDIA = "nvidia"
    APPLE = "apple"
    NONE = "none"


@dataclass(slots=True)
class GpuInfo:
    vendor: GpuVendor
    name: str | None = None
    vram_total_mb: int | None = None
    cuda_available: bool = False
    metal_available: bool = False


@dataclass(slots=True)
class HardwareInfo:
    """Снимок аппаратных характеристик машины на момент детекции."""

    cpu_count: int
    architecture: str
    total_ram_mb: int
    available_ram_mb: int
    free_disk_mb: int
    gpu: GpuInfo

    @property
    def is_apple_silicon(self) -> bool:
        return self.gpu.vendor == GpuVendor.APPLE

    @property
    def is_cpu_only(self) -> bool:
        return self.gpu.vendor == GpuVendor.NONE
