"""Профили моделей и их рекомендация по железу.

Три профиля из техзадания — LIGHT/STANDARD/PERFORMANCE. Размер модели
подбирается по собственной линейке каждого семейства: у Qwen, Gemma и Llama
разные шаги параметров, поэтому нельзя просто взять "одинаковое число
миллиардов параметров" для всех трёх — конкретные размеры зашиты явно.

Кольцевой fallback между семействами (Qwen → Gemma → Llama → Qwen → ...)
непрерывен на уровне сервиса — здесь описан только состав кольца для
выбранного профиля; порядок обхода реализует HealthAwareModelRing (V3).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from app.hardware.models import GpuVendor, HardwareInfo


class HardwareProfile(enum.StrEnum):
    LIGHT = "light"
    STANDARD = "standard"
    PERFORMANCE = "performance"


@dataclass(slots=True)
class ModelRingEntry:
    family: str
    model: str


@dataclass(slots=True)
class ProfileDefinition:
    profile: HardwareProfile
    ring: list[ModelRingEntry]
    min_ram_mb: int
    min_vram_mb: int | None
    description: str


_PROFILES: dict[HardwareProfile, ProfileDefinition] = {
    HardwareProfile.LIGHT: ProfileDefinition(
        profile=HardwareProfile.LIGHT,
        ring=[
            ModelRingEntry("qwen", "qwen3:4b"),
            ModelRingEntry("gemma", "gemma3:4b"),
            ModelRingEntry("llama", "llama3.1:8b"),
        ],
        min_ram_mb=8 * 1024,
        min_vram_mb=None,
        description="Минимальные модели каждого семейства — для CPU-only и слабых GPU.",
    ),
    HardwareProfile.STANDARD: ProfileDefinition(
        profile=HardwareProfile.STANDARD,
        ring=[
            ModelRingEntry("qwen", "qwen3:14b"),
            ModelRingEntry("gemma", "gemma3:12b"),
            ModelRingEntry("llama", "llama3.1:8b"),
        ],
        min_ram_mb=24 * 1024,
        min_vram_mb=12 * 1024,
        description="Модели среднего размера — для машин с GPU от 12 ГБ VRAM.",
    ),
    HardwareProfile.PERFORMANCE: ProfileDefinition(
        profile=HardwareProfile.PERFORMANCE,
        ring=[
            ModelRingEntry("qwen", "qwen3:32b"),
            ModelRingEntry("gemma", "gemma3:27b"),
            ModelRingEntry("llama", "llama3.1:70b"),
        ],
        min_ram_mb=64 * 1024,
        min_vram_mb=24 * 1024,
        description="Старшие модели каждого семейства — для мощных GPU/серверов.",
    ),
}


def get_profile_definition(profile: HardwareProfile) -> ProfileDefinition:
    return _PROFILES[profile]


class ProfileRecommender:
    """Только рекомендует профиль — пользователь всегда может переопределить."""

    def recommend(self, hardware: HardwareInfo) -> HardwareProfile:
        if self._meets(HardwareProfile.PERFORMANCE, hardware):
            return HardwareProfile.PERFORMANCE
        if self._meets(HardwareProfile.STANDARD, hardware):
            return HardwareProfile.STANDARD
        return HardwareProfile.LIGHT

    def _meets(self, profile: HardwareProfile, hardware: HardwareInfo) -> bool:
        definition = _PROFILES[profile]
        if hardware.total_ram_mb < definition.min_ram_mb:
            return False
        if definition.min_vram_mb is None:
            return True
        if hardware.gpu.vendor == GpuVendor.NONE:
            return False
        if hardware.gpu.vendor == GpuVendor.APPLE:
            # Unified memory: VRAM отдельно не выделена, ориентируемся на RAM.
            return hardware.total_ram_mb >= definition.min_vram_mb
        return (hardware.gpu.vram_total_mb or 0) >= definition.min_vram_mb
