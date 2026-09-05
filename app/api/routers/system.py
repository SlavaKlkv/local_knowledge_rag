"""Служебные эндпоинты: состояние приложения, окружения и оборудования."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.hardware.detector import HardwareDetector
from app.hardware.models import GpuVendor, HardwareInfo
from app.hardware.profiles import HardwareProfile, ProfileRecommender
from app.hardware.runtime_detector import (
    InferenceRuntime,
    RuntimeDetectionResult,
    RuntimeDetector,
)

router = APIRouter(prefix="/system", tags=["system"])


class HealthResponse(BaseModel):
    status: str
    app_env: str


class InfoResponse(BaseModel):
    app_env: str
    llm_model: str
    embedding_model: str
    embedding_dim: int
    qdrant_collection: str


class GpuResponse(BaseModel):
    vendor: GpuVendor
    name: str | None
    vram_total_mb: int | None
    cuda_available: bool
    metal_available: bool


class RuntimeAvailabilityResponse(BaseModel):
    runtime: InferenceRuntime
    available: bool
    base_url: str
    detail: str | None


class HardwareResponse(BaseModel):
    cpu_count: int
    architecture: str
    total_ram_mb: int
    available_ram_mb: int
    free_disk_mb: int
    gpu: GpuResponse
    recommended_profile: HardwareProfile
    detected_runtimes: list[RuntimeAvailabilityResponse]
    recommended_runtime: InferenceRuntime | None


def get_hardware_detector() -> HardwareDetector:
    return HardwareDetector()


def get_profile_recommender() -> ProfileRecommender:
    return ProfileRecommender()


def get_runtime_detector() -> RuntimeDetector:
    settings = get_settings()
    return RuntimeDetector(ollama_url=settings.ollama_url, vllm_url=settings.vllm_url)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings: Settings = get_settings()
    return HealthResponse(status="ok", app_env=settings.app_env)


@router.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    settings = get_settings()
    return InfoResponse(
        app_env=settings.app_env,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        qdrant_collection=settings.qdrant_collection,
    )


@router.get("/hardware", response_model=HardwareResponse)
async def hardware(
    detector: HardwareDetector = Depends(get_hardware_detector),
    recommender: ProfileRecommender = Depends(get_profile_recommender),
    runtime_detector: RuntimeDetector = Depends(get_runtime_detector),
) -> HardwareResponse:
    """Оборудование, рекомендуемый профиль и доступные runtime'ы.

    Приложение только рекомендует — выбор профиля и runtime всегда
    остаётся за пользователем (см. .env: HARDWARE_PROFILE_OVERRIDE).
    """
    info: HardwareInfo = detector.detect()
    runtimes: RuntimeDetectionResult = runtime_detector.detect()
    return HardwareResponse(
        cpu_count=info.cpu_count,
        architecture=info.architecture,
        total_ram_mb=info.total_ram_mb,
        available_ram_mb=info.available_ram_mb,
        free_disk_mb=info.free_disk_mb,
        gpu=GpuResponse(
            vendor=info.gpu.vendor,
            name=info.gpu.name,
            vram_total_mb=info.gpu.vram_total_mb,
            cuda_available=info.gpu.cuda_available,
            metal_available=info.gpu.metal_available,
        ),
        recommended_profile=recommender.recommend(info),
        detected_runtimes=[
            RuntimeAvailabilityResponse(
                runtime=r.runtime, available=r.available, base_url=r.base_url, detail=r.detail
            )
            for r in runtimes.runtimes
        ],
        recommended_runtime=runtimes.recommended,
    )
