"""Эндпоинты локального inference: профиль, кольцо моделей, их загрузка.

Загрузка весов запускается только явным POST от пользователя — приложение
не устанавливает многогигабайтные модели по собственной инициативе.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from app.api.dependencies import (
    get_active_profile,
    get_base_llm_provider,
    get_llm_provider,
    get_model_provisioner,
)
from app.core.config import get_settings
from app.hardware.detector import HardwareDetector
from app.hardware.profiles import HardwareProfile, get_profile_definition
from app.llm.base import LocalLLMProvider
from app.llm.provisioning import DownloadState, ModelProvisioner
from app.llm.ring import ModelHealth
from app.llm.ring_provider import RingLLMProvider

router = APIRouter(prefix="/inference", tags=["inference"])


class RingMemberResponse(BaseModel):
    family: str
    model: str
    health: ModelHealth
    consecutive_failures: int


class ModelStatusResponse(BaseModel):
    family: str
    model: str
    installed: bool
    download_size_gb: float
    min_ram_gb: int


class InferenceStatusResponse(BaseModel):
    provider: str
    provider_healthy: bool
    profile: HardwareProfile
    profile_description: str
    ring_enabled: bool
    ring: list[RingMemberResponse]
    models: list[ModelStatusResponse]
    ready: bool
    required_disk_gb: float
    free_disk_gb: float
    enough_disk_space: bool


class DownloadProgressResponse(BaseModel):
    model: str
    state: DownloadState
    percent: float
    completed_bytes: int
    total_bytes: int
    status: str | None = None
    error: str | None = None


def get_hardware_detector() -> HardwareDetector:
    return HardwareDetector()


def _free_disk_gb(detector: HardwareDetector) -> float:
    return detector.detect().free_disk_mb / 1024


@router.get("/status", response_model=InferenceStatusResponse)
def status(
    profile: HardwareProfile = Depends(get_active_profile),
    provider: LocalLLMProvider = Depends(get_llm_provider),
    base_provider: LocalLLMProvider = Depends(get_base_llm_provider),
    provisioner: ModelProvisioner = Depends(get_model_provisioner),
    detector: HardwareDetector = Depends(get_hardware_detector),
) -> InferenceStatusResponse:
    settings = get_settings()
    plan = provisioner.build_plan(profile, free_disk_gb=_free_disk_gb(detector))

    ring: list[RingMemberResponse] = []
    if isinstance(provider, RingLLMProvider):
        ring = [
            RingMemberResponse(
                family=member.family,
                model=member.model,
                health=member.state,
                consecutive_failures=member.consecutive_failures,
            )
            for member in provider.ring.members
        ]

    return InferenceStatusResponse(
        provider=settings.inference_provider,
        provider_healthy=base_provider.health_check(),
        profile=profile,
        profile_description=get_profile_definition(profile).description,
        ring_enabled=settings.model_ring_enabled,
        ring=ring,
        models=[
            ModelStatusResponse(
                family=m.family,
                model=m.model,
                installed=m.installed,
                download_size_gb=m.download_size_gb,
                min_ram_gb=m.min_ram_gb,
            )
            for m in plan.models
        ],
        ready=plan.ready,
        required_disk_gb=plan.required_disk_gb,
        free_disk_gb=plan.free_disk_gb,
        enough_disk_space=plan.enough_disk_space,
    )


@router.post(
    "/models/{model:path}/download",
    response_model=DownloadProgressResponse,
    status_code=202,
)
def download_model(
    model: str,
    background_tasks: BackgroundTasks,
    profile: HardwareProfile = Depends(get_active_profile),
    provisioner: ModelProvisioner = Depends(get_model_provisioner),
) -> DownloadProgressResponse:
    """Запускает загрузку весов. Только по явному запросу пользователя."""
    progress = provisioner.start_download(model, profile)
    if progress.state == DownloadState.PENDING:
        background_tasks.add_task(provisioner.run_download, model)
    return _to_response(progress)


@router.get("/downloads", response_model=list[DownloadProgressResponse])
def list_downloads(
    provisioner: ModelProvisioner = Depends(get_model_provisioner),
) -> list[DownloadProgressResponse]:
    return [_to_response(p) for p in provisioner.all_progress()]


@router.get("/downloads/{model:path}", response_model=DownloadProgressResponse)
def download_progress(
    model: str, provisioner: ModelProvisioner = Depends(get_model_provisioner)
) -> DownloadProgressResponse:
    return _to_response(provisioner.progress(model))


def _to_response(progress) -> DownloadProgressResponse:
    return DownloadProgressResponse(
        model=progress.model,
        state=progress.state,
        percent=progress.percent,
        completed_bytes=progress.completed_bytes,
        total_bytes=progress.total_bytes,
        status=progress.status,
        error=progress.error,
    )
