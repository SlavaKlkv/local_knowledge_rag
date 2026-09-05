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
from app.hardware.runtime_detector import (
    InferenceRuntime,
    RuntimeDetectionResult,
    RuntimeDetector,
)
from app.hardware.runtime_installer import RuntimeInstaller
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


class InstallationOfferResponse(BaseModel):
    runtime: InferenceRuntime
    supported: bool
    manual_command: str | None
    documentation_url: str
    note: str
    automatic_install_enabled: bool


class RuntimeStatusResponse(BaseModel):
    runtime: InferenceRuntime
    available: bool
    base_url: str
    detail: str | None
    # Предложение появляется только для отсутствующего runtime'а —
    # предлагать установку уже установленного нечего.
    installation_offer: InstallationOfferResponse | None


class RuntimesResponse(BaseModel):
    selected: str
    recommended: InferenceRuntime | None
    runtimes: list[RuntimeStatusResponse]


class InstallationRequest(BaseModel):
    # Явное подтверждение обязательно: приложение не ставит системные
    # компоненты по собственной инициативе.
    confirm: bool = False


class InstallationResultResponse(BaseModel):
    runtime: InferenceRuntime
    succeeded: bool
    output: str
    available_after_install: bool


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


def get_runtime_detector() -> RuntimeDetector:
    settings = get_settings()
    return RuntimeDetector(ollama_url=settings.ollama_url, vllm_url=settings.vllm_url)


def get_runtime_installer() -> RuntimeInstaller:
    return RuntimeInstaller()


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


@router.get("/runtimes", response_model=RuntimesResponse)
def runtimes(
    detector: RuntimeDetector = Depends(get_runtime_detector),
    installer: RuntimeInstaller = Depends(get_runtime_installer),
) -> RuntimesResponse:
    """Обнаруженные runtime'ы, а для отсутствующих — как их поставить."""
    detection: RuntimeDetectionResult = detector.detect()
    return RuntimesResponse(
        selected=get_settings().inference_provider,
        recommended=detection.recommended,
        runtimes=[
            RuntimeStatusResponse(
                runtime=r.runtime,
                available=r.available,
                base_url=r.base_url,
                detail=r.detail,
                installation_offer=(
                    None if r.available else _to_offer(installer.offer(r.runtime))
                ),
            )
            for r in detection.runtimes
        ],
    )


@router.post(
    "/runtimes/{runtime}/install", response_model=InstallationResultResponse
)
def install_runtime(
    runtime: InferenceRuntime,
    payload: InstallationRequest,
    installer: RuntimeInstaller = Depends(get_runtime_installer),
    detector: RuntimeDetector = Depends(get_runtime_detector),
) -> InstallationResultResponse:
    """Ставит runtime — только при confirm=true и включённой настройке."""
    result = installer.install(runtime, confirmed=payload.confirm)
    # Health check после установки: успешный код возврата ещё не значит,
    # что сервис поднялся и отвечает.
    available = detector.detect().is_available(runtime)
    return InstallationResultResponse(
        runtime=result.runtime,
        succeeded=result.succeeded,
        output=result.output,
        available_after_install=available,
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


def _to_offer(offer) -> InstallationOfferResponse:
    return InstallationOfferResponse(
        runtime=offer.runtime,
        supported=offer.supported,
        manual_command=offer.manual_command,
        documentation_url=offer.documentation_url,
        note=offer.note,
        automatic_install_enabled=offer.automatic_install_enabled,
    )


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
