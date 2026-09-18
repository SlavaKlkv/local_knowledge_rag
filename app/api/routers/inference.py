"""Эндпоинты локального inference: профиль, кольцо моделей, их загрузка.

Загрузка весов запускается только явным POST от пользователя — приложение
не устанавливает многогигабайтные модели по собственной инициативе.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_active_profile,
    get_base_llm_provider,
    get_llm_provider,
    get_model_provisioner,
    get_ring_config_store,
    refreshed_llm_provider,
    reset_llm_provider,
)
from app.core.config import get_settings
from app.core.errors import InferenceError, ValidationError
from app.hardware.detector import HardwareDetector
from app.hardware.profiles import (
    HardwareProfile,
    ModelRingEntry,
    get_profile_definition,
    heavier_profile_models,
)
from app.hardware.runtime_detector import (
    InferenceRuntime,
    RuntimeDetectionResult,
    RuntimeDetector,
)
from app.hardware.runtime_installer import RuntimeInstaller
from app.llm.base import LocalLLMProvider
from app.llm.provisioning import DownloadState, ModelProvisioner
from app.llm.ring import ModelHealth
from app.llm.ring_config import RingConfigStore, is_embedding_model
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
    # Модель профиля и добавленная пользователем различаются в интерфейсе:
    # у второй неизвестна цена загрузки и не проверено соответствие железу.
    in_profile: bool


class RingSlotResponse(BaseModel):
    """Модель в списке состава или в списке кандидатов на добавление."""

    family: str
    model: str
    installed: bool
    download_size_gb: float
    min_ram_gb: int
    in_profile: bool


class RingConfigResponse(BaseModel):
    profile: HardwareProfile
    # "profile" — состав по железу, "custom" — выбранный пользователем.
    source: str
    models: list[RingSlotResponse]
    available: list[RingSlotResponse]


class RingConfigUpdate(BaseModel):
    # Порядок в списке и есть порядок обхода кольца.
    models: list[str] = Field(min_length=1)


class InferenceStatusResponse(BaseModel):
    provider: str
    provider_healthy: bool
    # Исполняется ли inference на машине пользователя. Признак приходит от
    # самого провайдера, чтобы отметка в UI не разъехалась с реальностью,
    # когда появится runtime к удалённому API.
    provider_is_local: bool
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


class ModelRemovalResponse(BaseModel):
    model: str
    # False — зарегистрированных весов не было: модель уже удалена либо
    # загрузка прервалась до регистрации в runtime.
    deleted: bool
    partials_deleted: int = 0
    detail: str


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
    store: RingConfigStore = Depends(get_ring_config_store),
) -> InferenceStatusResponse:
    settings = get_settings()
    # Кольцо могло собраться, когда модель ещё лежала на диске: сверяем его
    # с runtime'ом до того, как отдать состав интерфейсу.
    installed = _installed_models(base_provider)
    provider = refreshed_llm_provider(provider, installed or None)
    plan = provisioner.build_plan(
        profile,
        free_disk_gb=_free_disk_gb(detector),
        # План загрузки — про то, чего не хватает, поэтому он строится по
        # полному составу: отфильтруй здесь неустановленные, и скачать их
        # стало бы неоткуда.
        entries=store.resolve(get_profile_definition(profile).ring),
    )

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
        provider_is_local=base_provider.is_local,
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
                in_profile=m.model in _profile_catalog(profile),
            )
            for m in plan.models
        ],
        ready=plan.ready,
        required_disk_gb=plan.required_disk_gb,
        free_disk_gb=plan.free_disk_gb,
        enough_disk_space=plan.enough_disk_space,
    )


@router.get("/ring", response_model=RingConfigResponse)
def ring_config(
    profile: HardwareProfile = Depends(get_active_profile),
    store: RingConfigStore = Depends(get_ring_config_store),
    base_provider: LocalLLMProvider = Depends(get_base_llm_provider),
) -> RingConfigResponse:
    """Текущий состав кольца и модели, которые можно в него добавить."""
    return _ring_config_response(profile, store, base_provider)


@router.put("/ring", response_model=RingConfigResponse)
def update_ring_config(
    payload: RingConfigUpdate,
    profile: HardwareProfile = Depends(get_active_profile),
    store: RingConfigStore = Depends(get_ring_config_store),
    base_provider: LocalLLMProvider = Depends(get_base_llm_provider),
) -> RingConfigResponse:
    """Задаёт состав и порядок обхода кольца.

    Модель, которой нет ни в каталоге профилей, ни среди установленных в
    runtime, отклоняется: кольцо из несуществующих весов молча деградировало
    бы до отказа отвечать.
    """
    embeddings = [model for model in payload.models if is_embedding_model(model)]
    if embeddings:
        raise ValidationError(
            f"Embedding-модели не отвечают на вопросы: {', '.join(embeddings)}. "
            "В кольцо входят только модели генерации."
        )

    installed = _installed_models(base_provider)
    missing = [model for model in payload.models if model not in installed]
    if missing:
        raise ValidationError(
            f"Не установлены: {', '.join(missing)}. Кольцо принимает только "
            "установленные модели — скачайте их и добавьте в состав."
        )

    # Неустановленная модель в кольце — это гарантированный отказ на каждом
    # обходе: провайдер сходит к runtime и получит ошибку. Состав по профилю
    # такую модель содержать может (её ещё предстоит скачать), но сохранять
    # её выбором пользователя нельзя.
    installed = _installed_models(base_provider)
    absent = [model for model in payload.models if model not in installed]
    if absent:
        raise ValidationError(
            f"Не установлены: {', '.join(absent)}. Кольцо принимает только "
            "установленные модели — скачайте их или уберите из состава."
        )

    known = _selectable_models(profile, base_provider)
    unknown = [model for model in payload.models if model not in known]
    if unknown:
        raise ValidationError(
            f"Недоступные модели: {', '.join(unknown)}. "
            f"В кольцо профиля {profile} входят модели самого профиля и "
            "установленные в runtime."
        )

    store.save(payload.models)
    # Кольцо живёт в кэшированном провайдере — без сброса новый состав
    # начал бы работать только после перезапуска процесса.
    reset_llm_provider()
    return _ring_config_response(profile, store, base_provider)


@router.delete("/ring", response_model=RingConfigResponse)
def reset_ring_config(
    profile: HardwareProfile = Depends(get_active_profile),
    store: RingConfigStore = Depends(get_ring_config_store),
    base_provider: LocalLLMProvider = Depends(get_base_llm_provider),
) -> RingConfigResponse:
    """Возвращает кольцо к составу, рекомендованному профилем железа."""
    store.clear()
    reset_llm_provider()
    return _ring_config_response(profile, store, base_provider)


def _installed_models(provider: LocalLLMProvider) -> set[str]:
    # Недоступный runtime не должен ломать настройку: список установленного
    # тогда просто пуст, а каталог профилей известен и без него.
    try:
        return {info.name for info in provider.list_models()}
    except InferenceError:
        return set()


def _generative_models(provider: LocalLLMProvider) -> set[str]:
    return {name for name in _installed_models(provider) if not is_embedding_model(name)}


def _profile_catalog(profile: HardwareProfile) -> dict[str, ModelRingEntry]:
    """Модели, которые можно предложить скачать для этого железа.

    Кольца соседних профилей сюда не входят: их веса рассчитаны на другую
    машину, и предложить скачать llama3.1:70b на лёгком профиле — это сорок
    гигабайт ради модели, которая потом не запустится.
    """
    return {entry.model: entry for entry in get_profile_definition(profile).ring}


def _selectable_catalog(
    profile: HardwareProfile, provider: LocalLLMProvider
) -> dict[str, ModelRingEntry]:
    """Кольцо профиля плюс генеративные модели, уже стоящие в runtime.

    Установленной модели не нужен каталог: качать нечего, а отвечать через
    неё кольцо умеет — имя модели уходит в runtime как есть. Исключение —
    кольца профилей тяжелее текущего: такая модель железу не по силам,
    и установленность этого не меняет.
    """
    catalog = _profile_catalog(profile)
    too_heavy = heavier_profile_models(profile)
    for name in _generative_models(provider):
        if name in too_heavy:
            continue
        catalog.setdefault(name, ModelRingEntry(family=name.split(":", 1)[0], model=name))
    return catalog


def _selectable_models(
    profile: HardwareProfile, provider: LocalLLMProvider
) -> set[str]:
    return set(_selectable_catalog(profile, provider))


def _to_slot(
    entry: ModelRingEntry, installed: set[str], profile_models: set[str]
) -> RingSlotResponse:
    return RingSlotResponse(
        family=entry.family,
        model=entry.model,
        installed=entry.model in installed,
        download_size_gb=entry.download_size_gb,
        min_ram_gb=entry.min_ram_gb,
        in_profile=entry.model in profile_models,
    )


def _ring_config_response(
    profile: HardwareProfile,
    store: RingConfigStore,
    base_provider: LocalLLMProvider,
) -> RingConfigResponse:
    profile_ring = get_profile_definition(profile).ring
    profile_models = {entry.model for entry in profile_ring}
    installed = _installed_models(base_provider)

    current = store.resolve(profile_ring, installed=installed or None)
    current_models = {entry.model for entry in current}

    catalog = _selectable_catalog(profile, base_provider)

    return RingConfigResponse(
        profile=profile,
        source="profile" if store.load() is None else "custom",
        models=[_to_slot(entry, installed, profile_models) for entry in current],
        available=[
            _to_slot(entry, installed, profile_models)
            for entry in sorted(catalog.values(), key=lambda e: e.model)
            if entry.model not in current_models
        ],
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
    store: RingConfigStore = Depends(get_ring_config_store),
    base_provider: LocalLLMProvider = Depends(get_base_llm_provider),
) -> DownloadProgressResponse:
    """Запускает загрузку весов. Только по явному запросу пользователя."""
    # Скачивать можно всё, что предлагает профиль, а не только текущий
    # состав кольца: неустановленная модель в кольцо не принимается, и
    # гейт по составу не давал бы установить её вообще.
    ring = store.resolve(
        get_profile_definition(profile).ring,
        installed=_installed_models(base_provider) or None,
    )
    allowed = {entry.model for entry in ring} | set(_profile_catalog(profile))
    progress = provisioner.start_download(model, profile, allowed=allowed)
    if progress.state == DownloadState.PENDING:
        background_tasks.add_task(provisioner.run_download, model)
    return _to_response(progress)


@router.post(
    "/models/{model:path}/download/cancel",
    response_model=DownloadProgressResponse,
)
def cancel_download(
    model: str,
    provisioner: ModelProvisioner = Depends(get_model_provisioner),
) -> DownloadProgressResponse:
    """Останавливает начатую загрузку весов по запросу пользователя."""
    return _to_response(provisioner.cancel_download(model))


@router.delete("/models/{model:path}", response_model=ModelRemovalResponse)
def delete_model(
    model: str,
    provisioner: ModelProvisioner = Depends(get_model_provisioner),
) -> ModelRemovalResponse:
    """Удаляет веса модели. Только по явному запросу пользователя."""
    removal = provisioner.delete_model(model)
    reset_llm_provider()
    if removal.deleted:
        return ModelRemovalResponse(
            model=model, deleted=True, detail="Веса удалены"
        )
    if removal.partials_deleted:
        return ModelRemovalResponse(
            model=model,
            deleted=False,
            partials_deleted=removal.partials_deleted,
            detail=(
                "Загрузка остановлена, скачанные части удалены "
                f"({removal.partials_deleted} файлов)."
            ),
        )
    return ModelRemovalResponse(
        model=model,
        deleted=False,
        detail="Весов модели уже нет.",
    )


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
