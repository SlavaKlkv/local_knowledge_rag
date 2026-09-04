import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.routers import inference
from app.hardware.models import GpuInfo, GpuVendor, HardwareInfo
from app.hardware.profiles import HardwareProfile
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.llm.provisioning import DownloadState, ModelProvisioner
from app.main import create_app


class FakeProvider(LocalLLMProvider):
    name = "fake"

    def __init__(self, installed: list[str] | None = None, healthy: bool = True) -> None:
        self.installed = installed or []
        self.healthy = healthy

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        return GenerationResult(text="ответ", model=model, provider=self.name, latency_ms=1)

    def health_check(self) -> bool:
        return self.healthy

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=name, provider=self.name) for name in self.installed]


class FakeHardwareDetector:
    def __init__(self, free_disk_mb: int = 200 * 1024) -> None:
        self.free_disk_mb = free_disk_mb

    def detect(self) -> HardwareInfo:
        return HardwareInfo(
            cpu_count=8,
            architecture="arm64",
            total_ram_mb=32 * 1024,
            available_ram_mb=16 * 1024,
            free_disk_mb=self.free_disk_mb,
            gpu=GpuInfo(vendor=GpuVendor.APPLE, metal_available=True),
        )


def _client(
    installed: list[str] | None = None,
    free_disk_mb: int = 200 * 1024,
    provisioner: ModelProvisioner | None = None,
) -> TestClient:
    base = FakeProvider(installed=installed)
    app = create_app()
    app.dependency_overrides[dependencies.get_active_profile] = (
        lambda: HardwareProfile.LIGHT
    )
    app.dependency_overrides[dependencies.get_base_llm_provider] = lambda: base
    app.dependency_overrides[dependencies.get_llm_provider] = lambda: base
    app.dependency_overrides[dependencies.get_model_provisioner] = (
        lambda: provisioner or ModelProvisioner(base)
    )
    app.dependency_overrides[inference.get_hardware_detector] = (
        lambda: FakeHardwareDetector(free_disk_mb)
    )
    return TestClient(app)


def test_status_lists_ring_models_with_install_state_and_cost():
    response = _client(installed=["qwen3:4b"]).get("/inference/status")

    assert response.status_code == 200
    body = response.json()
    models = {m["model"]: m for m in body["models"]}
    assert models["qwen3:4b"]["installed"] is True
    assert models["gemma3:4b"]["installed"] is False
    assert models["gemma3:4b"]["download_size_gb"] > 0
    assert models["gemma3:4b"]["min_ram_gb"] > 0


def test_status_reports_ready_when_at_least_one_model_is_installed():
    body = _client(installed=["llama3.1:8b"]).get("/inference/status").json()

    assert body["ready"] is True
    assert body["profile"] == "light"


def test_status_reports_not_ready_on_a_machine_without_models():
    body = _client(installed=[]).get("/inference/status").json()

    assert body["ready"] is False
    assert body["required_disk_gb"] > 0


def test_status_flags_insufficient_disk_space():
    body = _client(installed=[], free_disk_mb=1024).get("/inference/status").json()

    assert body["enough_disk_space"] is False


def test_status_exposes_provider_health():
    body = _client().get("/inference/status").json()

    assert body["provider_healthy"] is True
    assert body["provider"] in {"ollama", "vllm"}


def test_download_is_accepted_and_reported_as_pending(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(provisioner=provisioner)

    response = client.post("/inference/models/qwen3:4b/download")

    assert response.status_code == 202
    assert response.json()["model"] == "qwen3:4b"


def test_download_of_a_model_outside_the_ring_is_rejected():
    response = _client().post("/inference/models/mistral:7b/download")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_progress_of_a_never_started_download_is_404():
    response = _client().get("/inference/downloads/qwen3:4b")

    assert response.status_code == 404


def test_downloads_list_reports_registered_downloads(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(provisioner=provisioner)
    client.post("/inference/models/qwen3:4b/download")

    body = client.get("/inference/downloads").json()

    assert [d["model"] for d in body] == ["qwen3:4b"]


def test_progress_reflects_a_completed_download(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(provisioner=provisioner)
    client.post("/inference/models/qwen3:4b/download")

    progress = provisioner.progress("qwen3:4b")
    progress.state = DownloadState.COMPLETED
    progress.total_bytes = 100
    progress.completed_bytes = 100

    body = client.get("/inference/downloads/qwen3:4b").json()

    assert body["state"] == "completed"
    assert body["percent"] == 100.0


@pytest.mark.parametrize("installed", [[], ["qwen3:4b"]])
def test_status_never_triggers_a_download_on_its_own(monkeypatch, installed):
    """Проверка статуса не должна ничего скачивать: загрузка — только по POST."""
    provisioner = ModelProvisioner(FakeProvider(installed=installed))

    def fail(*args, **kwargs):  # pragma: no cover - не должен вызываться
        raise AssertionError("статус не имеет права запускать загрузку")

    monkeypatch.setattr(provisioner, "run_download", fail)
    monkeypatch.setattr(provisioner, "start_download", fail)

    response = _client(installed=installed, provisioner=provisioner).get(
        "/inference/status"
    )

    assert response.status_code == 200


class FakeRuntimeDetector:
    def __init__(self, ollama_available: bool = True, vllm_available: bool = False) -> None:
        self.ollama_available = ollama_available
        self.vllm_available = vllm_available

    def detect(self):
        from app.hardware.runtime_detector import (
            InferenceRuntime,
            RuntimeAvailability,
            RuntimeDetectionResult,
        )

        return RuntimeDetectionResult(
            runtimes=[
                RuntimeAvailability(
                    runtime=InferenceRuntime.OLLAMA,
                    available=self.ollama_available,
                    base_url="http://localhost:11434",
                    detail=None if self.ollama_available else "connection refused",
                ),
                RuntimeAvailability(
                    runtime=InferenceRuntime.VLLM,
                    available=self.vllm_available,
                    base_url="http://localhost:8000",
                    detail=None if self.vllm_available else "connection refused",
                ),
            ]
        )


def _runtime_client(detector: FakeRuntimeDetector, installer=None) -> TestClient:
    from app.hardware.runtime_installer import RuntimeInstaller

    app = create_app()
    app.dependency_overrides[inference.get_runtime_detector] = lambda: detector
    app.dependency_overrides[inference.get_runtime_installer] = lambda: (
        installer
        or RuntimeInstaller(
            run_command=lambda args: (0, "ok"),
            system="Darwin",
            has_binary=lambda name: name == "brew",
        )
    )
    return TestClient(app)


def test_runtimes_endpoint_offers_installation_only_for_missing_runtimes():
    client = _runtime_client(FakeRuntimeDetector(ollama_available=True))

    body = client.get("/inference/runtimes").json()

    by_runtime = {r["runtime"]: r for r in body["runtimes"]}
    assert by_runtime["ollama"]["installation_offer"] is None
    assert by_runtime["vllm"]["installation_offer"] is not None
    assert body["recommended"] == "ollama"


def test_runtimes_endpoint_offers_ollama_installation_when_it_is_missing():
    client = _runtime_client(FakeRuntimeDetector(ollama_available=False))

    body = client.get("/inference/runtimes").json()

    offer = next(r for r in body["runtimes"] if r["runtime"] == "ollama")["installation_offer"]
    assert offer["manual_command"] == "brew install ollama"
    assert "ollama.com" in offer["documentation_url"]
    assert body["recommended"] is None


def test_installation_without_confirmation_is_rejected():
    client = _runtime_client(FakeRuntimeDetector(ollama_available=False))

    response = client.post("/inference/runtimes/ollama/install", json={"confirm": False})

    assert response.status_code == 422
    assert "подтверждения" in response.json()["error"]["message"]


def test_installation_is_refused_while_disabled_by_configuration(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("RUNTIME_INSTALL_ENABLED", "false")
    get_settings.cache_clear()
    client = _runtime_client(FakeRuntimeDetector(ollama_available=False))

    response = client.post("/inference/runtimes/ollama/install", json={"confirm": True})

    get_settings.cache_clear()
    assert response.status_code == 422
    assert "недоступна" in response.json()["error"]["message"]


def test_confirmed_installation_reports_result_and_rechecks_availability(monkeypatch):
    from app.core.config import get_settings
    from app.hardware.runtime_installer import RuntimeInstaller

    monkeypatch.setenv("RUNTIME_INSTALL_ENABLED", "true")
    get_settings.cache_clear()
    installer = RuntimeInstaller(
        run_command=lambda args: (0, "installed"),
        system="Darwin",
        has_binary=lambda name: name == "brew",
    )
    client = _runtime_client(FakeRuntimeDetector(ollama_available=True), installer)

    response = client.post("/inference/runtimes/ollama/install", json={"confirm": True})

    get_settings.cache_clear()
    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] is True
    assert body["available_after_install"] is True
