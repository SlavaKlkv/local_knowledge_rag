from pathlib import Path
from tempfile import mkdtemp

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.routers import inference
from app.hardware.models import GpuInfo, GpuVendor, HardwareInfo
from app.hardware.profiles import HardwareProfile
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.llm.provisioning import DownloadState, ModelProvisioner, ModelRemoval
from app.llm.ring_config import RingConfigStore
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
    store: RingConfigStore | None = None,
) -> TestClient:
    base = FakeProvider(installed=installed)
    app = create_app()
    # Состав кольца хранится в файле: тестам нужен свой, иначе они читали бы
    # и переписывали настройку рабочей машины.
    app.dependency_overrides[dependencies.get_ring_config_store] = (
        lambda: store or RingConfigStore(Path(mkdtemp()) / "ring.json")
    )
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


def test_status_reports_local_execution_for_a_local_runtime():
    body = _client().get("/inference/status").json()

    assert body["provider_is_local"] is True


def test_status_reports_remote_execution_for_a_remote_runtime():
    """Отметка «локально» в UI обязана погаснуть, а не остаться враньём."""

    class RemoteProvider(FakeProvider):
        name = "remote"
        is_local = False

    remote = RemoteProvider()
    app = create_app()
    app.dependency_overrides[dependencies.get_ring_config_store] = (
        lambda: RingConfigStore(Path(mkdtemp()) / "ring.json")
    )
    app.dependency_overrides[dependencies.get_active_profile] = (
        lambda: HardwareProfile.LIGHT
    )
    app.dependency_overrides[dependencies.get_base_llm_provider] = lambda: remote
    app.dependency_overrides[dependencies.get_llm_provider] = lambda: remote
    app.dependency_overrides[dependencies.get_model_provisioner] = (
        lambda: ModelProvisioner(remote)
    )
    app.dependency_overrides[inference.get_hardware_detector] = (
        lambda: FakeHardwareDetector()
    )

    body = TestClient(app).get("/inference/status").json()

    assert body["provider_is_local"] is False


def test_download_is_accepted_and_reported_as_pending(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(provisioner=provisioner)

    response = client.post("/inference/models/qwen3:4b/download")

    assert response.status_code == 202
    assert response.json()["model"] == "qwen3:4b"


def test_cancel_marks_a_pending_download_cancelled(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(provisioner=provisioner)
    client.post("/inference/models/qwen3:4b/download")

    response = client.post("/inference/models/qwen3:4b/download/cancel")

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"


def test_cancel_of_a_never_started_download_is_404():
    response = _client().post("/inference/models/qwen3:4b/download/cancel")

    assert response.status_code == 404


def test_delete_removes_the_weights(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(
        provisioner, "delete_model", lambda model: ModelRemoval(model, True)
    )

    response = _client(provisioner=provisioner).delete("/inference/models/qwen3:4b")

    assert response.status_code == 200
    assert response.json()["deleted"] is True


def test_delete_of_absent_weights_reports_that_nothing_remains(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(
        provisioner, "delete_model", lambda model: ModelRemoval(model, False)
    )

    response = _client(provisioner=provisioner).delete("/inference/models/gemma3:4b")

    body = response.json()
    assert response.status_code == 200
    assert body["deleted"] is False
    assert body["partials_deleted"] == 0
    assert body["detail"] == "Весов модели уже нет."


def test_delete_during_pull_reports_immediate_partial_cleanup(monkeypatch):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(
        provisioner,
        "delete_model",
        lambda model: ModelRemoval(model, False, partials_deleted=17),
    )

    response = _client(provisioner=provisioner).delete("/inference/models/gemma3:4b")

    body = response.json()
    assert response.status_code == 200
    assert body["deleted"] is False
    assert body["partials_deleted"] == 17
    assert "остановлена" in body["detail"]


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


def test_ring_config_reports_the_profile_composition_by_default():
    body = _client(installed=["qwen3:4b"]).get("/inference/ring").json()

    assert body["source"] == "profile"
    # В составе — только то, что реально может ответить; остальное профиля
    # ждёт загрузки и предлагается в кандидатах.
    assert [m["model"] for m in body["models"]] == ["qwen3:4b"]
    assert body["models"][0]["installed"] is True
    assert {"gemma3:4b", "llama3.1:8b"} <= {m["model"] for m in body["available"]}


def test_ring_config_does_not_offer_uninstalled_models_of_other_profiles():
    """Веса чужого профиля рассчитаны на другое железо — их не предлагают."""
    body = _client().get("/inference/ring").json()
    available = {m["model"] for m in body["available"]}

    assert "qwen3:32b" not in available
    assert "llama3.1:70b" not in available
    assert available.isdisjoint({m["model"] for m in body["models"]})


def test_uninstalled_model_of_another_profile_cannot_enter_the_ring():
    response = _client().put("/inference/ring", json={"models": ["qwen3:32b"]})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_model_of_a_heavier_profile_stays_out_even_when_installed():
    """Установленность не делает старшую модель посильной для этого железа."""
    client = _client(installed=["qwen3:4b", "qwen3:32b"])

    available = {m["model"] for m in client.get("/inference/ring").json()["available"]}
    response = client.put("/inference/ring", json={"models": ["qwen3:32b"]})

    assert "qwen3:32b" not in available
    assert response.status_code == 422


def test_ring_config_offers_models_installed_in_the_runtime():
    body = _client(installed=["mistral:7b"]).get("/inference/ring").json()

    offered = {m["model"]: m for m in body["available"]}
    assert offered["mistral:7b"]["installed"] is True
    assert offered["mistral:7b"]["in_profile"] is False


def test_status_marks_models_added_outside_the_profile(tmp_path):
    """Интерфейсу нужно отличать модель профиля от добавленной вручную."""
    store = RingConfigStore(tmp_path / "ring.json")
    client = _client(installed=["qwen3:4b", "mistral:7b"], store=store)
    client.put("/inference/ring", json={"models": ["qwen3:4b", "mistral:7b"]})

    models = {m["model"]: m for m in client.get("/inference/status").json()["models"]}

    assert models["qwen3:4b"]["in_profile"] is True
    assert models["mistral:7b"]["in_profile"] is False


def test_ring_composition_is_saved_with_its_order(tmp_path):
    store = RingConfigStore(tmp_path / "ring.json")
    client = _client(installed=["llama3.1:8b", "qwen3:4b"], store=store)

    response = client.put(
        "/inference/ring", json={"models": ["llama3.1:8b", "qwen3:4b"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "custom"
    assert [m["model"] for m in body["models"]] == ["llama3.1:8b", "qwen3:4b"]
    assert store.load() == ["llama3.1:8b", "qwen3:4b"]


def test_saved_composition_is_reflected_in_status(tmp_path):
    store = RingConfigStore(tmp_path / "ring.json")
    client = _client(installed=["llama3.1:8b"], store=store)
    client.put("/inference/ring", json={"models": ["llama3.1:8b"]})

    body = client.get("/inference/status").json()

    assert [m["model"] for m in body["models"]] == ["llama3.1:8b"]


def test_ring_shows_the_whole_profile_when_nothing_is_installed():
    """На чистой машине состав читается как план: скрывать нечего и незачем."""
    body = _client().get("/inference/ring").json()

    assert [m["installed"] for m in body["models"]] == [False, False, False]


def test_unknown_model_cannot_enter_the_ring():
    response = _client().put("/inference/ring", json={"models": ["invented:1b"]})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_empty_ring_is_rejected():
    response = _client().put("/inference/ring", json={"models": []})

    assert response.status_code == 422


def test_duplicate_model_in_the_ring_is_rejected():
    response = _client().put(
        "/inference/ring", json={"models": ["qwen3:4b", "qwen3:4b"]}
    )

    assert response.status_code == 422


def test_reset_returns_the_ring_to_the_profile(tmp_path):
    store = RingConfigStore(tmp_path / "ring.json")
    client = _client(store=store)
    client.put("/inference/ring", json={"models": ["llama3.1:8b"]})

    body = client.delete("/inference/ring").json()

    assert body["source"] == "profile"
    assert len(body["models"]) == 3
    assert store.load() is None


def test_model_added_to_the_ring_becomes_downloadable(monkeypatch, tmp_path):
    provisioner = ModelProvisioner(FakeProvider())
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(
        installed=["mistral:7b"],
        provisioner=provisioner,
        store=RingConfigStore(tmp_path / "r.json"),
    )
    client.put("/inference/ring", json={"models": ["mistral:7b"]})

    response = client.post("/inference/models/mistral:7b/download")

    assert response.status_code == 202


def test_profile_model_outside_the_ring_stays_downloadable(monkeypatch, tmp_path):
    """Иначе неустановленную модель нельзя было бы поставить вообще:
    в кольцо она не принимается, а гейт загрузки шёл по составу."""
    provisioner = ModelProvisioner(FakeProvider(installed=["qwen3:4b"]))
    # Без подмены фоновая задача пошла бы в живой runtime и правда скачала
    # бы веса: TestClient выполняет background tasks синхронно.
    monkeypatch.setattr(provisioner, "run_download", lambda model: None)
    client = _client(
        installed=["qwen3:4b"],
        provisioner=provisioner,
        store=RingConfigStore(tmp_path / "r.json"),
    )
    client.put("/inference/ring", json={"models": ["qwen3:4b"]})

    response = client.post("/inference/models/llama3.1:8b/download")

    assert response.status_code == 202


def test_model_outside_the_profile_is_not_downloadable(tmp_path):
    client = _client(store=RingConfigStore(tmp_path / "r.json"))

    response = client.post("/inference/models/qwen3:32b/download")

    assert response.status_code == 422


def test_uninstalled_model_cannot_be_saved_in_the_ring(tmp_path):
    store = RingConfigStore(tmp_path / "r.json")
    client = _client(installed=["qwen3:4b"], store=store)

    response = client.put(
        "/inference/ring", json={"models": ["qwen3:4b", "llama3.1:8b"]}
    )

    assert response.status_code == 422
    assert "llama3.1:8b" in response.json()["error"]["message"]
    assert store.load() is None


def test_model_removed_from_the_runtime_disappears_from_the_ring(tmp_path):
    """Состав, показанный интерфейсу, не содержит удалённых с диска весов."""
    store = RingConfigStore(tmp_path / "r.json")
    saved = _client(installed=["qwen3:4b", "gemma3:4b"], store=store)
    saved.put("/inference/ring", json={"models": ["qwen3:4b", "gemma3:4b"]})

    # Та же настройка, но gemma3:4b из runtime'а исчезла.
    body = _client(installed=["qwen3:4b"], store=store).get("/inference/ring").json()

    assert [m["model"] for m in body["models"]] == ["qwen3:4b"]
    assert store.load() == ["qwen3:4b", "gemma3:4b"]


def test_profile_ring_shows_only_installed_models(tmp_path):
    """Строки «не установлена» в кольце быть не должно — она там не работает."""
    body = _client(installed=["qwen3:4b"]).get("/inference/ring").json()

    assert body["source"] == "profile"
    assert [m["model"] for m in body["models"]] == ["qwen3:4b"]
    assert "gemma3:4b" in {m["model"] for m in body["available"]}


def test_download_plan_keeps_a_model_the_ring_dropped(tmp_path):
    """Выпавшую из кольца модель по-прежнему предлагается скачать —
    иначе вернуть её в состав было бы нечем."""
    store = RingConfigStore(tmp_path / "r.json")
    _client(installed=["qwen3:4b", "gemma3:4b"], store=store).put(
        "/inference/ring", json={"models": ["qwen3:4b", "gemma3:4b"]}
    )
    client = _client(installed=["qwen3:4b"], store=store)

    plan = client.get("/inference/status").json()["models"]
    ring = client.get("/inference/ring").json()["models"]

    assert [m["model"] for m in plan] == ["qwen3:4b", "gemma3:4b"]
    assert [m["model"] for m in ring] == ["qwen3:4b"]


def test_embedding_model_is_not_offered_for_the_ring():
    body = _client(installed=["nomic-embed-text:latest"]).get("/inference/ring").json()

    assert "nomic-embed-text:latest" not in {m["model"] for m in body["available"]}


def test_embedding_model_cannot_enter_the_ring(tmp_path):
    client = _client(
        installed=["nomic-embed-text:latest"],
        store=RingConfigStore(tmp_path / "ring.json"),
    )

    response = client.put(
        "/inference/ring", json={"models": ["qwen3:4b", "nomic-embed-text:latest"]}
    )

    assert response.status_code == 422
    assert "Embedding" in response.json()["error"]["message"]


def test_ring_is_rebuilt_when_a_model_disappears_from_the_runtime():
    """Кольцо кэшируется на процесс — удаление модели должно его пересобрать."""
    from app.api.dependencies import refreshed_llm_provider, reset_llm_provider
    from app.hardware.profiles import ModelRingEntry
    from app.llm.ring import ModelRing
    from app.llm.ring_provider import RingLLMProvider

    base = FakeProvider(installed=["qwen3:4b"])
    stale = RingLLMProvider(
        ModelRing(
            base,
            [
                ModelRingEntry(family="qwen", model="qwen3:4b"),
                ModelRingEntry(family="gemma", model="gemma3:4b"),
            ],
        )
    )

    try:
        fresh = refreshed_llm_provider(stale, installed={"qwen3:4b"})
        unchanged = refreshed_llm_provider(stale, installed={"qwen3:4b", "gemma3:4b"})
    finally:
        reset_llm_provider()

    # Пересобранный провайдер — уже не тот объект, а состав без удалённой.
    assert fresh is not stale
    assert unchanged is stale


def test_provider_without_a_ring_is_left_alone():
    from app.api.dependencies import refreshed_llm_provider

    base = FakeProvider(installed=["qwen3:4b"])

    assert refreshed_llm_provider(base, installed=set()) is base
