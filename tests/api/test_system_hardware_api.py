from fastapi.testclient import TestClient

from app.api.routers import system
from app.hardware.models import GpuInfo, GpuVendor, HardwareInfo
from app.hardware.runtime_detector import (
    InferenceRuntime,
    RuntimeAvailability,
    RuntimeDetectionResult,
)
from app.main import create_app


class FakeHardwareDetector:
    def detect(self) -> HardwareInfo:
        return HardwareInfo(
            cpu_count=8,
            architecture="arm64",
            total_ram_mb=32 * 1024,
            available_ram_mb=16 * 1024,
            free_disk_mb=100_000,
            gpu=GpuInfo(vendor=GpuVendor.APPLE, name="Apple Silicon", metal_available=True),
        )


class FakeRuntimeDetector:
    def detect(self) -> RuntimeDetectionResult:
        return RuntimeDetectionResult(
            runtimes=[
                RuntimeAvailability(
                    runtime=InferenceRuntime.OLLAMA,
                    available=True,
                    base_url="http://localhost:11434",
                ),
                RuntimeAvailability(
                    runtime=InferenceRuntime.VLLM,
                    available=False,
                    base_url="http://localhost:8000",
                    detail="connection refused",
                ),
            ]
        )


def test_hardware_endpoint_reports_gpu_profile_and_runtimes():
    app = create_app()
    app.dependency_overrides[system.get_hardware_detector] = lambda: FakeHardwareDetector()
    app.dependency_overrides[system.get_runtime_detector] = lambda: FakeRuntimeDetector()
    client = TestClient(app)

    response = client.get("/system/hardware")

    assert response.status_code == 200
    body = response.json()
    assert body["gpu"]["vendor"] == "apple"
    assert body["recommended_profile"] == "standard"
    assert body["recommended_runtime"] == "ollama"
    assert {r["runtime"]: r["available"] for r in body["detected_runtimes"]} == {
        "ollama": True,
        "vllm": False,
    }


def test_hardware_endpoint_works_with_real_detector():
    """Без переопределений — HardwareDetector реально опрашивает машину;
    важно, что эндпоинт не падает и возвращает согласованные типы."""
    client = TestClient(create_app())

    response = client.get("/system/hardware")

    assert response.status_code == 200
    body = response.json()
    assert body["cpu_count"] >= 1
    assert body["recommended_profile"] in {"light", "standard", "performance"}
