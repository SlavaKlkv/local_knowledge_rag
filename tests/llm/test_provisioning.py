import httpx
import pytest

from app.core.errors import InferenceError, NotFoundError, ValidationError
from app.hardware.profiles import HardwareProfile
from app.llm.base import GenerationRequest, GenerationResult, LocalLLMProvider, ModelInfo
from app.llm.provisioning import DownloadState, ModelProvisioner


class FakeProvider(LocalLLMProvider):
    name = "fake"

    def __init__(self, installed: list[str] | None = None) -> None:
        self.installed = installed or []

    def generate(self, request: GenerationRequest, model: str) -> GenerationResult:
        return GenerationResult(text="ответ", model=model, provider=self.name, latency_ms=1)

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=name, provider=self.name) for name in self.installed]


@pytest.fixture
def provisioner() -> ModelProvisioner:
    return ModelProvisioner(FakeProvider(), base_url="http://localhost:11434")


def test_plan_marks_every_model_missing_on_a_clean_machine(provisioner):
    plan = provisioner.build_plan(HardwareProfile.LIGHT, free_disk_gb=100.0)

    assert [m.model for m in plan.models] == ["qwen3:4b", "gemma3:4b", "llama3.1:8b"]
    assert len(plan.missing) == 3
    assert plan.ready is False


def test_plan_reports_download_size_and_requirements_before_downloading():
    plan = ModelProvisioner(FakeProvider()).build_plan(
        HardwareProfile.LIGHT, free_disk_gb=100.0
    )

    assert plan.required_disk_gb == pytest.approx(2.6 + 3.3 + 4.9, abs=0.05)
    assert all(m.min_ram_gb > 0 for m in plan.models)
    assert plan.enough_disk_space is True


def test_plan_flags_insufficient_disk_space():
    plan = ModelProvisioner(FakeProvider()).build_plan(
        HardwareProfile.PERFORMANCE, free_disk_gb=10.0
    )

    assert plan.enough_disk_space is False


def test_installed_model_is_recognised_by_exact_tag():
    plan = ModelProvisioner(FakeProvider(installed=["qwen3:4b"])).build_plan(
        HardwareProfile.LIGHT, free_disk_gb=100.0
    )

    qwen = next(m for m in plan.models if m.model == "qwen3:4b")
    assert qwen.installed is True
    assert plan.ready is True


def test_manually_installed_family_with_another_tag_counts_as_installed():
    plan = ModelProvisioner(FakeProvider(installed=["qwen3:8b"])).build_plan(
        HardwareProfile.LIGHT, free_disk_gb=100.0
    )

    qwen = next(m for m in plan.models if m.model == "qwen3:4b")
    assert qwen.installed is True


def test_ring_is_ready_when_at_least_one_model_is_present():
    plan = ModelProvisioner(FakeProvider(installed=["llama3.1:8b"])).build_plan(
        HardwareProfile.LIGHT, free_disk_gb=100.0
    )

    assert plan.ready is True
    assert len(plan.missing) == 2


def test_download_of_a_model_outside_the_profile_ring_is_rejected(provisioner):
    with pytest.raises(ValidationError, match="не входит в кольцо"):
        provisioner.start_download("mistral:7b", HardwareProfile.LIGHT)


def test_progress_for_a_never_started_download_is_not_found(provisioner):
    with pytest.raises(NotFoundError, match="не запускалась"):
        provisioner.progress("qwen3:4b")


def test_starting_a_download_registers_pending_progress(provisioner):
    progress = provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)

    assert progress.state == DownloadState.PENDING
    assert provisioner.progress("qwen3:4b") is progress


def test_repeated_start_does_not_duplicate_an_active_download(provisioner):
    first = provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)
    second = provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)

    assert first is second
    assert len(provisioner.all_progress()) == 1


def test_download_streams_progress_to_completion(monkeypatch, provisioner):
    lines = [
        '{"status": "pulling manifest"}',
        '{"status": "downloading", "total": 1000, "completed": 250}',
        '{"status": "downloading", "total": 1000, "completed": 1000}',
        '{"status": "success"}',
    ]
    monkeypatch.setattr("app.llm.provisioning.httpx.stream", _stream_stub(lines))
    provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)

    progress = provisioner.run_download("qwen3:4b")

    assert progress.state == DownloadState.COMPLETED
    assert progress.percent == 100.0
    assert progress.completed_bytes == 1000


def test_malformed_progress_line_does_not_break_the_download(monkeypatch, provisioner):
    lines = ["не json вовсе", '{"total": 10, "completed": 10}']
    monkeypatch.setattr("app.llm.provisioning.httpx.stream", _stream_stub(lines))
    provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)

    progress = provisioner.run_download("qwen3:4b")

    assert progress.state == DownloadState.COMPLETED


def test_error_line_in_the_stream_marks_the_download_failed(monkeypatch, provisioner):
    lines = ['{"error": "model not found"}']
    monkeypatch.setattr("app.llm.provisioning.httpx.stream", _stream_stub(lines))
    provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)

    provisioner.run_download("qwen3:4b")

    progress = provisioner.progress("qwen3:4b")
    assert progress.error == "model not found"


def test_unreachable_runtime_fails_the_download_with_inference_error(
    monkeypatch, provisioner
):
    def broken_stream(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.llm.provisioning.httpx.stream", broken_stream)
    provisioner.start_download("qwen3:4b", HardwareProfile.LIGHT)

    with pytest.raises(InferenceError, match="Не удалось загрузить"):
        provisioner.run_download("qwen3:4b")

    assert provisioner.progress("qwen3:4b").state == DownloadState.FAILED


def _stream_stub(lines: list[str]):
    class _Response:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(lines)

    class _Stream:
        def __enter__(self):
            return _Response()

        def __exit__(self, *args):
            return False

    def stream(*args, **kwargs):
        return _Stream()

    return stream
