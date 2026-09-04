from app.hardware.detector import HardwareDetector
from app.hardware.models import GpuVendor


def test_detects_basic_cpu_ram_disk_via_psutil():
    detector = HardwareDetector(run_command=lambda args: None, system="Linux", machine="x86_64")

    info = detector.detect()

    assert info.cpu_count >= 1
    assert info.total_ram_mb > 0
    assert info.free_disk_mb > 0
    assert info.architecture == "x86_64"


def test_no_gpu_tooling_falls_back_to_cpu_only():
    detector = HardwareDetector(run_command=lambda args: None, system="Linux", machine="x86_64")

    info = detector.detect()

    assert info.gpu.vendor == GpuVendor.NONE
    assert info.is_cpu_only is True


def test_apple_silicon_is_detected_without_nvidia_tooling():
    detector = HardwareDetector(run_command=lambda args: None, system="Darwin", machine="arm64")

    info = detector.detect()

    assert info.gpu.vendor == GpuVendor.APPLE
    assert info.gpu.metal_available is True
    assert info.is_apple_silicon is True


def test_intel_mac_without_gpu_tooling_is_cpu_only():
    detector = HardwareDetector(run_command=lambda args: None, system="Darwin", machine="x86_64")

    info = detector.detect()

    assert info.gpu.vendor == GpuVendor.NONE


def test_nvidia_gpu_is_parsed_from_nvidia_smi_output():
    def run_command(args):
        if args[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 4090, 24564\n"
        return None

    detector = HardwareDetector(run_command=run_command, system="Linux", machine="x86_64")

    info = detector.detect()

    assert info.gpu.vendor == GpuVendor.NVIDIA
    assert info.gpu.name == "NVIDIA GeForce RTX 4090"
    assert info.gpu.vram_total_mb == 24564
    assert info.gpu.cuda_available is True


def test_nvidia_takes_priority_over_apple_silicon_flag():
    """На практике не сочетается, но приоритет должен быть детерминирован."""

    def run_command(args):
        return "Tesla T4, 16384\n" if args[0] == "nvidia-smi" else None

    detector = HardwareDetector(run_command=run_command, system="Darwin", machine="arm64")

    info = detector.detect()

    assert info.gpu.vendor == GpuVendor.NVIDIA


def test_malformed_nvidia_smi_output_falls_back_gracefully():
    detector = HardwareDetector(
        run_command=lambda args: "garbage output without commas\n",
        system="Linux",
        machine="x86_64",
    )

    info = detector.detect()

    assert info.gpu.vendor == GpuVendor.NONE


def test_command_runner_is_only_invoked_for_nvidia_smi():
    calls = []

    def run_command(args):
        calls.append(args[0])
        return None

    HardwareDetector(run_command=run_command, system="Linux", machine="x86_64").detect()

    assert calls == ["nvidia-smi"]
