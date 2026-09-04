from app.hardware.models import GpuInfo, GpuVendor, HardwareInfo
from app.hardware.profiles import (
    HardwareProfile,
    ProfileRecommender,
    get_profile_definition,
)


def _hardware(ram_mb: int, gpu: GpuInfo | None = None) -> HardwareInfo:
    return HardwareInfo(
        cpu_count=8,
        architecture="x86_64",
        total_ram_mb=ram_mb,
        available_ram_mb=ram_mb,
        free_disk_mb=100_000,
        gpu=gpu or GpuInfo(vendor=GpuVendor.NONE),
    )


def test_cpu_only_with_little_ram_gets_light():
    recommender = ProfileRecommender()

    profile = recommender.recommend(_hardware(ram_mb=8 * 1024))

    assert profile == HardwareProfile.LIGHT


def test_cpu_only_with_lots_of_ram_still_gets_light_without_gpu():
    """min_vram_mb для STANDARD/PERFORMANCE требует GPU — RAM одной не хватает."""
    recommender = ProfileRecommender()

    profile = recommender.recommend(_hardware(ram_mb=128 * 1024))

    assert profile == HardwareProfile.LIGHT


def test_standard_gpu_gets_standard_profile():
    recommender = ProfileRecommender()
    gpu = GpuInfo(vendor=GpuVendor.NVIDIA, vram_total_mb=16 * 1024, cuda_available=True)

    profile = recommender.recommend(_hardware(ram_mb=32 * 1024, gpu=gpu))

    assert profile == HardwareProfile.STANDARD


def test_high_end_gpu_gets_performance_profile():
    recommender = ProfileRecommender()
    gpu = GpuInfo(vendor=GpuVendor.NVIDIA, vram_total_mb=48 * 1024, cuda_available=True)

    profile = recommender.recommend(_hardware(ram_mb=128 * 1024, gpu=gpu))

    assert profile == HardwareProfile.PERFORMANCE


def test_high_end_gpu_with_insufficient_ram_is_capped_below_performance():
    recommender = ProfileRecommender()
    gpu = GpuInfo(vendor=GpuVendor.NVIDIA, vram_total_mb=48 * 1024, cuda_available=True)

    profile = recommender.recommend(_hardware(ram_mb=32 * 1024, gpu=gpu))

    assert profile == HardwareProfile.STANDARD


def test_apple_silicon_uses_unified_memory_for_vram_threshold():
    recommender = ProfileRecommender()
    gpu = GpuInfo(vendor=GpuVendor.APPLE, metal_available=True)

    # 32 ГБ хватает STANDARD (RAM >= 24 ГБ, unified-VRAM >= 12 ГБ), но
    # недостаточно для PERFORMANCE (нужно 64 ГБ RAM).
    profile = recommender.recommend(_hardware(ram_mb=32 * 1024, gpu=gpu))

    assert profile == HardwareProfile.STANDARD


def test_apple_silicon_with_plenty_of_unified_memory_gets_performance():
    recommender = ProfileRecommender()
    gpu = GpuInfo(vendor=GpuVendor.APPLE, metal_available=True)

    profile = recommender.recommend(_hardware(ram_mb=64 * 1024, gpu=gpu))

    assert profile == HardwareProfile.PERFORMANCE


def test_each_profile_ring_covers_qwen_gemma_llama_in_order():
    for profile in HardwareProfile:
        definition = get_profile_definition(profile)
        assert [entry.family for entry in definition.ring] == ["qwen", "gemma", "llama"]


def test_profile_ring_model_sizes_follow_each_familys_own_lineup():
    light = get_profile_definition(HardwareProfile.LIGHT)
    performance = get_profile_definition(HardwareProfile.PERFORMANCE)

    assert light.ring[0].model == "qwen3:4b"
    assert performance.ring[0].model == "qwen3:32b"
    assert performance.ring[2].model == "llama3.1:70b"
