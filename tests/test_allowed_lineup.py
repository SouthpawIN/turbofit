from turbofit_runtime.allowed_lineup import (
    apple_mlx_main,
    check_local_options,
    expert_residency,
    filter_aliases,
    is_allowed_aux,
    is_allowed_main,
    is_banned,
    local_aux_for_host,
    local_main_for_vram_gb,
    low_vram_moe_main,
    shared_main_for_total_memory_gb,
)


def test_low_vram_is_maple_not_nine_b() -> None:
    assert local_main_for_vram_gb(8) == "maple-preview-tq2"
    assert is_banned("ornith-9b")
    assert not is_allowed_main("ornith-9b")
    assert is_banned("qwen3-8-27b-uncensored-mlx-2bit")
    assert not is_allowed_main("qwen3-8-27b-uncensored-mlx-2bit")


def test_eight_gb_without_32gb_ram_still_offers_ornith_from_disk() -> None:
    assert expert_residency(host_ram_gb=16)["mode"] == "disk"
    options = check_local_options(vram_gb=8, host_ram_gb=16)
    aliases = [item["alias"] for item in options]
    assert "maple-preview-tq2" in aliases
    assert "ornith-1-5-35a3b" in aliases
    assert low_vram_moe_main() == "ornith-1-5-35a3b"


def test_eight_gb_aux_does_not_stack_two_models() -> None:
    assert local_aux_for_host(vram_gb=8, host_ram_gb=16) == "auto"
    assert is_allowed_aux("auto:bonsai-27b")
    assert local_aux_for_host(vram_gb=24, host_ram_gb=16) == "ornith-1-5-35a3b"


def test_unleashed_bands() -> None:
    assert local_main_for_vram_gb(16) == "qwen3-8-27b-unleashed-ud-iq3-xxs"
    assert local_main_for_vram_gb(24) == "qwen3-8-27b-unleashed-ud-q3-k-xl"
    assert local_main_for_vram_gb(96) == "qwen3-8-27b-bf16"


def test_shared_total_memory_is_not_treated_as_vram() -> None:
    assert shared_main_for_total_memory_gb(8) == "maple-preview-tq2"
    assert shared_main_for_total_memory_gb(16) == "ornith-1-5-35a3b"
    assert shared_main_for_total_memory_gb(24) == "qwen3-8-27b-unleashed-ud-q3-k-xl"


def test_apple_uses_orcarouter_mlx_not_two_bit() -> None:
    assert apple_mlx_main(unified_ram_gb=16) == "ornith-1-5-35a3b"
    assert apple_mlx_main(unified_ram_gb=24) == "qwen3-8-27b-uncensored-mlx-4bit"
    assert apple_mlx_main(unified_ram_gb=32) == "qwen3-8-27b-uncensored-mlx-6bit"
    assert apple_mlx_main(unified_ram_gb=64) == "qwen3-8-27b-uncensored-mlx-8bit"
    apple = check_local_options(
        vram_gb=0, host_ram_gb=32, memory_pool="unified", backend="metal", vendor="apple"
    )
    assert apple[0]["alias"] == "qwen3-8-27b-uncensored-mlx-6bit"
    assert apple[0]["repo"] == "orcarouter/Qwen3.8-27B-Uncensored-MLX"


def test_integrated_unified_uses_total_memory_bands() -> None:
    igpu = check_local_options(
        vram_gb=8, host_ram_gb=32, memory_pool="unified", backend="vulkan", vendor="amd"
    )
    assert [item["alias"] for item in igpu] == ["qwen3-8-27b-unleashed-ud-q3-k-xl"]
    small = check_local_options(
        vram_gb=2, host_ram_gb=8, memory_pool="unified", backend="vulkan", vendor="amd"
    )
    assert [item["alias"] for item in small] == ["maple-preview-tq2"]


def test_filter_drops_nine_b() -> None:
    assert filter_aliases(
        ["maple-preview-tq2", "ornith-9b", "qwen3-8-27b-unleashed-ud-q3-k-xl"],
        role="main",
    ) == ["maple-preview-tq2", "qwen3-8-27b-unleashed-ud-q3-k-xl"]
