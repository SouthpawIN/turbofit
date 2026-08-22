from turbofit_runtime.allowed_lineup import (
    check_local_options,
    expert_residency,
    filter_aliases,
    is_allowed_aux,
    is_allowed_main,
    is_banned,
    local_aux_for_host,
    local_main_for_vram_gb,
    low_vram_moe_main,
)


def test_low_vram_is_bonsai_not_nine_b() -> None:
    assert local_main_for_vram_gb(8) == "bonsai-27b"
    assert local_main_for_vram_gb(10) == "bonsai-27b"
    assert is_banned("ornith-9b")
    assert is_banned("Qwen3.5-9B")
    assert not is_allowed_main("ornith-9b")
    assert not is_allowed_main("qwen3.8-9b-cyber")


def test_eight_gb_without_32gb_ram_still_offers_ornith_from_disk() -> None:
    assert expert_residency(host_ram_gb=16)["mode"] == "disk"
    assert expert_residency(host_ram_gb=8)["mmap"] is True
    assert expert_residency(host_ram_gb=8)["mlock"] is False
    assert expert_residency(host_ram_gb=8)["cpu_moe"] is True
    assert expert_residency(host_ram_gb=64)["mode"] == "ram"
    options = check_local_options(vram_gb=8, host_ram_gb=16)
    aliases = [item["alias"] for item in options]
    assert "bonsai-27b" in aliases
    assert "ornith-1-5-35a3b" in aliases
    assert low_vram_moe_main() == "ornith-1-5-35a3b"
    assert is_allowed_main("ornith-1-5-35a3b")
    assert not is_allowed_aux("ornith-9b")


def test_eight_gb_aux_does_not_stack_two_models() -> None:
    assert local_aux_for_host(vram_gb=8, host_ram_gb=16) == "auto"
    assert local_aux_for_host(vram_gb=24, host_ram_gb=16) == "ornith-1-5-35a3b"


def test_unleashed_bands() -> None:
    assert local_main_for_vram_gb(16) == "qwen3-8-27b-unleashed-ud-iq3-xxs"
    assert local_main_for_vram_gb(24) == "qwen3-8-27b-unleashed-ud-q3-k-xl"
    assert local_main_for_vram_gb(48) == "qwen3-8-27b-unleashed-ud-q3-k-xl"
    assert local_main_for_vram_gb(96) == "qwen3-8-27b-bf16"


def test_filter_drops_nine_b() -> None:
    assert filter_aliases(
        ["bonsai-27b", "ornith-9b", "qwen3-8-27b-unleashed-ud-q3-k-xl"],
        role="main",
    ) == ["bonsai-27b", "qwen3-8-27b-unleashed-ud-q3-k-xl"]
