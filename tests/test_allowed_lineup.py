from turbofit_runtime.allowed_lineup import (
    filter_aliases,
    is_allowed_aux,
    is_allowed_main,
    is_banned,
    local_aux_for_host,
    local_main_for_vram_gb,
)


def test_low_vram_is_bonsai_not_nine_b() -> None:
    assert local_main_for_vram_gb(8) == "bonsai-27b"
    assert local_main_for_vram_gb(10) == "bonsai-27b"
    assert is_banned("ornith-9b")
    assert is_banned("Qwen3.5-9B")
    assert not is_allowed_main("ornith-9b")
    assert not is_allowed_main("qwen3.8-9b-cyber")


def test_low_vram_aux_is_ornith_15_when_ram_allows() -> None:
    assert local_aux_for_host(vram_gb=8, host_ram_gb=64) == "ornith-1-5-35a3b"
    assert local_aux_for_host(vram_gb=8, host_ram_gb=16) == "auto"
    assert is_allowed_aux("ornith-1-5-35a3b")
    assert not is_allowed_aux("ornith-9b")


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
