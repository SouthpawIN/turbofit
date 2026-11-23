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
    assert is_banned("qwen3.5-9b")
    assert not is_allowed_main("qwen3.5-9b")
    assert is_banned("qwen3-8-27b-uncensored-mlx-2bit")
    assert not is_allowed_main("qwen3-8-27b-uncensored-mlx-2bit")


def test_eight_gb_without_32gb_ram_still_offers_cyber_tiel_from_disk() -> None:
    assert expert_residency(host_ram_gb=16)["mode"] == "disk"
    options = check_local_options(vram_gb=8, host_ram_gb=16)
    aliases = [item["alias"] for item in options]
    assert "maple-preview-tq2" in aliases
    assert "cyber-tiel-coder-35b-a3b-ud-iq3-xxs" in aliases
    assert low_vram_moe_main() == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"


def test_eight_gb_aux_does_not_stack_two_models() -> None:
    assert local_aux_for_host(vram_gb=8, host_ram_gb=16) == "maple-preview-tq2"
    assert is_allowed_aux("auto:bonsai-27b")
    assert local_aux_for_host(vram_gb=24, host_ram_gb=16) == "maple-preview-tq2"
    assert local_aux_for_host(vram_gb=48, host_ram_gb=64) == "maple-preview-tq2"
    assert local_aux_for_host(vram_gb=64, host_ram_gb=64) == "maple-preview-tq2"


def test_maple_is_the_only_real_auxiliary() -> None:
    assert is_allowed_aux("maple-preview-tq2")
    assert is_allowed_aux("auto")
    assert not is_allowed_aux("carwin-nano")
    assert not is_allowed_aux("carwin-moe-nano")
    assert not is_allowed_aux("cyber-tiel-coder-35b-a3b-ud-q4-k-xl")
    assert not is_allowed_aux("qwen3-8-flash-next-strata")
    assert not is_allowed_aux("qwen3-8-27b-gsq-rco-iq3s")


def test_gsq_rco_bands() -> None:
    assert local_main_for_vram_gb(16) == "qwen3-8-27b-gsq-rco-iq3xxs"
    assert local_main_for_vram_gb(24) == "qwen3-8-27b-gsq-rco-iq3s"
    assert local_main_for_vram_gb(96) == "qwen3-8-27b-bf16"
    from turbofit_runtime.allowed_lineup import speed_family_rank, speed_main_for_vram_gb

    assert speed_main_for_vram_gb(8) == "maple-preview-tq2"
    assert speed_main_for_vram_gb(16) == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"
    assert speed_main_for_vram_gb(24) == "cyber-tiel-coder-35b-a3b-ud-q4-k-xl"
    assert speed_main_for_vram_gb(48) == "cyber-tiel-coder-35b-a3b-ud-q4-k-xl"
    assert speed_family_rank("cyber-tiel-coder-35b-a3b-ud-q4-k-xl") > speed_family_rank("qwen3-8-27b-gsq-rco-iq3s")
    options = check_local_options(vram_gb=24, host_ram_gb=64)
    aliases = [item["alias"] for item in options if item["role"] == "main"]
    assert "qwen3-8-27b-gsq-rco-iq3s" in aliases
    assert "cyber-tiel-coder-35b-a3b-ud-q4-k-xl" in aliases


def test_shared_total_memory_is_not_treated_as_vram() -> None:
    assert shared_main_for_total_memory_gb(8) == "maple-preview-tq2"
    assert shared_main_for_total_memory_gb(16) == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"
    assert shared_main_for_total_memory_gb(24) == "qwen3-8-27b-gsq-rco-iq3s"


def test_apple_uses_gsq_rco_fit_list_not_orcarouter_mlx() -> None:
    assert apple_mlx_main(unified_ram_gb=16) == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"
    assert apple_mlx_main(unified_ram_gb=24) == "qwen3-8-27b-gsq-rco-iq3s"
    assert apple_mlx_main(unified_ram_gb=32) == "qwen3-8-27b-gsq-rco-iq3s"
    assert apple_mlx_main(unified_ram_gb=64) == "qwen3-8-27b-gsq-rco-iq3s"
    apple = check_local_options(
        vram_gb=0, host_ram_gb=32, memory_pool="unified", backend="metal", vendor="apple"
    )
    assert apple[0]["alias"] == "qwen3-8-27b-gsq-rco-iq3s"
    assert apple[0]["engine"] == "llama.cpp"
    assert apple[0]["repo"] is None


def test_integrated_unified_uses_total_memory_bands() -> None:
    igpu = check_local_options(
        vram_gb=8, host_ram_gb=32, memory_pool="unified", backend="vulkan", vendor="amd"
    )
    aliases = [item["alias"] for item in igpu]
    assert aliases[0] == "qwen3-8-27b-gsq-rco-iq3s", "MoE stays the unified default"
    assert "arete-27b-q4" in aliases, "dense exception is offered at 24 GB+ pool"
    assert aliases[-1] == "maple-preview-tq2", "aux stays last"
    small = check_local_options(
        vram_gb=2, host_ram_gb=8, memory_pool="unified", backend="vulkan", vendor="amd"
    )
    assert [item["alias"] for item in small] == ["maple-preview-tq2", "maple-preview-tq2"]


def test_filter_drops_nine_b() -> None:
    assert filter_aliases(
        ["maple-preview-tq2", "qwen3.5-9b", "qwen3-8-27b-gsq-rco-iq3s"],
        role="main",
    ) == ["maple-preview-tq2", "qwen3-8-27b-gsq-rco-iq3s"]


def test_eight_gb_vram_thirty_two_gb_ram_is_maple_or_cyber_tiel_not_dense_27b() -> None:
    from turbofit_runtime.allowed_lineup import (
        main_matches_recommendable,
        recommendable_mains_for_hardware,
    )
    from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint

    hardware = HardwareFingerprint(
        "linux",
        "x86_64",
        32 * 1024,
        devices=(
            AcceleratorDevice(0, "gpu-0", "8gb", "nvidia", "cuda", 8 * 1024, "8.6", None),
        ),
    )
    allowed = recommendable_mains_for_hardware(hardware)
    assert allowed == frozenset({"maple-preview-tq2", "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"})
    assert main_matches_recommendable("maple-preview-tq2-128k-main", allowed)
    assert main_matches_recommendable("cyber-tiel-coder-35b-a3b-ud-iq3-xxs", allowed)
    assert not main_matches_recommendable("qwen3-8-27b-gsq-rco-iq3s", allowed)
    assert not main_matches_recommendable("qwen3-8-27b-q4-k-m", allowed)


def test_sixteen_gb_gets_the_64k_floor_not_a_dense_27b_that_cannot_hold_it() -> None:
    """A 27B Q3/Q4 cannot hold 64K on a 12-16 GB card; the ladder says so."""
    eight = [o["alias"] for o in check_local_options(vram_gb=8, host_ram_gb=64)]
    twelve = [o["alias"] for o in check_local_options(vram_gb=12, host_ram_gb=64)]
    sixteen = [o["alias"] for o in check_local_options(vram_gb=16, host_ram_gb=64)]
    sixtyfour = [o["alias"] for o in check_local_options(vram_gb=64, host_ram_gb=64)]

    assert "arete-27b-q3" not in twelve, "12.5 GiB of weights cannot hold 64K at 12 GB"
    assert "qwen3-8-27b-gsq-rco-iq3xxs" in twelve
    assert "arete-27b-q4" not in sixteen, "16.9 GiB of weights cannot hold 64K at 16 GB"
    assert "arete-27b-q4" not in eight
    assert "arete-27b-q8" in sixtyfour
    for alias in ("arete-27b-q3", "arete-27b-q4", "arete-27b-q5", "arete-27b-q8"):
        assert is_allowed_main(alias), f"{alias} must survive filter_aliases"


def test_matrix_alternatives_are_offered_not_defaulted() -> None:
    """New families appear as options; the band default is unchanged."""
    sixteen = [o["alias"] for o in check_local_options(vram_gb=16, host_ram_gb=64)]
    assert "flint-40a2-5b-mix12g" in sixteen
    assert sixteen[0] == "qwen3-8-27b-gsq-rco-iq3xxs", "default must not shift"
    # Strata is only claimed when the caller passes the probed CPU flags.
    assert "qwen3-8-flash-next-iq3xxs" not in sixteen
    capable = [
        o["alias"] for o in check_local_options(
            vram_gb=64, host_ram_gb=64, cpu_flags={"avx2"},
        )
    ]
    assert str(capable[0]).startswith("inkling-2-mini"), "Inkling leads once Strata is eligible"
    assert any(str(a).startswith("qwen3-8-flash-next") for a in capable), (
        "Qwen Flash Next is next in the chain"
    )


def test_glm_maya_is_offered_where_project_maya_says_it_runs() -> None:
    """Maya-XS targets 64 GB RAM + a 24 GB card; it must not appear without the RAM."""
    from turbofit_runtime.allowed_lineup import strata_options

    thin = [o["alias"] for o in check_local_options(vram_gb=24, host_ram_gb=32)]
    assert "glm-5-3-flash-maya-xs" not in thin

    rich = [
        o["alias"] for o in check_local_options(
            vram_gb=24, host_ram_gb=64, cpu_flags={"avx2"},
        )
    ]
    assert rich[0].startswith("inkling-2-mini"), "Inkling leads the Strata chain by owner preference"
    # The recommendation lists the strongest three rows; Inkling SQ-20 (262K,
    # native lossless, fully resident) earns a slot ahead of the 128K Maya
    # packs. Maya-XS stays admitted — the tier view keeps the full set for
    # audit even when it is outside the top three.
    assert "inkling-2-mini-sq20" in rich
    _preferred, view24 = strata_options(
        vram_gb=24, host_ram_gb=64, cpu_flags={"avx2"},
    )
    assert view24 is not None
    assert "glm-5-3-flash-maya-xs" in {row.model_id for row in view24.admitted}

    _preferred, view = strata_options(
        vram_gb=64, host_ram_gb=64, cpu_flags={"avx2"},
    )
    assert view is not None
    assert "glm-5-3-flash-maya-xs" in {row.model_id for row in view.admitted}
    assert "glm-5-3-flash-maya-s24" in {row.model_id for row in view.admitted}

    small = [
        o["alias"] for o in check_local_options(
            vram_gb=8, host_ram_gb=64, cpu_flags={"avx2"},
        )
    ]
    assert not any(a.startswith("glm-5-3-flash") for a in small), "Maya-XS is 24 GB+"


def test_unified_offers_arete_as_dense_exception() -> None:
    options = check_local_options(
        vram_gb=0, host_ram_gb=16, memory_pool="unified", backend="cpu", vendor="other"
    )
    mains = [o["alias"] for o in options if o["role"] == "main"]
    assert "arete-27b-q3" in mains
    assert mains[0] == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs", "MoE stays the unified default"
