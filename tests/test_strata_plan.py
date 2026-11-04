"""Strata 0.2.10 launch planning: the rules the release notes actually state."""

from __future__ import annotations

import pytest

from turbofit_runtime import strata
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint


def card(index: int, vram_mb: int, arch: str | None, vendor: str = "nvidia", backend: str = "cuda"):
    return AcceleratorDevice(
        index=index,
        uuid=f"gpu-{index}",
        name=f"{vendor} card {index}",
        vendor=vendor,
        backend=backend,
        memory_total_mb=vram_mb,
        compute_capability=arch,
        bus_id=f"0000:0{index}:00.0",
    )


def machine(os_name: str, ram_mb: int, devices, architecture: str = "x86_64"):
    return HardwareFingerprint(
        os=os_name,
        architecture=architecture,
        system_ram_mb=ram_mb,
        devices=tuple(devices),
    )


def test_eight_gigabyte_card_starts_at_128k_with_dflash2():
    plan = strata.plan(machine("linux", 32 * 1024, [card(0, 8 * 1024, "8.6")]))

    assert plan.context_tokens == strata.SMALL_CARD_CONTEXT == 131_072
    assert plan.drafting == "dflash2"
    assert plan.split_engine == "off"
    assert plan.quant == "IQ3_S"
    assert "--context" in plan.launch_args
    assert plan.launch_args[plan.launch_args.index("--context") + 1] == "131072"


def test_two_3090s_with_ram_get_the_split_engine_at_128k():
    plan = strata.plan(
        machine("linux", 64 * 1024, [card(0, 24 * 1024, "8.6"), card(1, 24 * 1024, "8.6")])
    )

    assert plan.split_engine == "auto"
    assert plan.drafting == "mtp"  # split + MTP decodes faster than layer split + DFlash2
    assert plan.context_tokens == strata.SPLIT_CONTEXT_CEILING
    assert plan.quant == "UD-Q4_K_XL"  # the file the split numbers were measured on
    assert any("split engine" in reason for reason in plan.reasons)


def test_two_3090s_without_the_ram_keep_the_layer_split():
    plan = strata.plan(
        machine("linux", 32 * 1024, [card(0, 24 * 1024, "8.6"), card(1, 24 * 1024, "8.6")])
    )

    assert plan.split_engine == "off"
    assert "43 GiB" in plan.reasons[0]
    assert plan.context_tokens == strata.FULL_CONTEXT
    assert plan.drafting == "dflash2"
    assert plan.quant == "IQ3_S"


def test_explicit_context_is_never_quietly_lowered():
    plan = strata.plan(
        machine("linux", 64 * 1024, [card(0, 24 * 1024, "8.6"), card(1, 24 * 1024, "8.6")]),
        context_tokens=262_144,
    )

    assert plan.context_tokens == 262_144
    assert plan.split_engine == "off"
    assert "caps context" in plan.reasons[0]


def test_two_concurrent_requests_force_mtp():
    plan = strata.plan(
        machine("linux", 64 * 1024, [card(0, 24 * 1024, "8.9")]), parallel=2
    )

    assert plan.drafting == "mtp"
    assert "--parallel" in plan.launch_args
    assert plan.launch_args[plan.launch_args.index("--parallel") + 1] == "2"


def test_deepseek_is_its_own_family_and_never_uses_dflash2():
    plan = strata.plan(
        machine("linux", 128 * 1024, [card(0, 80 * 1024, "12.0")]),
        family=strata.FAMILY_DEEPSEEK,
    )

    assert plan.model == "deepseek-v4.1-flash"
    assert plan.drafting == "mtp"
    assert plan.split_engine == "off"


def test_apple_silicon_runs_dflash2_on_metal():
    plan = strata.plan(
        machine(
            "darwin",
            64 * 1024,
            [card(0, 32 * 1024, None, vendor="apple", backend="metal")],
            architecture="arm64",
        )
    )

    assert plan.drafting == "dflash2"
    assert plan.split_engine == "off"
    assert plan.context_tokens == strata.FULL_CONTEXT


def test_intel_arc_gets_dflash2_and_the_measured_caveat():
    plan = strata.plan(
        machine("linux", 64 * 1024, [card(0, 8 * 1024, None, vendor="intel", backend="vulkan")])
    )

    assert plan.drafting == "dflash2"
    assert any("2.6% slower" in warning for warning in plan.warnings)


def test_split_is_linux_only():
    ok, reason = strata.split_engine_eligible(
        machine("windows", 64 * 1024, [card(0, 24 * 1024, "8.6"), card(1, 24 * 1024, "8.6")]),
        context_tokens=131_072,
        family=strata.FAMILY_QWEN,
    )

    assert ok is False
    assert "Linux-only" in reason


def test_three_cards_are_not_a_split_shape():
    plan = strata.plan(
        machine(
            "linux",
            128 * 1024,
            [card(0, 24 * 1024, "8.6"), card(1, 24 * 1024, "8.6"), card(2, 24 * 1024, "8.6")],
        )
    )

    assert plan.split_engine == "off"
    assert "not a split shape" in plan.reasons[0]


def test_mixed_strata_install_is_reported():
    problems = strata.version_problems("0.2.9", "0.1.55")

    assert len(problems) == 2
    assert strata.STRATA_APP_VERSION in problems[0]
    assert strata.STRATA_ENGINE_VERSION in problems[1]
    assert strata.version_problems(strata.STRATA_APP_VERSION, strata.STRATA_ENGINE_VERSION) == []


def test_update_feed_follows_the_architecture():
    assert strata.feed_url("linux", "x86_64") == strata.FEED_URL
    assert strata.feed_url("linux", "aarch64") == strata.FEED_URL_ARM64


def test_plan_round_trips_to_a_mapping():
    plan = strata.plan(machine("linux", 64 * 1024, [card(0, 24 * 1024, "8.6")]))
    mapping = plan.to_mapping()

    assert mapping["model"] == plan.model
    assert mapping["launch_args"] == list(plan.launch_args)
    assert mapping["context_tokens"] == plan.context_tokens
    assert mapping["evidence"] == "documented"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
