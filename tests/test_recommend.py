from __future__ import annotations

from dataclasses import replace

import pytest

from test_runtime_profile import valid_mapping
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.recommend import (
    EvidenceCandidate,
    LiveRuntimeState,
    NoRecommendation,
    hardware_satisfies,
    priority_key,
    recommend,
)
from turbofit_runtime.runtime_profile import Turbofile


EVIDENCE = "sha256:" + "e" * 64


def hardware(*memory_mb: int) -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=131072,
        devices=tuple(
            AcceleratorDevice(
                index=index,
                uuid=f"GPU-{index}",
                name="GPU",
                vendor="nvidia",
                backend="cuda",
                memory_total_mb=memory,
                compute_capability="8.6",
                bus_id=f"0{index}",
            )
            for index, memory in enumerate(memory_mb)
        ),
    )


def profile(profile_id: str, *, topology: str, context: int = 131072) -> Turbofile:
    mapping = valid_mapping()
    mapping["id"] = profile_id
    mapping["hardware"]["topology"] = topology
    sizes = [int(part.split("x", 1)[1]) for part in topology.split("+")]
    counts = [int(part.split("x", 1)[0]) for part in topology.split("+")]
    mapping["hardware"]["min_devices"] = sum(counts)
    mapping["hardware"]["per_device_min_gb"] = min(sizes)
    mapping["hardware"]["class_vram_gb"] = min(sizes)
    mapping["hardware"]["total_vram_gb"] = sum(
        count * size for count, size in zip(counts, sizes, strict=True)
    )
    for rung in mapping["rungs"]:
        rung["context"] = context
    return Turbofile.from_mapping(mapping)


def candidate(
    item: Turbofile,
    *,
    quality: int = 20,
    tps: float = 30,
    evidence: str | None = EVIDENCE,
    variant: str = "quality-first",
) -> EvidenceCandidate:
    return EvidenceCandidate(
        profile=item,
        quality_rank=quality,
        min_tps=tps,
        evidence_digest=evidence,
        policy_variant=variant,
    )


def live(item: HardwareFingerprint, free: tuple[int, ...]) -> LiveRuntimeState:
    return LiveRuntimeState(
        hardware=item,
        current_profile_id="current-profile",
        current_rung_id="shared-main-128k",
        free_vram_mb_by_uuid={f"GPU-{index}": value for index, value in enumerate(free)},
    )


def test_recommendation_uses_physical_capacity_not_current_free_vram() -> None:
    physical = hardware(24576)
    candidates = (candidate(profile("quality-24", topology="1x24")),)

    empty = recommend(live(physical, (0,)), candidates)
    clear = recommend(live(physical, (24576,)), candidates)

    assert empty.recommended.profile.id == clear.recommended.profile.id == "quality-24"
    assert empty.current_profile_id == "current-profile"
    assert empty.current_rung_id == "shared-main-128k"


def test_topology_constraint_distinguishes_one_48_from_two_24() -> None:
    candidates = (
        candidate(profile("one-48", topology="1x48"), quality=30),
        candidate(profile("two-24", topology="2x24"), quality=20),
    )

    result = recommend(live(hardware(24576, 24576), (100, 100)), candidates)

    assert result.recommended.profile.id == "two-24"
    assert tuple(item.profile.id for item in result.eligible) == ("two-24",)


def test_portable_llama_cpp_profile_matches_cuda_and_metal_backends() -> None:
    portable = profile("portable-24", topology="1x24")
    portable = replace(
        portable,
        hardware=replace(portable.hardware, accelerator="llama.cpp-local", compute_capability_min=None),
    )
    metal = HardwareFingerprint(
        os="darwin",
        architecture="arm64",
        system_ram_mb=98304,
        devices=(AcceleratorDevice(0, "apple-unified-memory", "Apple Silicon", "apple", "metal", 98304, None, None),),
    )

    assert hardware_satisfies(hardware(24576), portable.hardware)
    assert hardware_satisfies(metal, portable.hardware)
    cuda_only = replace(portable, hardware=replace(portable.hardware, accelerator="nvidia-cuda"))
    assert not hardware_satisfies(metal, cuda_only.hardware)


def test_only_candidates_with_valid_evidence_are_eligible() -> None:
    item = profile("quality-24", topology="1x24")
    candidates = (
        candidate(item, evidence=None, quality=100),
        candidate(replace(item, id="evidenced"), quality=10),
    )

    result = recommend(live(hardware(24576), (24576,)), candidates)

    assert result.recommended.profile.id == "evidenced"


def test_named_policy_variant_is_preserved_and_selectable() -> None:
    item = profile("quality-24", topology="1x24")
    candidates = (
        candidate(item, variant="quality-first", quality=30),
        candidate(replace(item, id="fast-24"), variant="speed-first", quality=20, tps=100),
    )

    result = recommend(
        live(hardware(24576), (100,)), candidates, policy_variant="speed-first"
    )

    assert result.recommended.policy_variant == "speed-first"
    assert result.policy_variant == "speed-first"


def test_priority_is_lexicographic_without_weighted_score() -> None:
    assert priority_key(131072, 5, 20) > priority_key(65536, 200, 20)
    assert priority_key(262144, 30, 20) > priority_key(131072, 100, 20)
    assert priority_key(1048576, 100, 20) > priority_key(262144, 100, 20)
    assert priority_key(262144, 60, 20) > priority_key(1048576, 100, 10)


def test_twenty_tokens_per_second_is_interactive() -> None:
    assert priority_key(262144, 100, 20)[2] == 20


def test_clear_card_shortfall_has_no_eligible_recommendation() -> None:
    candidates = (candidate(profile("quality-24", topology="1x24")),)

    with pytest.raises(NoRecommendation, match="physical hardware"):
        recommend(live(hardware(16384), (16384,)), candidates)


def test_candidate_ranking_inputs_reject_booleans_and_nonfinite_values() -> None:
    item = profile("quality-24", topology="1x24")
    with pytest.raises(ValueError, match="quality_rank"):
        candidate(item, quality=True)
    with pytest.raises(ValueError, match="min_tps"):
        candidate(item, tps=float("nan"))
    with pytest.raises(ValueError, match="policy_variant"):
        candidate(item, variant="")


def test_live_free_vram_metadata_is_copied_and_immutable() -> None:
    values = {"GPU-0": 100}
    state = LiveRuntimeState(hardware(24576), None, None, values)
    values["GPU-0"] = 0

    assert state.free_vram_mb_by_uuid["GPU-0"] == 100
    with pytest.raises(TypeError):
        state.free_vram_mb_by_uuid["GPU-0"] = 0  # type: ignore[index]


def test_accelerator_pair_match_does_not_cross_product_mixed_devices() -> None:
    mixed = HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=131072,
        devices=(
            AcceleratorDevice(0, "n", "N", "nvidia", "cuda", 24576, "8.6", "01"),
            AcceleratorDevice(1, "a", "A", "amd", "rocm", 24576, None, "02"),
        ),
    )
    constraint = replace(
        profile("quality-24", topology="1x24").hardware,
        min_devices=1,
        total_vram_gb=24,
        per_device_min_gb=24,
        accelerator="amd-cuda",
        compute_capability_min=None,
        topology="any",
    )

    assert hardware_satisfies(mixed, constraint) is False
