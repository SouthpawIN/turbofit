from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.tier_report import _exact_physical_topology, build_tier_report


ROOT = Path(__file__).parents[1]


def host() -> HardwareFingerprint:
    return HardwareFingerprint(
        "linux", "x86_64", 393216,
        devices=(
            AcceleratorDevice(0, "GPU-0", "RTX 3090", "nvidia", "cuda", 24576, "8.6", "01"),
            AcceleratorDevice(1, "GPU-1", "RTX 3090", "nvidia", "cuda", 24576, "8.6", "02"),
        ),
    )


def test_exact_physical_topology_is_strict_except_for_300_plus() -> None:
    hardware = host()
    assert _exact_physical_topology(hardware, {
        "total_vram_gb": 48, "min_devices": 2, "per_device_min_gb": 24,
    })
    assert not _exact_physical_topology(hardware, {
        "total_vram_gb": 24, "min_devices": 1, "per_device_min_gb": 24,
    })
    larger = HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=1_048_576,
        devices=tuple(
            AcceleratorDevice(
                index=index, uuid=f"GPU-{index}", name="GPU", vendor="nvidia",
                backend="cuda", memory_total_mb=102400, compute_capability="9.0",
                bus_id=f"{index:02d}",
            )
            for index in range(4)
        ),
    )
    assert _exact_physical_topology(larger, {
        "total_vram_gb": 300, "min_devices": 3, "per_device_min_gb": 100,
    })


def test_tier_report_covers_exact_project_tiers_and_current_machine() -> None:
    report = build_tier_report(ROOT, host())

    assert [tier["capacity_gb"] for tier in report["tiers"]] == [8, 16, 24, 48, 64, 96, 200, 300]
    assert report["current_hardware"]["native_tier_gb"] == 48
    assert report["current_hardware"]["system_ram_mb"] == 393216
    assert report["current_hardware"]["memory_pool_kind"] == "dedicated"
    assert report["current_hardware"]["host_usable_memory_mb"] == 385024


def test_tier_candidates_reject_stale_winner_and_keep_requirements_separate() -> None:
    report = build_tier_report(ROOT, host())
    tier48 = next(tier for tier in report["tiers"] if tier["capacity_gb"] == 48)
    candidate = next(
        item for item in tier48["candidates"]
        if item["configuration_id"] == "qwen3-8-27b-unleashed-ud-q3-k-xl--auto--262k"
    )

    assert tier48["recommendations"]["measured_winner"] is None
    assert tier48["status"] == "catalog-candidates-only"
    assert candidate["artifact_storage_bytes"] > 0
    assert candidate["host_memory_requirement"]["inferred_artifact_load_floor_gb"] > 8
    assert candidate["host_memory_requirement"]["physically_measured_peak_gb"] is None
    assert candidate["accelerator_requirement"]["aggregate_gb"] == 48
    assert candidate["accelerator_requirement"]["per_device_min_gb"] == 24
    assert candidate["fit"]["inferred"] is True
    assert candidate["fit"]["physically_demonstrated"] is False
    assert candidate["intelligence_score"] is None


def test_unproven_tier_is_not_misrepresented_as_measured_recommendation() -> None:
    report = build_tier_report(ROOT, host())
    tier96 = next(tier for tier in report["tiers"] if tier["capacity_gb"] == 96)

    assert tier96["recommendations"]["measured_winner"] is None
    assert tier96["status"] == "catalog-candidates-only"
    assert all(not item["fit"]["physically_demonstrated"] for item in tier96["candidates"])
