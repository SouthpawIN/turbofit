from __future__ import annotations

import runpy
from pathlib import Path

from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/turbofit-runtime-recommend"),
    run_name="turbofit_runtime_recommend",
)
priority_key = MODULE["priority_key"]
hardware_memory_budgets = MODULE["hardware_memory_budgets"]
fit_requirements = MODULE["fit_requirements"]
recommendation_sort_key = MODULE["recommendation_sort_key"]
load_intelligence_scores = MODULE["load_intelligence_scores"]
current_intelligence_recipes = MODULE["current_intelligence_recipes"]
Recommendation = MODULE["Recommendation"]


def test_recommendation_ladder_prioritizes_128k_before_raw_speed() -> None:
    assert priority_key(131_072, 5.0, 20) > priority_key(65_536, 200.0, 20)


def test_recommendation_ladder_prioritizes_speed_until_30_tps() -> None:
    assert priority_key(131_072, 29.0, 20) > priority_key(262_144, 6.0, 20)


def test_recommendation_ladder_prioritizes_262k_after_30_tps() -> None:
    assert priority_key(262_144, 30.0, 20) > priority_key(131_072, 100.0, 20)


def test_recommendation_ladder_prioritizes_speed_until_50_tps_at_262k() -> None:
    assert priority_key(262_144, 49.0, 20) > priority_key(1_048_576, 31.0, 20)


def test_recommendation_ladder_prioritizes_1m_after_50_tps() -> None:
    assert priority_key(1_048_576, 50.0, 20) > priority_key(262_144, 50.0, 20)


def test_quality_tier_precedes_context_speed_ladder() -> None:
    assert priority_key(262_144, 60.0, 20) > priority_key(1_048_576, 100.0, 10)


def test_cpu_only_recommendations_use_system_ram_as_shared_memory() -> None:
    hardware = HardwareFingerprint("linux", "x86_64", 131072)

    budgets, shared = hardware_memory_budgets(hardware)

    assert budgets == {0: 124518}
    assert shared is True


def test_rocm_recommendations_use_each_physical_card_budget() -> None:
    hardware = HardwareFingerprint(
        "linux",
        "x86_64",
        65536,
        devices=(
            AcceleratorDevice(0, "AMD-0", "Radeon", "amd", "rocm", 24576, None, "01"),
            AcceleratorDevice(1, "AMD-1", "Radeon", "amd", "rocm", 24576, None, "02"),
        ),
    )

    budgets, shared = hardware_memory_budgets(hardware)

    assert budgets == {0: 24576, 1: 24576}
    assert shared is False


def test_unified_memory_is_not_counted_as_ram_plus_vram() -> None:
    hardware = HardwareFingerprint(
        "darwin",
        "arm64",
        65536,
        devices=(AcceleratorDevice(
            0, "apple-unified-memory", "Apple Silicon Unified Memory",
            "apple", "metal", 57344, None, None,
        ),),
    )

    budgets, shared = hardware_memory_budgets(hardware)

    assert budgets == {0: 62259}
    assert shared is True
    assert hardware.total_usable_memory_mb == 62259


def test_discrete_fit_uses_safe_system_ram_for_accelerator_spill() -> None:
    hardware = HardwareFingerprint(
        "linux", "x86_64", 65_536,
        devices=(AcceleratorDevice(0, "GPU-0", "GPU", "nvidia", "cuda", 8_192, "8.6", "01"),),
    )

    fit, reasons = fit_requirements(
        hardware=hardware,
        requirements={0: 40_000},
        available_budgets={0: 8_192},
        shared_memory=False,
        safety_floor_mb=1_024,
    )

    assert fit is True
    assert any("host spill" in reason for reason in reasons)


def _recommendation(name: str, *, tps: float, quality: int, context: int) -> object:
    return Recommendation(
        profile=name, score=0.0, quality_points=quality, context=context,
        main=name, aux="auto", aux_mode="shared-main", min_tps=tps,
        vision=False, methods=["baseline"], gpu_requirements_mb={}, fit=True,
        fit_reason="fit", confidence="measured", command="run", evidence="evidence",
    )


def test_speed_preference_can_choose_faster_lower_quality_profile() -> None:
    intelligent = _recommendation("intelligent", tps=20.0, quality=50, context=262_144)
    fast = _recommendation("fast", tps=80.0, quality=20, context=131_072)

    assert recommendation_sort_key(fast, "speed") > recommendation_sort_key(intelligent, "speed")
    assert recommendation_sort_key(intelligent, "intelligence") > recommendation_sort_key(fast, "intelligence")


def test_intelligence_scores_are_loaded_only_from_exact_measured_configuration_records(tmp_path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(__import__("json").dumps({
        "records": [{
            "main": "main-q4", "auxiliary": "aux-q4", "context": 131072,
            "intelligence_score": 73.5, "benchmark_level": "screening",
            "production_recipe_sha256": "sha256:" + "a" * 64,
        }]
    }))

    scores = load_intelligence_scores(path)

    assert scores[("main-q4", "aux-q4", 131072)]["score"] == 73.5
    assert ("main-q8", "aux-q4", 131072) not in scores


def test_auto_auxiliary_intelligence_uses_runtime_recipe_alias() -> None:
    recipes = current_intelligence_recipes()

    assert ("bonsai-27b", "auto:bonsai-27b", 65_536) in recipes
    assert ("bonsai-27b", "auto", 65_536) not in recipes
