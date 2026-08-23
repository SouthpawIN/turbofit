from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.controller import load_rung_requirements
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.manual_profiles import build_manual_profile_payload, write_manual_profile
from turbofit_runtime.profile_io import load_profile
from turbofit_runtime.recipes import ResolvedComponent, ResolvedRecipe
from turbofit_runtime.routes import load_runtime_resolutions


def hardware() -> HardwareFingerprint:
    return HardwareFingerprint(
        "linux", "x86_64", 262_144,
        devices=(
            AcceleratorDevice(0, "gpu-0", "RTX", "nvidia", "cuda", 24_576, "8.6", "01"),
            AcceleratorDevice(1, "gpu-1", "RTX", "nvidia", "cuda", 24_576, "8.6", "02"),
        ),
    )


def recipe() -> ResolvedRecipe:
    return ResolvedRecipe(
        row_id="main--aux--64k",
        profile_name="main-aux-64k",
        main_alias="main-q4",
        aux_alias="aux-q4",
        aux_mode="dedicated",
        components=(
            ResolvedComponent("main", "main-family", "main-q4", "process", "baseline", "1", 11605, ("/bin/true",)),
            ResolvedComponent("aux", "aux-family", "aux-q4", "process", "baseline", "0", 11610, ("/bin/true",)),
        ),
    )


def shared_main_recipe() -> ResolvedRecipe:
    return ResolvedRecipe(
        row_id="bonsai--auto--64k",
        profile_name="bonsai-auto-64k",
        main_alias="bonsai-27b",
        aux_alias="auto:bonsai-27b",
        aux_mode="shared-main",
        components=(
            ResolvedComponent("main", "bonsai", "bonsai-27b", "process", "baseline", "0", 11605, ("/bin/true",)),
        ),
    )


def entry() -> dict:
    return {
        "context": 65_536,
        "production_recipe_sha256": "sha256:" + "a" * 64,
        "metrics": {"gpu_peak_mb": {"0": 8_000, "1": 12_000}},
    }


def test_manual_profile_contains_exact_local_rung_and_safe_api_fallback() -> None:
    profile, resolutions, requirements = build_manual_profile_payload(
        profile_id="manual-main-aux-64k",
        profile_entry=entry(),
        recipe=recipe(),
        hardware=hardware(),
    )

    assert [item["id"] for item in profile["rungs"]] == ["manual-exact", "api"]
    assert profile["rungs"][0]["aux_mode"] == "dedicated"
    assert set(resolutions["profiles"]["manual-main-aux-64k"]["manual-exact"]) == {"main", "aux"}
    assert requirements["profiles"]["manual-main-aux-64k"][0]["required_mb_per_card"] == [8_000, 12_000]
    assert requirements["profiles"]["manual-main-aux-64k"][1]["required_mb_per_card"] == []


def test_shared_main_profile_retargets_measured_residency_to_one_six_gb_card() -> None:
    single_gpu = HardwareFingerprint(
        "linux", "x86_64", 14_336,
        devices=(AcceleratorDevice(0, "gpu-0", "RTX 2060", "nvidia", "cuda", 6_144, "7.5", "01"),),
    )
    portable_entry = {
        "context": 65_536,
        "metrics": {"gpu_peak_mb": {"0": 6_117, "1": 195}},
    }

    profile, resolutions, requirements = build_manual_profile_payload(
        profile_id="manual-bonsai-auto-64k",
        profile_entry=portable_entry,
        recipe=shared_main_recipe(),
        hardware=single_gpu,
    )

    assert profile["hardware"]["topology"] == "1x6"
    assert requirements["profiles"]["manual-bonsai-auto-64k"][0]["required_mb_per_card"] == [5_120]
    assert set(resolutions["profiles"]["manual-bonsai-auto-64k"]["manual-exact"]) == {"main"}


def test_written_manual_sidecars_load_through_production_validators(tmp_path: Path) -> None:
    profile_path, resolutions_path, requirements_path = write_manual_profile(
        tmp_path,
        profile_id="manual-main-aux-64k",
        profile_entry=entry(),
        recipe=recipe(),
        hardware=hardware(),
    )

    profile = load_profile(profile_path)
    resolutions = load_runtime_resolutions(resolutions_path)
    requirements = load_rung_requirements(requirements_path, profile)

    assert profile.id == "manual-main-aux-64k"
    assert resolutions[profile.id]["manual-exact"]["main"]["family"] == "main-family"
    assert requirements.required_mb_by_rung == ((8_000, 12_000), ())
    assert json.loads(requirements_path.read_text())["schema"] == "turbofit.rung-requirements/v1"
