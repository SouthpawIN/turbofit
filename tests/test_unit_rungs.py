"""Externally-managed rungs: an init system owns the process, we own the unit.

A box may already run a model server under systemd (Strata, a bespoke install,
a distro package). TurboFit must be able to treat that server as a ladder rung —
including releasing it under external GPU pressure — without ever signalling a
PID it did not launch. The portable profile declares the unit, which is the
whole authority boundary.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from test_runtime_profile import DIGEST_A, valid_mapping
from turbofit_runtime.runtime_profile import LAUNCHER_SYSTEMD_UNIT, Turbofile

ROOT = Path(__file__).resolve().parents[1]


def unit_profile(**overrides) -> Turbofile:
    mapping = valid_mapping()
    rung = {
        "id": "strata-32k",
        "context": 32768,
        "aux_mode": "shared-main",
        "evidence": "sha256:" + "4" * 64,
        "main_manifest": DIGEST_A,
        "launcher": LAUNCHER_SYSTEMD_UNIT,
        "unit": "turbofit-strata.service",
    }
    rung.update(overrides)
    mapping["rungs"] = [rung]
    return Turbofile.from_mapping(mapping)


def test_unit_rung_survives_a_mapping_round_trip() -> None:
    profile = unit_profile()

    item = profile.to_mapping()["rungs"][0]

    assert item["launcher"] == LAUNCHER_SYSTEMD_UNIT
    assert item["unit"] == "turbofit-strata.service"
    assert Turbofile.from_mapping(profile.to_mapping()).rungs[0].unit == "turbofit-strata.service"


def test_unit_without_launcher_is_rejected() -> None:
    mapping = valid_mapping()
    mapping["rungs"] = [
        {
            "id": "bare-unit",
            "context": 32768,
            "aux_mode": "shared-main",
            "evidence": "sha256:" + "5" * 64,
            "main_manifest": DIGEST_A,
            "unit": "turbofit-strata.service",
        }
    ]

    with pytest.raises(ValueError, match="unit requires launcher"):
        Turbofile.from_mapping(mapping)


def test_unknown_launcher_is_rejected() -> None:
    with pytest.raises(ValueError, match="launcher must equal"):
        unit_profile(launcher="docker")


def test_api_rung_cannot_declare_a_unit() -> None:
    mapping = valid_mapping()
    mapping["rungs"] = [
        {
            "id": "api",
            "context": 131072,
            "aux_mode": "api",
            "evidence": "sha256:" + "6" * 64,
            "main_api_policy": "api:auto",
            "aux_api_policy": "api:auto",
            "launcher": LAUNCHER_SYSTEMD_UNIT,
            "unit": "turbofit-strata.service",
        }
    ]

    with pytest.raises(ValueError, match="API rung cannot name a launcher unit"):
        Turbofile.from_mapping(mapping)


def test_missing_unit_artifacts_never_block_the_ladder() -> None:
    """A unit rung must not be pruned for a recipe path it does not use."""
    from turbofit_runtime.native_backend import rung_artifact_problem
    from turbofit_runtime.recipes import RecipeBook

    profile = unit_profile()
    resolutions = {
        profile.id: {
            "strata-32k": {
                "main": {"family": "Qwen 3.8 Flash Next (Strata)", "gpu": "0", "port": 8082,
                         "model_tag": "qwen3-8-flash-next-strata"}
            }
        }
    }

    problem = rung_artifact_problem(
        profile, "strata-32k", resolutions,
        RecipeBook.load(ROOT / "references/model-recipes.json"),
    )

    assert problem is None


def test_recommender_labels_a_foreign_device_index_as_host_spill() -> None:
    """Requirements recorded on another box must not invent a phantom card."""
    module = runpy.run_path(
        str(ROOT / "scripts/turbofit-runtime-recommend"),
        run_name="turbofit_runtime_recommend",
    )
    fit_requirements = module["fit_requirements"]
    from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint

    hardware = HardwareFingerprint(
        "linux", "x86_64", 65_536,
        devices=(AcceleratorDevice(0, "GPU-0", "RTX 4080 SUPER", "nvidia", "cuda", 16_376, "8.9", "01"),),
    )

    fit, reasons = fit_requirements(
        hardware=hardware,
        requirements={0: 563, 1: 19_391},
        available_budgets={0: 16_376},
        shared_memory=False,
        safety_floor_mb=1_024,
    )

    assert fit is True
    assert any("no device 1 here" in reason for reason in reasons)
    assert not any("device 1 resident" in reason for reason in reasons)
