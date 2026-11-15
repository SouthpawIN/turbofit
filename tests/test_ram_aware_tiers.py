"""Placement is two-dimensional: a VRAM band *and* a system RAM budget.

These are pure-logic tests against a synthetic catalog, plus checks that the
committed artefacts agree with the same arithmetic. No GPU, network or model
files are involved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.tier_fit import (
    AUXILIARY_ID,
    DEFAULT_RAM_GIB,
    GIB,
    RAM_GIB_ENV,
    TIERS,
    build_footprints,
    default_ram_gib,
    load_footprints,
    placement_for,
    ram_budget_bytes,
    select_offload_candidates,
    select_placements,
    unmeasured_moe_mains,
)
from turbofit_runtime.tier_tournament import (
    offload_candidate_ids,
    offload_candidates_for_tier,
    validate_tournaments,
)

ROOT = Path(__file__).parents[1]

MODELS = [
    {"id": "dense-fit", "name": "Dense Fit", "moe": False, "roles": ["main"],
     "quantization": "Q4_K_M", "runtime_features": [], "supported_contexts": [65_536]},
    {"id": "moe-offload", "name": "MoE Offload", "moe": True, "roles": ["main"],
     "quantization": "IQ1_M", "runtime_features": [], "supported_contexts": [65_536]},
    {"id": "moe-huge", "name": "MoE Huger Than RAM", "moe": True, "roles": ["main"],
     "quantization": "Q4_K_M", "runtime_features": [], "supported_contexts": [65_536]},
    {"id": "moe-unmeasured", "name": "MoE Unmeasured", "moe": True, "roles": ["main"],
     "quantization": "Q4_K_M", "runtime_features": [], "supported_contexts": [65_536]},
    {"id": "dense-huge", "name": "Dense Huge", "moe": False, "roles": ["main"],
     "quantization": "Q4_K_M", "runtime_features": [], "supported_contexts": [65_536]},
]

WEIGHTS = {
    "dense-fit": 20 * GIB,
    # 100 GiB total, 90 GiB of it experts over 30 layers: ~10 GiB resident.
    "moe-offload": 100 * GIB,
    # 300 GiB total, 280 GiB experts over 40 layers: cannot fit VRAM + 128 GiB.
    "moe-huge": 300 * GIB,
    "moe-unmeasured": 100 * GIB,
    "dense-huge": 300 * GIB,
}

EXPERTS = {
    "moe-offload": (90 * GIB, 30),
    "moe-huge": (280 * GIB, 40),
}


def _catalog() -> dict:
    return {"models": [dict(model) for model in MODELS]}


def _footprints() -> dict:
    models = {}
    for model in MODELS:
        record = {"weight_bytes": WEIGHTS[model["id"]], "moe": model["moe"]}
        if model["id"] in EXPERTS:
            record["expert_bytes"], record["expert_layers"] = EXPERTS[model["id"]]
        models[model["id"]] = record
    return {"schema": "turbofit.model-footprints/v1", "models": models}


def _configurations() -> dict:
    return {"rows": [
        {"id": f"{model['id']}--{AUXILIARY_ID}--64k", "main": model["id"],
         "auxiliary": AUXILIARY_ID, "context": 65_536}
        for model in MODELS
    ]}


def test_model_over_vram_but_inside_vram_plus_ram_is_an_offload_option() -> None:
    placements = select_placements(
        _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(128),
    )
    offload = [item for item in placements if item.placement == "offload"]

    target = next(item for item in offload if item.main == "moe-offload")
    assert target.configuration_id == f"moe-offload--{AUXILIARY_ID}--64k"
    assert target.evidence == "measured-expert-footprint"
    # It states where the bytes live instead of just claiming it fits.
    assert target.ram_bytes > 0
    assert target.vram_bytes > 0
    assert target.ram_gib + target.vram_gib >= 100
    assert target.vram_bytes <= 48 * GIB
    assert 1 <= target.n_cpu_moe <= target.expert_layers


def test_offload_options_are_never_mixed_into_the_full_gpu_list() -> None:
    placements = select_placements(
        _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(128),
    )
    gpu = {item.main for item in placements if item.placement == "gpu"}
    offload = {item.main for item in placements if item.placement == "offload"}

    assert "moe-offload" in offload
    assert "moe-offload" not in gpu
    assert gpu.isdisjoint(offload)

    # The API a full-GPU caller uses stays full-GPU only.
    from turbofit_runtime.tier_fit import select_candidates
    assert "moe-offload" not in {
        item.split("--", 1)[0]
        for item in select_candidates(_catalog(), _configurations(), _footprints(), 48)
    }


def test_model_over_vram_and_ram_is_not_offered() -> None:
    for vram_gb in (16, 48, 96):
        placements = {
            item.main for item in select_placements(
                _catalog(), _configurations(), _footprints(), vram_gb,
                ram_budget_bytes(128),
            )
        }
        assert "moe-huge" not in placements
        assert "dense-huge" not in placements


def test_model_that_fits_the_band_is_full_gpu_with_no_ram() -> None:
    placements = select_placements(
        _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(128),
    )
    dense = next(item for item in placements if item.main == "dense-fit")
    assert dense.placement == "gpu"
    assert dense.ram_bytes == 0 and dense.n_cpu_moe == 0

    # The same MoE model becomes full-GPU once the band is big enough.
    big = next(
        item for item in select_placements(
            _catalog(), _configurations(), _footprints(), 128, ram_budget_bytes(128),
        ) if item.main == "moe-offload"
    )
    assert big.placement == "gpu" and big.ram_bytes == 0


def test_unmeasured_moe_is_not_advertised_as_offloadable() -> None:
    placements = {
        item.main for item in select_placements(
            _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(128),
        )
    }
    assert "moe-unmeasured" not in placements
    assert unmeasured_moe_mains(_catalog(), _footprints()) == ["moe-unmeasured"]


def test_ram_default_is_documented_and_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEFAULT_RAM_GIB == 128
    assert default_ram_gib({}) == 128.0
    assert default_ram_gib({RAM_GIB_ENV: "377"}) == 377.0
    with pytest.raises(ValueError):
        default_ram_gib({RAM_GIB_ENV: "lots"})
    with pytest.raises(ValueError):
        default_ram_gib({RAM_GIB_ENV: "0"})

    monkeypatch.setenv(RAM_GIB_ENV, "377")
    assert default_ram_gib() == 377.0

    # The budget is actually used: 57 GiB of experts cannot fit a 30 GiB host.
    tight = select_placements(
        _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(30),
    )
    assert "moe-offload" not in {item.main for item in tight}
    assert "moe-offload" in {
        item.main for item in select_placements(
            _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(128),
        )
    }


def test_placement_is_a_pure_function_of_budget_and_context() -> None:
    record = _footprints()["models"]["moe-offload"]
    first = placement_for("moe-offload", record, 65_536, 48, ram_budget_bytes(128))
    second = placement_for("moe-offload", record, 65_536, 48, ram_budget_bytes(128))
    assert first == second
    # A context below the 64K floor is never placeable.
    assert placement_for("moe-offload", record, 32_768, 384, ram_budget_bytes(377)) is None


def test_regeneration_is_idempotent() -> None:
    catalog, configurations = _catalog(), _configurations()
    first = [item.as_dict() for item in select_placements(
        catalog, configurations, _footprints(), 48, ram_budget_bytes(128),
    )]
    second = [item.as_dict() for item in select_placements(
        catalog, configurations, _footprints(), 48, ram_budget_bytes(128),
    )]
    assert first == second

    manifest = {"artifacts": [
        {"repo_id": "org/dense", "path": "dense.gguf", "size_bytes": WEIGHTS["dense-fit"],
         "families": ["dense-fit"]},
    ]}
    build_catalog = {"models": [MODELS[0]]}
    assert build_footprints(build_catalog, manifest, generated_at="t") == build_footprints(
        build_catalog, manifest, generated_at="t",
    )


def test_committed_tournaments_match_the_stated_ram_assumption() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = json.loads((ROOT / "references/model-catalog.json").read_text())
    footprints = load_footprints(ROOT)
    tournaments = json.loads((ROOT / "references/hardware-tier-tournaments.json").read_text())

    validate_tournaments(tournaments, configurations)
    assumption = tournaments["ram_assumption"]
    assert assumption["override"]
    assert assumption["source"]
    ram_bytes = ram_budget_bytes(assumption["gib"])

    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}
    assert tuple(by_tier) == TIERS
    for vram_gb in TIERS:
        placements = select_placements(catalog, configurations, footprints, vram_gb, ram_bytes)
        assert by_tier[vram_gb]["candidates"] == [
            item.configuration_id for item in placements if item.placement == "gpu"
        ]
        assert [
            entry["configuration_id"] for entry in by_tier[vram_gb]["offload_candidates"]
        ] == [item.configuration_id for item in placements if item.placement == "offload"]
        assert not set(by_tier[vram_gb]["candidates"]) & set(offload_candidate_ids(
            {"tiers": [by_tier[vram_gb]]}
        ))


def test_committed_artefacts_offer_the_oversized_families_as_offload() -> None:
    tournaments = json.loads((ROOT / "references/hardware-tier-tournaments.json").read_text())
    report = json.loads((ROOT / "references/hardware-tier-report.json").read_text())

    # 48 GB is the user's case: big MoE families are placeable via system RAM.
    offload = offload_candidates_for_tier(tournaments, 48)
    families = {entry["configuration_id"].split("--", 1)[0] for entry in offload}
    for family in ("deepseek-v4-1-flash-1017-iq1m", "glm-5-3-flash-maya-xs",
                   "qwen3-8-flash-next-strata"):
        assert family in families
    assert all(float(entry["ram_gib"]) > 0 for entry in offload)
    assert all(int(entry["n_cpu_moe"]) >= 1 for entry in offload)

    report_tier = next(item for item in report["tiers"] if item["capacity_gb"] == 48)
    assert report["ram_budget"]["gib"] == tournaments["ram_assumption"]["gib"]
    assert {row["configuration_id"] for row in report_tier["offload_candidates"]} == {
        entry["configuration_id"] for entry in offload
    }
    assert all(row["fit"]["placement"] == "gpu" for row in report_tier["candidates"])


def test_validation_rejects_a_model_in_both_placements() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    payload = json.loads((ROOT / "references/hardware-tier-tournaments.json").read_text())
    tier = payload["tiers"][3]
    duplicated = dict(tier["offload_candidates"][0])
    duplicated["configuration_id"] = tier["candidates"][0]

    tier["offload_candidates"] = tier["offload_candidates"] + [duplicated]
    with pytest.raises(ValueError, match="both full-GPU and offloaded"):
        validate_tournaments(payload, configurations)


def test_offload_selection_api_filters_on_placement() -> None:
    entries = select_offload_candidates(
        _catalog(), _configurations(), _footprints(), 48, ram_budget_bytes(128),
    )
    assert [entry["placement"] for entry in entries] == ["offload"] * len(entries)
    assert all(entry["ram_gib"] > 0 for entry in entries)
