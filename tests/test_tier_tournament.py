from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.catalog_campaign import build_selected_campaign_matrix
from turbofit_runtime.schema import load_matrix
from turbofit_runtime.tier_fit import AUXILIARY_ID, TIERS, load_footprints, select_candidates
from turbofit_runtime.tier_tournament import candidate_ids, candidate_ids_for_tier, load_tournaments, validate_tournaments

ROOT = Path(__file__).parents[1]


def test_hardware_tournaments_cover_every_physical_tier_and_valid_configuration(tmp_path: Path) -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")
    selected = set(candidate_ids(tournaments))
    output = tmp_path / "tier-matrix.json"

    build_selected_campaign_matrix(configurations, catalog, output, selected)
    matrix = load_matrix(output)

    assert [item["vram_gb"] for item in tournaments["tiers"]] == list(TIERS)
    assert len(matrix.rows) == len(selected)


def test_every_tier_lists_every_fitting_main_against_maple(tmp_path: Path) -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = json.loads((ROOT / "references/model-catalog.json").read_text())
    footprints = load_footprints(ROOT)
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram_gb in TIERS:
        tier = by_tier[vram_gb]
        assert tier["candidates"] == select_candidates(catalog, configurations, footprints, vram_gb)
        assert len({item.split("--", 1)[0] for item in tier["candidates"]}) >= 4
        assert all(item.split("--")[1] == AUXILIARY_ID for item in tier["candidates"])
        assert all(item.rsplit("--", 1)[1] in {"64k", "128k", "262k", "1m"} for item in tier["candidates"])
        assert tier["winner"] is None or tier["winner"]["configuration"] in tier["candidates"]


def test_candidate_ids_for_tier_preserves_tournament_priority() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    expected = next(item["candidates"] for item in tournaments["tiers"] if item["vram_gb"] == 48)

    assert candidate_ids_for_tier(tournaments, 48) == tuple(expected)


def test_48gb_tier_includes_qwen_38_dflash2_candidate() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)

    assert f"qwen3-8-27b-q4-dflash2--{AUXILIARY_ID}--64k" in candidate_ids_for_tier(tournaments, 48)


def test_64_and_96gb_strata_candidates_are_promoted() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    policy = json.loads((ROOT / "references/catalog-campaign-policy.json").read_text())
    deferred = set(policy["deferred_models"])
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram in (64, 96):
        candidates = by_tier[vram]["candidates"]
        assert any("strata" in item for item in candidates)
        assert any(item.startswith("qwen3-8-") for item in candidates)
        assert all(item.split("--", 1)[0] not in deferred for item in candidates)


def test_192gb_and_above_keep_qwen_candidates_and_fit_is_honest() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = json.loads((ROOT / "references/model-catalog.json").read_text())
    footprints = load_footprints(ROOT)
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram in (192, 256, 384):
        assert any("qwen3-8-" in item for item in by_tier[vram]["candidates"])

    # A large IQ1_M checkpoint (112 GiB) cannot be ranked below the 128 GB band.
    for vram in (8, 12, 16, 24, 32, 48, 64, 96):
        assert all("deepseek" not in item for item in by_tier[vram]["candidates"])
    assert any("deepseek" in item for item in by_tier[128]["candidates"])

    # Every listed main only ever appears in a band its weights + 64K KV fit.
    weights = {key: value["weight_bytes"] for key, value in footprints["models"].items()}
    for vram, tier in by_tier.items():
        for identifier in tier["candidates"]:
            assert identifier.split("--", 1)[0] in weights
            assert identifier in select_candidates(catalog, configurations, footprints, vram)


def test_tier_winner_requires_hash_bound_physical_evidence() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    payload = json.loads((ROOT / "references/hardware-tier-tournaments.json").read_text())
    payload["tiers"][0]["winner"] = {
        "configuration": payload["tiers"][0]["candidates"][0],
        "evidence": "not-a-hash",
        "hardware_fingerprint": "host-a",
    }

    with pytest.raises(ValueError, match="sha256-bound"):
        validate_tournaments(payload, configurations)
