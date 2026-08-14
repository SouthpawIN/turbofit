from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.catalog_campaign import build_selected_campaign_matrix
from turbofit_runtime.schema import load_matrix
from turbofit_runtime.tier_tournament import candidate_ids, load_tournaments, validate_tournaments

ROOT = Path(__file__).parents[1]


def test_hardware_tournaments_cover_every_physical_tier_and_valid_configuration(tmp_path: Path) -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")
    selected = set(candidate_ids(tournaments))
    output = tmp_path / "tier-matrix.json"

    build_selected_campaign_matrix(configurations, catalog, output, selected)
    matrix = load_matrix(output)

    assert [item["vram_gb"] for item in tournaments["tiers"]] == [8, 16, 24, 48, 64, 96, 200, 300]
    assert len(matrix.rows) == len(selected)
    winners = {item["id"]: item["winner"] for item in tournaments["tiers"]}
    assert all(winner is None for winner in winners.values())


def test_64_and_96gb_qwen_candidates_are_active_but_unpromoted() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    policy = json.loads((ROOT / "references/catalog-campaign-policy.json").read_text())
    deferred = set(policy["deferred_models"])
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram in (64, 96):
        candidates = by_tier[vram]["candidates"]
        assert all(item.startswith("qwen3-8-27b-") for item in candidates)
        assert all(item.split("--", 1)[0] not in deferred for item in candidates)
        assert by_tier[vram]["winner"] is None


def test_200_and_300gb_tiers_include_deepseek() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations)
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram in (200, 300):
        assert any(item.startswith("deepseek-v4-flash-0731") for item in by_tier[vram]["candidates"])


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
