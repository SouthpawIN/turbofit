from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.catalog_campaign import build_selected_campaign_matrix
from turbofit_runtime.schema import load_matrix
from turbofit_runtime.tier_fit import AUXILIARY_ID, TIERS, load_footprints, plan_candidates, select_candidates
from turbofit_runtime.quant_policy import classify_entry
from turbofit_runtime.tier_tournament import (
    candidate_ids,
    candidate_ids_for_tier,
    load_tournaments,
    validate_tournaments,
)

ROOT = Path(__file__).parents[1]


def _catalog() -> dict:
    return json.loads((ROOT / "references/model-catalog.json").read_text())


def test_hardware_tournaments_cover_every_physical_tier_and_valid_configuration(tmp_path: Path) -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")
    tournaments = load_tournaments(
        ROOT / "references/hardware-tier-tournaments.json", configurations, catalog,
    )
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
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations, catalog)
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram_gb in TIERS:
        tier = by_tier[vram_gb]
        assert tier["candidates"] == select_candidates(catalog, configurations, footprints, vram_gb)
        assert len({item.split("--", 1)[0] for item in tier["candidates"]}) >= 4
        assert all(item.split("--")[1] == AUXILIARY_ID for item in tier["candidates"])
        assert all(item.rsplit("--", 1)[1] in {"64k", "128k", "262k", "1m"} for item in tier["candidates"])
        assert tier["winner"] is None or tier["winner"]["configuration"] in tier["candidates"]


def test_no_q5_or_q6_candidate_is_listed_by_default() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = _catalog()
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations, catalog)
    models = {item["id"]: item for item in catalog["models"]}
    footprints = load_footprints(ROOT)
    by_tier = {item["vram_gb"]: item for item in tournaments["tiers"]}

    for vram_gb, tier in by_tier.items():
        for identifier in tier["candidates"]:
            main = identifier.split("--", 1)[0]
            # classify_entry reads the first quant token, so a mixed build such
            # as Maya-M (IQ2_S routed / Q6_K attention) is classified by its
            # routing quant and stays eligible.
            assert not classify_entry(models[main]).is_excluded, f"{main} at {vram_gb} GB"
            assert "Q5" not in main and "Q6" not in main, f"{main} at {vram_gb} GB"
        # Every held-out family is listed with a reason rather than vanishing.
        for record in tier["quant_excluded"]:
            assert record["reason"]
            assert record["main"] not in {item.split("--", 1)[0] for item in tier["candidates"]}
        # Every Q8/full-precision candidate carries a visible reason.
        for record in tier["quant_exceptions"]:
            assert record["reason"]
            assert record["quant_class"] == "allowed-exception"

    plan = plan_candidates(catalog, configurations, footprints, 384)
    assert "darwin-180b-rsi-strata" in {item["main"] for item in plan.excluded}


def test_q5_entry_with_a_recorded_justification_is_allowed() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = _catalog()
    footprints = load_footprints(ROOT)
    footprints = {
        **footprints,
        "models": {
            **footprints["models"],
            "arete-27b-q5": {**footprints["models"]["arete-27b-q5"], "weight_bytes": 6 * 1024 ** 3},
        },
    }
    allowed = plan_candidates(catalog, configurations, footprints, 384)
    assert "arete-27b-q5" in {item["main"] for item in allowed.excluded}

    justified = json.loads(json.dumps(catalog))
    for model in justified["models"]:
        if model["id"] == "arete-27b-q5":
            model["quantization_justification"] = "Q5_K_M is the only published Arete 27B build with MTP"
    plan = plan_candidates(justified, configurations, footprints, 384)
    assert any(item.startswith("arete-27b-q5--") for item in plan.candidates)
    assert "arete-27b-q5" not in {item["main"] for item in plan.excluded}


def test_candidate_ids_for_tier_preserves_tournament_priority() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(
        ROOT / "references/hardware-tier-tournaments.json", configurations, _catalog(),
    )
    expected = next(item["candidates"] for item in tournaments["tiers"] if item["vram_gb"] == 48)

    assert candidate_ids_for_tier(tournaments, 48) == tuple(expected)


def test_48gb_tier_includes_qwen_38_dflash2_candidate() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(
        ROOT / "references/hardware-tier-tournaments.json", configurations, _catalog(),
    )

    assert f"qwen3-8-27b-q4-dflash2--{AUXILIARY_ID}--64k" in candidate_ids_for_tier(tournaments, 48)


def test_64_and_96gb_strata_candidates_are_promoted() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    tournaments = load_tournaments(
        ROOT / "references/hardware-tier-tournaments.json", configurations, _catalog(),
    )
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
    tournaments = load_tournaments(ROOT / "references/hardware-tier-tournaments.json", configurations, catalog)
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
        validate_tournaments(payload, configurations, _catalog())


def test_a_q5_candidate_without_a_justification_is_rejected_by_validation() -> None:
    configurations = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    catalog = _catalog()
    payload = json.loads((ROOT / "references/hardware-tier-tournaments.json").read_text())
    tier = payload["tiers"][-1]
    q5 = next(
        row["id"] for row in configurations["rows"] if row["main"] == "arete-27b-q5"
    )
    tier["candidates"].append(q5)

    with pytest.raises(ValueError, match="still a candidate"):
        validate_tournaments(payload, configurations, catalog)

    # With the justification recorded on the catalog entry, the same row is a
    # legitimate candidate and validation accepts it.
    justified = json.loads(json.dumps(catalog))
    for model in justified["models"]:
        if model["id"] == "arete-27b-q5":
            model["quantization_justification"] = "only published Arete 27B build with MTP"
    tier["quant_excluded"] = [
        record for record in tier["quant_excluded"] if record["main"] != "arete-27b-q5"
    ]
    tier["quant_exceptions"].append({
        "main": "arete-27b-q5",
        "quantization": "Q5_K_M",
        "quant_class": "excluded-by-default",
        "quant_token": "Q5_K_M",
        "reason": "justified: only published Arete 27B build with MTP",
    })
    validate_tournaments(payload, configurations, justified)

    # A Q5 candidate in the audit list without a recorded justification is refused.
    tier["quant_exceptions"][-1]["reason"] = "it was faster in a screenshot"
    with pytest.raises(ValueError, match="recorded justification"):
        validate_tournaments(payload, configurations, justified)
