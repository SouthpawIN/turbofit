"""Contract tests for the deterministic benchmark leaderboard.

The leaderboard joins two committed reference sources — the hardware tier
report and the engine/serve matrix — into one document per tier: the best
model (measured winner where physically validated, top catalog candidate
otherwise) plus the serve engines for that exact model. No step may invent
data: engine resolution is exact fit-list-slug match or an explicit
``native`` default, and the rendered page is wrapped in generated-content
markers so it can be replaced by tooling.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from turbofit_runtime.leaderboard import (
    GENERATED_BEGIN,
    GENERATED_END,
    LEADERBOARD_SCHEMA,
    build_leaderboard,
    load_engine_matrix,
    load_tier_report,
    publish_leaderboard,
    render_leaderboard_markdown,
    resolve_engines,
    select_winner,
)


ROOT = Path(__file__).parents[1]
REPORT = ROOT / "references" / "hardware-tier-report.json"
MATRIX = ROOT / "references" / "engine-serve-matrix.json"


def _loaded():
    return load_tier_report(REPORT), load_engine_matrix(MATRIX)


def test_committed_references_build_expected_leaderboard() -> None:
    report, matrix = _loaded()
    board = build_leaderboard(report, matrix)
    assert board["schema"] == LEADERBOARD_SCHEMA
    assert [t["capacity_gb"] for t in board["tiers"]] == [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    by_cap = {t["capacity_gb"]: t for t in board["tiers"]}

    # The Maple-aux tournaments have no hash-bound physical evidence yet, so no
    # tier may claim a measured winner; every tier falls back to a catalog
    # candidate rather than reusing a winner from a retired aux policy.
    t48 = by_cap[48]
    assert t48["status"] == "catalog-candidates-only"
    assert t48["winner"]["basis"] == "catalog-candidate"
    assert t48["winner"]["measured_tps"] is None
    assert t48["winner"]["evidence"] is None
    assert t48["engine_source"] == "default"
    assert "native" in t48["engines"]
    # The winner's own configuration must never appear as a runner-up.
    assert t48["winner"]["configuration_id"] not in t48["runners_up"]

    t8 = by_cap[8]
    assert t8["winner"]["basis"] == "catalog-candidate"
    assert t8["winner"]["measured_tps"] is None
    assert t8["engine_source"] == "default"
    assert t8["engines"] == ["native"]


def test_engines_resolve_by_exact_slug_only() -> None:
    _, matrix = _loaded()
    exact = resolve_engines("qwen3-8-27b-gsq-rco-iq3s", matrix)
    assert exact["engine_source"] == "matrix"
    assert exact["engines"]
    assert exact["artifact"]
    # A prefix of a real slug must NOT match — no fuzzy mapping.
    fuzzy = resolve_engines("qwen3-8-27b-gsq-rco-iq3", matrix)
    assert fuzzy["engine_source"] == "default"
    assert fuzzy["engines"] == ["native"]
    # The 8 GB tier model id does not match any Bonsai matrix slug.
    assert resolve_engines("bonsai-27b", matrix)["engine_source"] == "default"


def test_selection_prefers_measured_winner_when_validated() -> None:
    tier = {
        "id": "hardware-48gb",
        "capacity_gb": 48,
        "status": "physically-validated",
        "recommendations": {
            "measured_winner": {
                "configuration_id": "measured-cfg",
                "main": {"id": "m"},
                "context": 1,
            }
        },
        "candidates": [
            {"configuration_id": "cat-a", "main": {"id": "m"}, "intelligence_score": 99.0, "context": 999999},
            {"configuration_id": "measured-cfg", "main": {"id": "m"}, "context": 1},
        ],
    }
    winner, basis = select_winner(tier)
    assert basis == "measured"
    assert winner["configuration_id"] == "measured-cfg"
    # An unvalidated tier ignores the measured label and ranks the catalog.
    tier["status"] = "catalog"
    winner2, basis2 = select_winner(tier)
    assert basis2 == "catalog-candidate"
    assert winner2["configuration_id"] == "cat-a"


def test_catalog_ranking_is_deterministic() -> None:
    tier = {
        "id": "hardware-8gb",
        "capacity_gb": 8,
        "status": "catalog",
        "candidates": [
            {"configuration_id": "b", "main": {"id": "x"}, "context": 65536},
            {"configuration_id": "a", "main": {"id": "x"}, "context": 131072},
        ],
    }
    winner, _basis = select_winner(tier)
    assert winner["configuration_id"] == "a"
    # Context tie → ascending configuration_id.
    tier["candidates"][1]["context"] = 65536
    winner2, _ = select_winner(tier)
    assert winner2["configuration_id"] == "a"


def test_build_is_deterministic() -> None:
    report, matrix = _loaded()
    assert build_leaderboard(report, matrix) == build_leaderboard(report, matrix)


def test_markdown_has_generated_markers_and_all_tiers() -> None:
    report, matrix = _loaded()
    md = render_leaderboard_markdown(build_leaderboard(report, matrix))
    assert md.startswith(GENERATED_BEGIN)
    assert md.rstrip().endswith(GENERATED_END)
    for cap in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384):
        assert f"### {cap} GB" in md
    assert "## Method" in md
    assert "hardware-tier-report.json" in md
    assert "engine-serve-matrix.json" in md


def test_publish_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "benchmark-leaderboard.md"
    first = publish_leaderboard(REPORT, MATRIX, target)
    text1 = first.read_text(encoding="utf-8")
    second = publish_leaderboard(REPORT, MATRIX, target)
    assert second == target
    assert target.read_text(encoding="utf-8") == text1
    assert not list(tmp_path.glob("*.tmp"))


def test_leaderboard_endpoint_contract() -> None:
    pytest.importorskip("fastapi")
    spec = importlib.util.spec_from_file_location(
        "turbofit_leaderboard_plugin_api", ROOT / "dashboard" / "plugin_api.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    board = module._leaderboard_rows()
    assert board["schema"] == LEADERBOARD_SCHEMA
    report, matrix = _loaded()
    assert board == build_leaderboard(report, matrix)
