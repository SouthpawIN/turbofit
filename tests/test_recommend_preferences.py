from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/turbofit-runtime-recommend"),
    run_name="turbofit_runtime_recommend",
)
lineup = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "src/turbofit_runtime/allowed_lineup.py"),
    run_name="allowed_lineup_under_test",
)
recommendation_sort_key = MODULE["recommendation_sort_key"]
rank_profiles = MODULE["rank_profiles"]
Recommendation = MODULE["Recommendation"]


def _rec(name: str, *, tps: float, quality: int, context: int, main: str = "maple-preview-tq2") -> Recommendation:
    return Recommendation(
        profile=name, score=0.0, quality_points=quality, context=context,
        main=main, aux="auto", aux_mode="shared-main", min_tps=tps,
        vision=False, methods=["baseline"], gpu_requirements_mb={}, fit=True,
        fit_reason="fit", confidence="measured", command="run", evidence="evidence",
    )


def test_speed_prefers_measured_faster_dense_over_unmeasured_moe() -> None:
    """Evidence-first: measured 80 t/s dense beats an unmeasured MoE prior."""
    dense = _rec("dense-measured", tps=80.0, quality=20, context=131_072, main="qwen3-8-27b-unleashed")
    moe = _rec("moe-unmeasured", tps=0.0, quality=20, context=131_072, main="ornith-1-5-35a3b")
    speeds = {"qwen3-8-27b-unleashed": {"tps": 80.0}}

    assert recommendation_sort_key(dense, "speed", speeds) > recommendation_sort_key(moe, "speed", speeds)


def test_speed_still_ranks_measured_moe_over_slower_measured_dense() -> None:
    """Benchmarks decide both directions: a faster MoE still wins the fast lane."""
    dense = _rec("dense-slow", tps=12.0, quality=50, context=262_144, main="qwen3-8-27b-unleashed")
    moe = _rec("moe-fast", tps=55.0, quality=20, context=131_072, main="ornith-1-5-35a3b")
    speeds = {
        "qwen3-8-27b-unleashed": {"tps": 12.0},
        "ornith-1-5-35a3b": {"tps": 55.0},
    }

    assert recommendation_sort_key(moe, "speed", speeds) > recommendation_sort_key(dense, "speed", speeds)


def test_speed_without_evidence_falls_back_to_family_prior() -> None:
    """No measurements anywhere → documented family prior keeps old behavior."""
    dense = _rec("dense", tps=0.0, quality=20, context=131_072, main="qwen3-8-27b-unleashed")
    moe = _rec("moe", tps=0.0, quality=20, context=131_072, main="ornith-1-5-35a3b")

    assert recommendation_sort_key(moe, "speed", None) > recommendation_sort_key(dense, "speed", None)


def test_intelligence_preference_is_never_flipped_by_speed_evidence() -> None:
    """The intelligence lane must stay quality-ranked regardless of TPS evidence."""
    smart_slow = _rec("smart", tps=5.0, quality=90, context=262_144)
    fast_dumb = _rec("fast", tps=120.0, quality=20, context=131_072)
    speeds = {
        "qwen3-8-27b-unleashed": {"tps": 120.0},
        "qwen3-8-27b-unleashed-ud-q3-k-xl": {"tps": 5.0},
    }

    key_smart = recommendation_sort_key(smart_slow, "intelligence", speeds)
    key_fast = recommendation_sort_key(fast_dumb, "intelligence", speeds)
    assert key_smart > key_fast


def test_speed_rank_value_reads_measured_alias_tps(tmp_path: Path) -> None:
    path = tmp_path / "speed-rankings.json"
    path.write_text(json.dumps({
        "schema": "turbofit.speed-rankings/v1",
        "records": [
            {"alias": "ornith-1-5-35a3b", "tps": 61.2, "evidence": True},
            {"alias": "stale-alias", "tps": 999.0, "evidence": False},
            {"alias": "bad-type", "tps": "fast"},
        ],
    }))

    load_measured_speeds = MODULE["load_measured_speeds"]
    speeds = load_measured_speeds(path)

    assert speeds == {"ornith-1-5-35a3b": {"tps": 61.2}}
