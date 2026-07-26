from __future__ import annotations

import runpy
from pathlib import Path


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/turbofit-runtime-recommend"),
    run_name="turbofit_runtime_recommend",
)
priority_key = MODULE["priority_key"]


def test_recommendation_ladder_prioritizes_128k_before_raw_speed() -> None:
    assert priority_key(131_072, 5.0, 20) > priority_key(65_536, 200.0, 20)


def test_recommendation_ladder_prioritizes_speed_until_30_tps() -> None:
    assert priority_key(131_072, 29.0, 20) > priority_key(262_144, 6.0, 20)


def test_recommendation_ladder_prioritizes_262k_after_30_tps() -> None:
    assert priority_key(262_144, 30.0, 20) > priority_key(131_072, 100.0, 20)


def test_recommendation_ladder_prioritizes_speed_until_100_tps_at_262k() -> None:
    assert priority_key(262_144, 99.0, 20) > priority_key(1_048_576, 31.0, 20)


def test_recommendation_ladder_prioritizes_1m_after_100_tps() -> None:
    assert priority_key(1_048_576, 100.0, 20) > priority_key(262_144, 100.0, 20)


def test_quality_tier_precedes_context_speed_ladder() -> None:
    assert priority_key(262_144, 60.0, 20) > priority_key(1_048_576, 100.0, 10)
