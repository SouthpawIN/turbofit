from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from turbofit_runtime.intelligence import (
    BenchmarkMeasurement,
    ConfigurationIntelligence,
    DEEPSWE_RUNNER_PROTOCOL,
    INTELLIGENCE_RECIPE_PROTOCOL,
    canonical_intelligence_recipe,
    intelligence_score,
    rank_configurations,
    refresh_score_payload,
    validate_measurement,
)
from turbofit_runtime.agentic_pair_benchmark import AgenticSuite, summarize_cases


ROOT = Path(__file__).parents[1]
SHA = "sha256:" + "a" * 64


def measurement(name: str, score: float) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        name=name,
        revision="deadbeef",
        score=score,
        tasks_total=100,
        tasks_passed=round(score * 100),
        raw_result="references/results/raw.json",
        raw_result_sha256=SHA,
    )


def test_intelligence_recipe_hash_binds_benchmark_protocols() -> None:
    component = SimpleNamespace(
        role="main", family="llama.cpp", alias="main", method="native",
        gpu="auto", port=11605, command=("llama-server",), model_path="model.gguf",
        projector_path=None,
    )
    recipe = SimpleNamespace(
        profile_name="test", main_alias="main", aux_alias="main",
        aux_mode="shared-main", components=(component,),
    )
    digest, payload = canonical_intelligence_recipe(
        {"id": "main--auto--64k", "context": 65536}, recipe,
    )
    assert digest.startswith("sha256:")
    assert payload["intelligence_recipe_protocol"] == INTELLIGENCE_RECIPE_PROTOCOL
    assert payload["benchmark_protocols"]["deep-swe"] == DEEPSWE_RUNNER_PROTOCOL


def configuration(identifier: str, *, deep: float, agent: float, tps: float) -> ConfigurationIntelligence:
    return ConfigurationIntelligence(
        configuration_id=identifier,
        hardware_tier_gb=48,
        main="main-q4",
        auxiliary="aux-q4",
        context=131_072,
        quantizations=("Q4_K_M", "Q4_K_M"),
        production_recipe_sha256=SHA,
        throughput_tps=tps,
        measurements=(measurement("deep-swe", deep), measurement("agentic-pair", agent)),
    )


def test_intelligence_score_is_equal_weight_arithmetic_mean_of_real_suites() -> None:
    item = configuration("pair-a", deep=0.25, agent=0.81, tps=20)

    assert intelligence_score(item) == pytest.approx(53.0)
    assert item.coverage == "complete"


def test_one_real_zero_suite_does_not_erase_other_measured_capability() -> None:
    item = configuration("pair-zero", deep=0.0, agent=0.81, tps=20)

    assert intelligence_score(item) == pytest.approx(40.5)


def test_refresh_score_payload_rebuilds_stale_zero_composites_from_raw_suite_counts() -> None:
    item = configuration("stale-zero", deep=0.0, agent=0.81, tps=20)
    payload = {
        **item.__dict__,
        "measurements": [measurement.__dict__ for measurement in item.measurements],
        "intelligence_score": 0.0,
        "balanced_score": 0.0,
        "benchmark_level": "screening",
    }

    refreshed = refresh_score_payload(payload)

    assert refreshed["intelligence_score"] == 40.5
    assert refreshed["balanced_score"] > 0
    assert refreshed["benchmark_level"] == "screening"


def test_intelligence_score_refuses_missing_required_suite() -> None:
    item = ConfigurationIntelligence(
        configuration_id="pair-a",
        hardware_tier_gb=48,
        main="main",
        auxiliary="auto",
        context=65_536,
        quantizations=("Q4",),
        production_recipe_sha256=SHA,
        throughput_tps=10,
        measurements=(measurement("deep-swe", 0.5),),
    )

    with pytest.raises(ValueError, match="missing required benchmark"):
        intelligence_score(item)


def test_measurement_requires_hash_bound_raw_result_and_consistent_counts() -> None:
    invalid = measurement("deep-swe", 0.5)
    invalid = BenchmarkMeasurement(**{**invalid.__dict__, "tasks_passed": 6})
    with pytest.raises(ValueError, match="score does not match"):
        validate_measurement(invalid)


def test_rankings_expose_smart_fast_and_balanced_choices_without_hardcoded_model_tiers() -> None:
    smart = configuration("smart", deep=0.81, agent=0.81, tps=10)
    fast = configuration("fast", deep=0.36, agent=0.49, tps=80)

    ranked = rank_configurations((smart, fast))

    assert ranked["intelligence"][0].configuration_id == "smart"
    assert ranked["speed"][0].configuration_id == "fast"
    assert {item.configuration_id for item in ranked["balanced"]} == {"smart", "fast"}


def test_agentic_pair_manifest_is_pinned_and_has_real_joint_main_aux_cases() -> None:
    path = ROOT / "references/benchmarks/agentic-pair-v1.json"
    suite = AgenticSuite.load(path)

    assert suite.revision == "agentic-pair-v1"
    assert len(suite.cases) >= 8
    assert all(case.tools and case.expected_tool for case in suite.cases)
    assert all(case.tool_result and case.final_validator for case in suite.cases)
    assert suite.identity == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_summary_scores_both_aux_tool_selection_and_main_synthesis() -> None:
    cases = [
        {"aux_passed": True, "main_passed": True},
        {"aux_passed": True, "main_passed": False},
    ]

    assert summarize_cases(cases) == {
        "tasks_total": 2,
        "tasks_passed": 1,
        "aux_tool_accuracy": 1.0,
        "main_synthesis_accuracy": 0.5,
        "score": 0.75,
    }
