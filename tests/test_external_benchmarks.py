from __future__ import annotations

import json

import pytest

from research.discover_external_benchmarks import normalize_deepswe


def test_deepswe_normalization_keeps_provenance_and_best_configuration() -> None:
    payload = {
        "generated_at": "2026-08-04T01:34:32+00:00",
        "n_tasks_in_set": 113,
        "rows": [
            {
                "model": "glm-5-2", "harness": "mini-swe-agent", "reasoning_effort": "high",
                "config": "glm_high", "pass_at_1": 0.40, "pass_at_4": 0.60,
                "n_passed": 40, "n_attempted": 100, "mean_cost_usd": 2.0,
                "median_output_tokens": 1000, "median_peak_context_tokens": 90000,
            },
            {
                "model": "glm-5-2", "harness": "mini-swe-agent", "reasoning_effort": "max",
                "config": "glm_max", "pass_at_1": 0.44, "pass_at_4": 0.65,
                "n_passed": 44, "n_attempted": 100, "mean_cost_usd": 3.0,
                "median_output_tokens": 1200, "median_peak_context_tokens": 150000,
            },
        ],
    }

    result = normalize_deepswe(payload, "https://example.test/leaderboard.json", "sha256:" + "a" * 64)

    assert result["schema"] == "turbofit.external-benchmarks/v1"
    assert result["source"]["artifact_identity"] == "sha256:" + "a" * 64
    assert result["benchmarks"] == [{
        "benchmark": "deep-swe-v1.1",
        "model": "glm-5-2",
        "turbofit_model": "glm-5-2-2-788bpw",
        "harness": "mini-swe-agent",
        "reasoning_effort": "max",
        "configuration": "glm_max",
        "pass_at_1": 0.44,
        "pass_at_4": 0.65,
        "passed": 44,
        "attempted": 100,
        "mean_cost_usd": 3.0,
        "median_output_tokens": 1200,
        "median_peak_context_tokens": 150000,
    }]


def test_deepswe_normalization_rejects_inconsistent_score() -> None:
    payload = {
        "generated_at": "2026-08-04T01:34:32+00:00",
        "n_tasks_in_set": 113,
        "rows": [{
            "model": "bad", "harness": "mini-swe-agent", "reasoning_effort": None,
            "config": "bad", "pass_at_1": 0.9, "pass_at_4": 0.9,
            "n_passed": 1, "n_attempted": 10, "mean_cost_usd": 1.0,
            "median_output_tokens": 1, "median_peak_context_tokens": 1,
        }],
    }

    with pytest.raises(ValueError, match="inconsistent pass_at_1"):
        normalize_deepswe(payload, "https://example.test", "sha256:" + "b" * 64)
