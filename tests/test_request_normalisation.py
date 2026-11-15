"""Unit tests for the Turbofit request normalisation helper (TF3/TF4).

Precedence extracted from the validated turbohaul-nostream-relay repair:
explicit finite budget (clamped) > reasoning disabled (0) > effort mapping
> finite default. Only backends with a configured reasoning policy are
normalised; no global cap is applied to every model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbofit_runtime.request_normalisation import (  # noqa: E402
    apply_stream_usage_default,
    caller_controls_reasoning,
    normalise_reasoning_budget,
)


POLICY = {"hard_cap": 4096, "default_budget": 1024}


def test_explicit_budget_wins_and_is_clamped():
    payload = {"thinking_budget_tokens": 100000}
    out = normalise_reasoning_budget(payload, POLICY)
    assert out["thinking_budget_tokens"] == 4096


def test_reasoning_budget_spelling_accepted():
    out = normalise_reasoning_budget({"reasoning_budget": 777}, POLICY)
    assert out["thinking_budget_tokens"] == 777


def test_invalid_explicit_budget_falls_through_to_disabled():
    out = normalise_reasoning_budget(
        {"thinking_budget_tokens": -5, "think": False}, POLICY
    )
    assert out["thinking_budget_tokens"] == 0


def test_disabled_reasoning_maps_to_zero_budget():
    out = normalise_reasoning_budget({"think": False}, POLICY)
    assert out["thinking_budget_tokens"] == 0


def test_reasoning_enabled_false_maps_to_zero_budget():
    out = normalise_reasoning_budget(
        {"reasoning": {"enabled": False, "effort": "high"}}, POLICY
    )
    assert out["thinking_budget_tokens"] == 0


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("none", 0),
        ("minimal", 0),
        ("low", 512),
        ("medium", 1024),
        ("high", 2048),
        ("xhigh", 4096),
    ],
)
def test_effort_levels_map_to_finite_budgets(effort, expected):
    out = normalise_reasoning_budget({"reasoning": {"effort": effort}}, POLICY)
    assert out["thinking_budget_tokens"] == expected


def test_top_level_reasoning_effort_is_honoured():
    out = normalise_reasoning_budget({"reasoning_effort": "low"}, POLICY)
    assert out["thinking_budget_tokens"] == 512


def test_unknown_or_missing_effort_gets_finite_default():
    out = normalise_reasoning_budget({"reasoning": {"effort": "bananas"}}, POLICY)
    assert out["thinking_budget_tokens"] == 1024
    out = normalise_reasoning_budget({}, POLICY)
    assert out["thinking_budget_tokens"] == 1024


def test_no_policy_leaves_payload_untouched():
    payload = {"reasoning": {"effort": "high"}, "thinking_budget_tokens": 99999999}
    assert normalise_reasoning_budget(payload, None) == payload
    assert normalise_reasoning_budget(payload, {}) == payload


def test_normalisation_is_copy_not_mutation():
    payload = {"reasoning": {"effort": "low"}}
    out = normalise_reasoning_budget(payload, POLICY)
    assert out is not payload
    assert "thinking_budget_tokens" not in payload


def test_stream_default_adds_include_usage_only_for_streams():
    out = apply_stream_usage_default({"stream": True})
    assert out["stream_options"] == {"include_usage": True}


def test_stream_explicit_false_survives():
    payload = {"stream": True, "stream_options": {"include_usage": False}}
    out = apply_stream_usage_default(payload)
    assert out["stream_options"]["include_usage"] is False


def test_non_stream_unchanged():
    payload = {"stream": False, "messages": []}
    assert apply_stream_usage_default(payload) == payload
    assert "stream_options" not in payload


def test_existing_stream_options_preserved():
    payload = {"stream": True, "stream_options": {"foo": 1}}
    out = apply_stream_usage_default(payload)
    assert out["stream_options"] == {"foo": 1, "include_usage": True}


def test_caller_controls_reasoning_detection():
    assert caller_controls_reasoning({"think": True})
    assert caller_controls_reasoning({"thinking_budget_tokens": 512})
    assert caller_controls_reasoning({"reasoning_budget": 512})
    assert caller_controls_reasoning(
        {"chat_template_kwargs": {"enable_thinking": True}}
    )
    assert caller_controls_reasoning({"chat_template_kwargs": {"thinking_mode": "disabled"}})
    # Effort metadata alone is normalisable, not a hard caller control.
    assert not caller_controls_reasoning({"reasoning": {"effort": "low"}})
    assert not caller_controls_reasoning({"reasoning_effort": "high"})
    assert not caller_controls_reasoning({"messages": []})
    assert not caller_controls_reasoning({"chat_template_kwargs": {"foo": 1}})


def test_disabled_reasoning_in_gateway_normalises_to_zero():
    out = normalise_reasoning_budget(
        {"reasoning": {"enabled": False, "effort": "high"}}, POLICY
    )
    assert out["thinking_budget_tokens"] == 0
    assert out["reasoning"]["enabled"] is False