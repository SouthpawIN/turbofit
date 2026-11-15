"""Quantization policy: a Q4 ceiling, low quants first-class, Q8/full-precision exceptions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from turbofit_runtime.quant_policy import (
    ALLOWED,
    ALLOWED_EXCEPTION,
    EXCLUDED_BY_DEFAULT,
    classify,
    default_admits,
    policy_table,
)

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Q4_K_M", ALLOWED),
        ("UD-Q4_K_XL", ALLOWED),
        ("UD-Q4_K_S", ALLOWED),
        ("Q4", ALLOWED),
        ("Q3_K_S", ALLOWED),
        ("IQ2_XXS", ALLOWED),
        ("UD-IQ3_XXS", ALLOWED),
        ("GSQ-RCO-IQ3_S", ALLOWED),
        ("IQ4_XS", ALLOWED),
        ("TQ2_0", ALLOWED),
        ("PTQ1_0", ALLOWED),
        ("2.788 bpw", ALLOWED),
        ("native-24GB", ALLOWED),
        ("Q5_K_M", EXCLUDED_BY_DEFAULT),
        ("UD-Q5_K_XL", EXCLUDED_BY_DEFAULT),
        ("Q6_K", EXCLUDED_BY_DEFAULT),
        ("Q8_0", ALLOWED_EXCEPTION),
        ("F16", ALLOWED_EXCEPTION),
        ("BF16", ALLOWED_EXCEPTION),
        ("bf16", ALLOWED_EXCEPTION),
        ("FP16", ALLOWED_EXCEPTION),
        ("FP16/BF16", ALLOWED_EXCEPTION),
    ],
)
def test_classification_table(label: str, expected: str) -> None:
    assert classify(label).quant_class == expected


def test_low_quants_are_not_marked_degraded() -> None:
    """A Q3/Q2 build is as allowed as a Q4 build: the ceiling is not a target."""
    for label in ("Q2_0", "IQ3_XXS", "Q3_K_S", "Q4_K_M"):
        verdict = classify(label)
        assert verdict.is_allowed
        assert verdict.is_excluded is False
        assert verdict.is_exception is False
        assert "degraded" not in verdict.reason.lower()
        assert "fallback" not in verdict.reason.lower()


def test_mixed_build_is_classified_by_its_routing_quant() -> None:
    verdict = classify("Maya-M (IQ2_S routed / Q6_K attention)")
    assert verdict.is_allowed and verdict.token == "IQ2_S"


def test_excluded_entry_needs_a_recorded_justification() -> None:
    entry = {"id": "arete-27b-q5", "quantization": "Q5_K_M"}
    admitted, reason = default_admits(entry)
    assert admitted is False
    assert "quantization_justification required" in reason

    justified = {**entry, "quantization_justification": "only build with the long-context patch"}
    admitted, reason = default_admits(justified)
    assert admitted is True
    assert reason == "justified: only build with the long-context patch"


def test_an_alias_cannot_smuggle_a_q5_past_the_ceiling() -> None:
    entry = {
        "id": "cyber-tiel-coder-35b-a3b-ud-q5-k-xl",
        "name": "Cyber-Tiel Coder 35B-A3B UD-Q5_K_XL",
        "artifact": "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_S.gguf",
        "quantization": "UD-Q4_K_S",
    }
    admitted, reason = default_admits(entry)
    assert admitted is False
    assert "Q5" in reason


def test_q8_exception_requires_a_visible_reason() -> None:
    entry = {"id": "qwen3-8-27b-q8", "quantization": "Q8_0"}
    admitted, reason = default_admits(entry)
    assert admitted is False

    admitted, reason = default_admits(entry, fits_fully_in_vram=True)
    assert admitted is True
    assert reason.startswith("fits-fully-in-vram")

    accuracy = {**entry, "accuracy_tier": True}
    admitted, reason = default_admits(accuracy)
    assert admitted is True
    assert reason.startswith("accuracy-tier")


def test_policy_table_covers_three_classes() -> None:
    assert {row["class"] for row in policy_table()} == {ALLOWED, ALLOWED_EXCEPTION, EXCLUDED_BY_DEFAULT}


def test_tournament_regeneration_is_idempotent(tmp_path: Path) -> None:
    script = ROOT / "scripts/generate-hardware-tier-tournaments"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    for output in (first, second):
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            check=True, cwd=ROOT, env=env, capture_output=True, text=True,
        )

    assert first.read_bytes() == second.read_bytes()
    # The committed artefact is the script's output, not a hand-edit.
    assert first.read_bytes() == (ROOT / "references/hardware-tier-tournaments.json").read_bytes()
    payload = json.loads(first.read_text())
    assert payload["schema"] == "turbofit.hardware-tier-tournaments/v2"
