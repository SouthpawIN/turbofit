"""Tests for the physics-predicted TurboFit List.

This suite also pins the built-in discovery fix: the Hermes Agent checkout
(``hermes_cli.local_runtime``) must be importable for the suite (see
``tests/conftest.py``), so ``hermes_fit.available()`` is True and the
predicted list is computed from real catalog physics — not fabricated.
"""
from __future__ import annotations

import pytest

from turbofit_runtime import hermes_fit
from turbofit_runtime.predicted_turbofit_list import (
    build_predicted_turbofit_list,
    render_predicted_turbofit_list,
)

TIERS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]


def test_built_in_runtime_is_importable() -> None:
    # Conftest wires the Hermes Agent checkout onto sys.path; this is the
    # root-cause fix that keeps the physics path from silently falling back.
    assert hermes_fit.available() is True


def test_predicted_list_covers_every_tier_and_is_labeled_predicted() -> None:
    payload = build_predicted_turbofit_list()

    assert payload["schema"] == "turbofit.list.predicted/v1"
    assert [item["hardware_level_gb"] for item in payload["levels"]] == TIERS
    # Every row is a prediction, never presented as measured evidence.
    assert all(item["status"] == "predicted" for item in payload["levels"])
    # The policy string must keep the honest distinction.
    assert "predicted" in payload["policy"] and "not measured" in payload["policy"]


def test_predicted_list_is_real_physics_not_fabricated() -> None:
    payload = build_predicted_turbofit_list()
    # At least one tier must resolve to a real catalog model with a numeric
    # predicted TPS — proof the physics ran (not a blank stub or invented row).
    fitted = [item for item in payload["levels"] if item["main"]]
    assert fitted, "physics produced no fits for any tier"
    for item in fitted:
        assert isinstance(item["predicted_tps"], float)
        assert item["predicted_tps"] >= 0.0
        assert item["required_gb"] is not None
        assert item["zero_spill"] is not None
        assert item["intelligence"] is not None


def test_rendered_predicted_list_is_clearly_labeled() -> None:
    text = render_predicted_turbofit_list(build_predicted_turbofit_list())

    assert text.startswith("# TurboFit List (physics-predicted)")
    assert "physics-predicted" in text
    assert "from measured benchmarks" in text
    assert "not evidence" in text
    # At least one real model name appears in the table.
    assert any(
        "`" in line and " GB |" in line
        for line in text.splitlines()
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
