"""The dashboard is an instrument cluster, not a JSON textarea.

The operator asked for round dials and a gear-shift tree they can shift by hand,
including gears served by another TurboFit node. The data already existed in
``status.dials``; these tests pin the shipped bundle to actually render it, so a
future refactor cannot quietly drop the cluster back to a text field.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "dashboard" / "dist" / "index.js").read_text(encoding="utf-8")
CSS = (ROOT / "dashboard" / "dist" / "style.css").read_text(encoding="utf-8")


def test_round_dials_are_rendered_as_svg_arcs() -> None:
    assert "function Gauge(" in JS
    assert "tf-gauge-needle" in JS
    assert "tf-gauge-track" in JS
    # three instruments: pressure, context, throughput
    for label in ("Memory pressure", "Context window", "Tokens / second"):
        assert label in JS


def test_gear_tree_renders_the_chain_and_can_shift() -> None:
    assert "function GearTree(" in JS
    assert "dials.chain.levels" in JS
    assert "async function shiftTo(" in JS
    assert "/api/plugins/turbofit/shift" in JS
    assert "onShift: shiftTo" in JS


def test_tailnet_gears_are_labelled_and_cost_no_local_vram() -> None:
    assert "level.remote || level.endpoint" in JS
    assert "no local VRAM" in JS
    assert "tailnet" in JS


def test_unmeasured_dials_read_unknown_not_zero() -> None:
    # A host that was never probed must not render as 0% pressure.
    assert 'ratio == null ? "—"' in JS
    assert "tf-gauge-unknown" in JS


def test_cluster_styles_ship_with_the_bundle() -> None:
    for selector in (".tf-dials", ".tf-gauge-needle", ".tf-gears", ".tf-gear-engaged", ".tf-gear-rail"):
        assert selector in CSS


def test_bundle_renders_the_cluster_without_throwing() -> None:
    """Execute the real bundle against a stub SDK and count what it draws."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        import pytest

        pytest.skip("node is not installed; the render smoke needs a JS runtime")
    result = subprocess.run(
        [node, str(ROOT / "dashboard" / "smoke_cluster.js")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RENDER OK" in result.stdout
    assert '"gauges":3' in result.stdout
    assert '"gears":3' in result.stdout
    assert '"tailnetBadge":true' in result.stdout
