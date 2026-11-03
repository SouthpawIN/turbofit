"""`/turbofit` on unmeasured hardware must not dead-end on an empty ladder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _plugin():
    parent = str(ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import turbofit  # the plugin package next to this repo directory

    return turbofit


def test_scan_applies_the_best_portable_lane_when_nothing_is_measured(monkeypatch) -> None:
    plugin = _plugin()
    snapshot = {
        "ok": True,
        "recommendations": {"intelligence": [], "balanced": [], "speed": []},
        "compatible_lanes": [
            {"profile": "bonsai-27b-q1-64k", "main": "bonsai-27b", "context": 65536,
             "evidence_scope": "portable-fit", "validation_required": True,
             "command": "turbofit-runtime use bonsai-27b-q1-64k"},
            {"profile": "maple-64k", "main": "maple-preview-tq2", "context": 65536,
             "evidence_scope": "portable-fit", "validation_required": True},
        ],
    }
    monkeypatch.setattr(plugin, "recommendation_snapshot", lambda *a, **k: snapshot)
    monkeypatch.setattr(plugin, "_ladder", lambda *a, **k: [])
    applied: list[str] = []

    def fake_apply(profile_id, *, reason, extra=None):
        applied.append(profile_id)
        payload = {"ok": True, "shifted": True, "profile": profile_id, "reason": reason}
        payload.update(extra or {})
        return payload

    monkeypatch.setattr(plugin, "_apply", fake_apply)

    payload = json.loads(plugin._slash_turbofit(""))

    assert applied == ["bonsai-27b-q1-64k"]
    assert payload["evidence_scope"] == "portable-fit"
    assert payload["validation_required"] is True
    assert "portable-fit lane" in payload["message"]


def test_scan_reports_no_working_selection_when_no_lane_fits(monkeypatch) -> None:
    plugin = _plugin()
    monkeypatch.setattr(plugin, "recommendation_snapshot", lambda *a, **k: {
        "ok": False, "recommendations": {"intelligence": []}, "compatible_lanes": [],
    })
    monkeypatch.setattr(plugin, "_ladder", lambda *a, **k: [])

    payload = json.loads(plugin._slash_turbofit(""))

    assert payload["ok"] is False
    assert "unchanged" in payload["message"]
