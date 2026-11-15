"""A campaign row that declares a dedicated auxiliary must never run as ``auto``.

Observed on omarchy: the row ``arete-27b-q5--maple-preview-tq2--64k`` declares
``maple-preview-tq2`` as its auxiliary, yet ``/status`` reported
``aux: auto:arete-27b-q5``.  The gateway's ``resolve_aux`` fell through
"declared aux not answering" into its shared-main convenience fallback, which
turned a dead aux engine into a fabricated route and a benchmark that measured
the wrong thing.  This module pins the behaviour: a declared *dedicated*
auxiliary is either that model or an explicit ``down`` record — never
``auto:*`` — while a recipe that really is ``shared-main`` keeps sharing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gateway():
    spec = importlib.util.spec_from_file_location(
        "turbofit_gateway_aux_routing", ROOT / "scripts" / "turbofit-gateway.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(tmp_path: Path, *, aux_mode: str, aux_alias: str, with_aux_component: bool = True):
    components = [
        {"role": "main", "kind": "process", "pid": 111, "port": 11605},
    ]
    if with_aux_component:
        components.insert(0, {"role": "aux", "kind": "process", "pid": 222, "port": 11610})
    path = tmp_path / "intelligence-campaign-runtime-state.json"
    path.write_text(
        json.dumps(
            {
                "active": "campaign:arete-27b-q5--maple-preview-tq2--64k",
                "context": 0,
                "expected": {
                    "main_alias": "arete-27b-q5",
                    "aux_alias": aux_alias,
                    "aux_mode": aux_mode,
                },
                "components": components,
                "activating": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _prepare(monkeypatch, tmp_path, *, aux_state, **state_kwargs):
    gateway = _gateway()
    path = _state(tmp_path, **state_kwargs)
    monkeypatch.setattr(gateway, "RUNTIME_STATE", str(path))
    monkeypatch.setattr(gateway, "campaign_lease_active", lambda: False)

    def fake_backend_state(port, expected_model_id=None):
        return aux_state if int(port) == 11610 else "ready"

    monkeypatch.setattr(gateway, "backend_state", fake_backend_state)
    monkeypatch.setattr(gateway, "_cache", {"main": None, "aux": None, "ts": 0.0})
    return gateway


def test_maple_row_with_dead_aux_reports_the_declared_aux_down(tmp_path, monkeypatch) -> None:
    gateway = _prepare(
        monkeypatch, tmp_path,
        aux_state="down",
        aux_mode="dedicated",
        aux_alias="maple-preview-tq2",
    )

    aux = gateway.resolve_aux()

    assert aux is not None
    assert aux["alias"] == "maple-preview-tq2"
    assert not str(aux["alias"]).startswith("auto:")
    assert aux["state"] == "down"
    assert aux["mode"] == "dedicated"
    assert aux["port"] == 11610
    assert aux.get("declared") is True
    assert "not answering" in aux["error"]


def test_maple_row_with_ready_aux_routes_to_the_maple_model(tmp_path, monkeypatch) -> None:
    gateway = _prepare(
        monkeypatch, tmp_path,
        aux_state="ready",
        aux_mode="dedicated",
        aux_alias="maple-preview-tq2",
    )

    aux = gateway.resolve_aux()

    assert aux["alias"] == "maple-preview-tq2"
    assert aux["state"] == "ready"
    assert aux["mode"] == "dedicated"
    assert aux["port"] == 11610


def test_declared_aux_can_never_yield_an_auto_route(tmp_path, monkeypatch) -> None:
    """Sweep every non-ready probe state: none may produce ``auto:``."""
    gateway = _prepare(
        monkeypatch, tmp_path,
        aux_state="down",
        aux_mode="dedicated",
        aux_alias="maple-preview-tq2",
    )
    for probed in ("down", "loading"):
        monkeypatch.setattr(
            gateway, "backend_state",
            lambda port, expected_model_id=None, _p=probed: _p if int(port) == 11610 else "ready",
        )
        monkeypatch.setattr(gateway, "_cache", {"main": None, "aux": None, "ts": 0.0})
        aux = gateway.resolve_aux()
        assert aux is not None
        assert aux["alias"] == "maple-preview-tq2", probed
        assert not str(aux["alias"]).startswith("auto:"), probed


def test_shared_main_row_still_shares_the_main_model(tmp_path, monkeypatch) -> None:
    gateway = _prepare(
        monkeypatch, tmp_path,
        aux_state="down",
        aux_mode="shared-main",
        aux_alias="auto:arete-27b-q5",
        with_aux_component=False,
    )

    aux = gateway.resolve_aux()

    assert aux["alias"] == "auto:arete-27b-q5"
    assert aux["mode"] == "shared-main"
    assert aux["shared_main_alias"] == "arete-27b-q5"
