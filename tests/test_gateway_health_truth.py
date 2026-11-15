"""Health must not call a hosted answer "ready".

Observed on omarchy: the controller had yielded residency to external GPU
pressure, the local rung's unit was stopped, and `/health` still answered
`{"ok": true, "main": "ready"}` while every token came from the Nous API. A
client (or an agent) reading that had no way to learn a local model was not
answering.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "turbofit_gateway_health", ROOT / "scripts" / "turbofit-gateway.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _health(module, monkeypatch, *, main, aux=None, remote=(), released=False,
            declared=None):
    captured: dict = {}

    def send_json(status, payload, extra_headers=None):
        captured["status"] = status
        captured["payload"] = payload

    monkeypatch.setattr(module, "resolve_main", lambda: main)
    monkeypatch.setattr(module, "resolve_aux", lambda: aux)
    monkeypatch.setattr(module, "_remote_models_or_empty", lambda: list(remote))
    monkeypatch.setattr(
        module.GatewayHandler, "_declared_local_route", staticmethod(lambda role: declared),
        raising=False,
    )
    monkeypatch.setattr(
        module.GatewayHandler, "_controller_released", staticmethod(lambda: released),
        raising=False,
    )

    handler = object.__new__(module.GatewayHandler)
    handler._send_json = send_json
    module.GatewayHandler._send_health(handler)
    return captured


@pytest.fixture()
def module():
    return _module()


def test_yielded_local_rung_is_not_reported_ready(module, monkeypatch):
    captured = _health(
        module,
        monkeypatch,
        main={"alias": "api-fallback", "is_api": True, "state": "ready"},
        declared={"kind": "local", "rung_id": "strata-flashnext-32k"},
        released=True,
    )

    payload = captured["payload"]
    assert payload["main"] == "yielded"
    assert payload["local"] is False
    assert payload["cloud_fallback"] is True
    assert payload["backend"] == "api-fallback"
    assert payload["declared_rung"] == "strata-flashnext-32k"
    assert "yielded" in payload["note"]


def test_dead_local_rung_with_api_answer_is_api_fallback(module, monkeypatch):
    captured = _health(
        module,
        monkeypatch,
        main={"alias": "api-fallback", "is_api": True, "state": "ready"},
        declared={"kind": "local", "rung_id": "iq3s-262k"},
        released=False,
    )

    payload = captured["payload"]
    assert payload["main"] == "api-fallback"
    assert payload["cloud_fallback"] is True
    assert "not serving" in payload["note"]


def test_a_real_local_backend_still_reports_ready(module, monkeypatch):
    captured = _health(
        module,
        monkeypatch,
        main={"alias": "qwen3-8-flash-next-strata", "is_api": False, "state": "ready"},
        declared={"kind": "local", "rung_id": "strata-flashnext-32k"},
    )

    payload = captured["payload"]
    assert payload["main"] == "ready"
    assert payload["local"] is True
    assert payload["cloud_fallback"] is False
    assert "note" not in payload


def test_a_gpu_less_peer_node_is_federated(module, monkeypatch):
    captured = _health(
        module,
        monkeypatch,
        main=None,
        remote=[{"remote_model": "qwen3-8-flash-next-strata"}],
    )

    payload = captured["payload"]
    assert payload["main"] == "federated"
    assert payload["backend"] == "none"
    assert payload["local"] is False


def test_health_payload_is_json_serialisable(module, monkeypatch):
    captured = _health(
        module,
        monkeypatch,
        main={"alias": "api-fallback", "is_api": True, "state": "ready"},
        declared={"kind": "local", "rung_id": "x"},
        released=True,
    )

    assert json.loads(json.dumps(captured["payload"]))["main"] == "yielded"
