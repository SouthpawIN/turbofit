"""The Windows installer must poll the gateway's real /health schema.

Regression: the gateway's ``_send_health()`` returns
``{"ok": bool, "main": state, "aux": state}``, but the installer waited on a
``status`` field that the endpoint has never emitted. A healthy stack therefore
false-failed after the full 10-minute readiness deadline.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-windows-native-service.ps1"

SPEC = importlib.util.spec_from_file_location(
    "turbofit_gateway_windows_health",
    ROOT / "scripts" / "turbofit-gateway.py",
)
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


def test_windows_installer_gateway_readiness_matches_live_health_schema(monkeypatch) -> None:
    """Serve the real gateway handler and probe it the way the installer does."""
    monkeypatch.setattr(
        GATEWAY,
        "resolve_main",
        lambda: {"alias": "bonsai-27b-1bit-128k-main", "state": "ready"},
    )
    monkeypatch.setattr(
        GATEWAY,
        "resolve_aux",
        lambda: {"alias": "shared-main", "state": "ready"},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        client.request("GET", "/health")
        response = client.getresponse()
        health = json.loads(response.read())
        assert response.status == 200
    finally:
        client.close()
        server.shutdown()
        server.server_close()

    assert health == {"ok": True, "main": "ready", "aux": "ready"}
    assert "status" not in health

    installer = INSTALLER.read_text(encoding="utf-8")
    checked = set(re.findall(r"\$GatewayHealth\.(\w+)", installer))
    assert checked == {"ok"}, (
        "Windows installer gateway readiness must read the gateway's real "
        f"`ok` field, found: {sorted(checked) or ['nothing']}"
    )
    assert checked <= set(health)
