from __future__ import annotations

import http.client
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "turbofit_gateway",
    Path(__file__).resolve().parents[1] / "scripts/turbofit-gateway.py",
)
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


def _profiles(tmp_path: Path) -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {
            "grm-carwin-262k": {"context": 262144, "description": "fast pair"},
            "grm-carwin-1m": {"context": 1048576, "description": "long pair"},
        },
    }))
    return path


def test_unified_catalog_uses_stable_profile_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    models = GATEWAY.provider_models()
    assert [item["id"] for item in models] == [
        "auto",
        "active:main",
        "active:aux",
    ]
    assert all(item["owned_by"] == "turbofit" for item in models)


def test_universal_model_strings_encode_role_without_a_second_provider():
    assert GATEWAY.parse_provider_model("auto") == ("auto", "main")
    assert GATEWAY.parse_provider_model("auto:aux") == ("auto", "aux")
    assert GATEWAY.parse_provider_model("active:main") == ("active", "main")
    assert GATEWAY.parse_provider_model("active:aux") == ("active", "aux")
    assert GATEWAY.parse_provider_model("grm-carwin-262k") == ("grm-carwin-262k", "main")
    assert GATEWAY.parse_provider_model("grm-carwin-262k:aux") == ("grm-carwin-262k", "aux")


def test_manual_selection_rejects_unknown_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    assert GATEWAY.resolve_requested_profile("not-in-catalog") is None


def test_auto_reuses_active_profile_without_rerunning_hardware_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    calls = []
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: "grm-carwin-1m")
    monkeypatch.setattr(GATEWAY, "recommend_profile", lambda: (_ for _ in ()).throw(AssertionError("must not recommend while a profile is active")))
    monkeypatch.setattr(GATEWAY, "activate_profile", lambda profile: calls.append(profile) or True)

    assert GATEWAY.resolve_requested_profile("auto") == "grm-carwin-1m"
    assert calls == []


def test_auto_without_active_profile_uses_hardware_recommender(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    calls = []
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: None)
    monkeypatch.setattr(GATEWAY, "recommend_profile", lambda: "grm-carwin-262k")
    monkeypatch.setattr(GATEWAY, "activate_profile", lambda profile: calls.append(profile) or True)

    assert GATEWAY.resolve_requested_profile("auto") == "grm-carwin-262k"
    assert calls == ["grm-carwin-262k"]


def test_manual_selection_uses_exact_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    calls = []
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: "grm-carwin-1m")
    monkeypatch.setattr(GATEWAY, "activate_profile", lambda profile: calls.append(profile) or True)

    assert GATEWAY.resolve_requested_profile("grm-carwin-1m") == "grm-carwin-1m"
    assert calls == []


def test_aux_stream_reaches_client_and_propagates_disconnect(monkeypatch):
    release_upstream = threading.Event()
    upstream_disconnected = threading.Event()
    seen_payload = {}

    class StreamingUpstream(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            seen_payload.update(json.loads(self.rfile.read(length)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n')
            self.wfile.flush()
            try:
                while not release_upstream.wait(timeout=0.05):
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                upstream_disconnected.set()

        def log_message(self, format, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), StreamingUpstream)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    monkeypatch.setattr(GATEWAY, "resolve_requested_profile", lambda _model: "active")
    monkeypatch.setattr(GATEWAY, "AUX_MAX_TOKENS", 64, raising=False)
    monkeypatch.setattr(
        GATEWAY,
        "resolve_aux",
        lambda: {
            "base_url": f"http://127.0.0.1:{upstream.server_port}",
            "alias": "test-aux",
            "state": "ready",
        },
    )
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    upstream_thread.start()
    gateway_thread.start()
    client = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=0.5)
    payload = json.dumps(
        {
            "model": "active:aux",
            "messages": [{"role": "user", "content": "stream now"}],
            "stream": True,
            "max_tokens": 65_536,
            "n_predict": 65_536,
        }
    )
    try:
        client.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        assert response.status == 200
        assert response.readline().startswith(b"data:")
        assert seen_payload["max_tokens"] == 64
        assert seen_payload["n_predict"] == 64
        assert seen_payload["chat_template_kwargs"] == {
            "enable_thinking": False,
            "thinking_mode": "disabled",
        }
        assert seen_payload["reasoning_format"] == "none"
        assert not release_upstream.is_set()
        response.close()
        client.close()
        assert upstream_disconnected.wait(timeout=2)
    finally:
        release_upstream.set()
        client.close()
        gateway.shutdown()
        upstream.shutdown()
        gateway.server_close()
        upstream.server_close()


def test_aux_auto_follows_active_pair_without_rerunning_recommendation(monkeypatch):
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: "grm-carwin-262k")
    monkeypatch.setattr(GATEWAY, "recommend_profile", lambda: (_ for _ in ()).throw(AssertionError("must not recommend")))
    assert GATEWAY.resolve_requested_profile("auto:aux") == "grm-carwin-262k"
