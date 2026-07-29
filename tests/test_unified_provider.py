from __future__ import annotations

import http.client
import importlib.util
import json
import select
import socket
import threading
import time
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


def test_provider_metadata_reports_active_runtime_context(tmp_path, monkeypatch):
    state = tmp_path / "runtime-state.json"
    state.write_text(json.dumps({
        "active": "mac-native",
        "routes": {
            "main": {
                "kind": "local",
                "alias": "bonsai",
                "port": 8092,
                "context_length": 65536,
            }
        },
    }))
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))

    assert GATEWAY.active_context_length() == 65536
    assert {model["context_length"] for model in GATEWAY.provider_models()} == {65536}


def test_v1_props_adapts_llama_props_for_hermes(monkeypatch):
    class PropsUpstream(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/props"
            body = json.dumps({
                "model_alias": "bonsai",
                "default_generation_settings": {"n_ctx": 65536},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), PropsUpstream)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    monkeypatch.setattr(
        GATEWAY,
        "resolve_main",
        lambda: {
            "base_url": f"http://127.0.0.1:{upstream.server_port}",
            "alias": "bonsai",
            "state": "ready",
            "context_length": 65536,
        },
    )
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    client = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=2)
    try:
        client.request("GET", "/v1/props")
        response = client.getresponse()
        props = json.loads(response.read())
        assert response.status == 200
        assert props["context_length"] == 65536
        assert props["provider_model"] == "auto"
    finally:
        client.close()
        gateway.shutdown()
        upstream.shutdown()
        gateway.server_close()
        upstream.server_close()


def test_duplicate_inflight_request_is_rejected_without_queueing(monkeypatch):
    request_started = threading.Event()
    release_request = threading.Event()

    class SlowUpstream(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request_started.set()
            release_request.wait(timeout=3)
            body = b'{"choices":[{"message":{"content":"ok"}}]}'
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), SlowUpstream)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    backend = {
        "base_url": f"http://127.0.0.1:{upstream.server_port}",
        "alias": "test-main",
        "state": "ready",
    }
    monkeypatch.setattr(GATEWAY, "resolve_requested_profile", lambda _model: "active")
    monkeypatch.setattr(GATEWAY, "resolve_main", lambda: backend)
    GATEWAY._inflight_requests.clear()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    payload = json.dumps({
        "model": "auto",
        "messages": [{"role": "user", "content": "same"}],
        "stream": True,
    })
    first = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=2)
    second = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=2)
    try:
        first.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        assert request_started.wait(timeout=1)
        started = time.monotonic()
        second.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        response = second.getresponse()
        body = json.loads(response.read())
        assert response.status == 409
        assert response.getheader("X-Turbofit-Deduplicated") == "true"
        assert body["error"] == "request_in_progress"
        assert time.monotonic() - started < 1
    finally:
        first.close()
        second.close()
        release_request.set()
        gateway.shutdown()
        upstream.shutdown()
        gateway.server_close()
        upstream.server_close()


def test_disconnect_before_first_upstream_byte_cancels_request(monkeypatch):
    request_started = threading.Event()
    release_request = threading.Event()
    upstream_disconnected = threading.Event()

    class PrefillUpstream(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request_started.set()
            while not release_request.wait(timeout=0.05):
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(
                    1, socket.MSG_PEEK | socket.MSG_DONTWAIT
                ):
                    upstream_disconnected.set()
                    return
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"late")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                upstream_disconnected.set()

        def log_message(self, format, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), PrefillUpstream)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    monkeypatch.setattr(GATEWAY, "resolve_requested_profile", lambda _model: "active")
    monkeypatch.setattr(
        GATEWAY,
        "resolve_main",
        lambda: {
            "base_url": f"http://127.0.0.1:{upstream.server_port}",
            "alias": "test-main",
            "state": "ready",
        },
    )
    GATEWAY._inflight_requests.clear()
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    client = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=2)
    payload = json.dumps({
        "model": "auto",
        "messages": [{"role": "user", "content": "long prefill"}],
        "stream": True,
    })
    try:
        client.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        assert request_started.wait(timeout=1)
        client.close()
        assert not GATEWAY._inflight_requests or _wait_until(
            lambda: not GATEWAY._inflight_requests, timeout=2
        )
        assert upstream_disconnected.wait(timeout=2)
    finally:
        release_request.set()
        client.close()
        gateway.shutdown()
        upstream.shutdown()
        gateway.server_close()
        upstream.server_close()


def _wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()
