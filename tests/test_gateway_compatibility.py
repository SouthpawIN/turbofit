"""HTTP-level gateway compatibility tests (TF3 + TF4).

A fake local upstream captures the actual forwarded payload and returns
ordered chunks. These are synthetic transport fixtures: they prove what the
gateway forwards and streams, not model behaviour.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "turbofit_gateway_compatibility",
    Path(__file__).resolve().parents[1] / "scripts/turbofit-gateway.py",
)
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


class FakeUpstream(BaseHTTPRequestHandler):
    """Captures the forwarded payload; replies with scripted ordered chunks."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        captured["payload"] = json.loads(self.rfile.read(length) or b"{}")
        captured["path"] = self.path
        mode = captured.setdefault("mode", "sse")
        if mode == "error":
            body = json.dumps({"error": {"message": "boom"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in captured["chunks"]:
            self.wfile.write(chunk)
            self.wfile.flush()
        self.close_connection = True


@pytest.fixture()
def fake_stack(monkeypatch):
    captured.clear()
    captured["chunks"] = [
        b'data: {"delta": "a"}\n\n',
        b'data: {"delta": "b"}\n\n',
        b'data: {"usage": {"prompt_tokens": 7, "completion_tokens": 2}}\n\n',
        b"data: [DONE]\n\n",
    ]
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    gateway = ThreadingHTTPServer(("127.0.0.1", 0), GATEWAY.GatewayHandler)
    monkeypatch.setattr(
        GATEWAY,
        "resolve_requested_profile",
        lambda _model: "active-profile",
    )
    monkeypatch.setattr(
        GATEWAY,
        "resolve_main",
        lambda: {
            "base_url": f"http://127.0.0.1:{upstream.server_port}",
            "alias": "qwen-test",
            "state": "ready",
        },
    )
    for thread in (
        threading.Thread(target=upstream.serve_forever, daemon=True),
        threading.Thread(target=gateway.serve_forever, daemon=True),
    ):
        thread.start()
    client = http.client.HTTPConnection("127.0.0.1", gateway.server_port, timeout=5)

    yield client, captured

    client.close()
    gateway.shutdown()
    upstream.shutdown()
    gateway.server_close()
    upstream.server_close()


captured: dict = {}


def post(client, payload):
    client.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    return client.getresponse()


def test_tf3_effort_becomes_finite_budget_when_policy_configured(
    fake_stack, monkeypatch
):
    monkeypatch.setattr(
        GATEWAY,
        "reasoning_policy_for",
        lambda backend: {"hard_cap": 4096, "default_budget": 1024},
        raising=False,
    )
    client, captured = fake_stack
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high", "enabled": True},
        },
    )
    assert response.status == 200
    response.read()
    assert captured["payload"]["thinking_budget_tokens"] == 2048


def test_tf3_no_policy_leaves_reasoning_fields_untouched(fake_stack):
    client, captured = fake_stack
    payload = {
        "model": "active:main",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "high"},
    }
    response = post(client, payload)
    response.read()
    assert "thinking_budget_tokens" not in captured["payload"]


def test_tf3_explicit_caller_controls_are_preserved_not_squashed(
    fake_stack, monkeypatch
):
    monkeypatch.setattr(
        GATEWAY,
        "reasoning_policy_for",
        lambda backend: {"hard_cap": 4096, "default_budget": 1024},
        raising=False,
    )
    client, captured = fake_stack
    caller_kwargs = {"enable_thinking": True, "thinking_mode": "enabled"}
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high", "enabled": True},
            "chat_template_kwargs": dict(caller_kwargs),
        },
    )
    response.read()
    forwarded = captured["payload"]["chat_template_kwargs"]
    assert forwarded["enable_thinking"] is True
    assert forwarded["thinking_mode"] == "enabled"


def test_tf3_explicit_budget_is_clamped_to_backend_cap(fake_stack, monkeypatch):
    monkeypatch.setattr(
        GATEWAY,
        "reasoning_policy_for",
        lambda backend: {"hard_cap": 4096, "default_budget": 1024},
        raising=False,
    )
    client, captured = fake_stack
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget_tokens": 2_147_483_647,
        },
    )
    response.read()
    assert captured["payload"]["thinking_budget_tokens"] == 4096


def test_tf4_streaming_defaults_include_usage(fake_stack):
    client, captured = fake_stack
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status == 200
    assert captured["payload"]["stream_options"] == {"include_usage": True}


def test_tf4_explicit_false_survives(fake_stack):
    client, captured = fake_stack
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": False},
        },
    )
    response.read()
    assert captured["payload"]["stream_options"]["include_usage"] is False


def test_tf4_nonstream_untouched(fake_stack):
    client, captured = fake_stack
    captured["mode"] = "error"  # any reply works; we only inspect the request
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status in (200, 503)
    response.read()
    assert "stream_options" not in captured["payload"]


def test_stream_passthrough_delivers_ordered_chunks_usage_and_done(fake_stack):
    client, captured = fake_stack
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "ping"}}],
        },
    )
    assert response.status == 200
    assert "text/event-stream" in response.getheader("Content-Type", "")
    body = response.read()
    lines = [line for line in body.split(b"\n") if line.startswith(b"data: ")]
    assert lines[0] == b'data: {"delta": "a"}'
    assert lines[1] == b'data: {"delta": "b"}'
    assert json.loads(lines[2][6:])["usage"]["prompt_tokens"] == 7
    assert lines[3] == b"data: [DONE]"
    # tools pass through to the upstream payload untouched
    assert captured["payload"]["tools"][0]["function"]["name"] == "ping"


def test_upstream_error_response_is_surfaced(fake_stack):
    client, captured = fake_stack
    captured["mode"] = "error"
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status == 503
    body = json.loads(response.read())
    assert body["error"] == "no_backend"


def test_model_rewritten_to_backing_alias(fake_stack):
    client, captured = fake_stack
    response = post(
        client,
        {
            "model": "active:main",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    response.read()
    assert captured["payload"]["model"] == "qwen-test"


def test_missing_runtime_state_names_env_override(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(tmp_path / "absent.json"))
    with caplog.at_level("ERROR", logger="gate"):
        assert GATEWAY.active_profile() is None
    assert "TURBOFIT_RUNTIME_STATE" in caplog.text