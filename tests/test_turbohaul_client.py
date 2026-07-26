from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from turbofit_runtime.turbohaul_client import (
    TurbohaulClient,
    TurbohaulClientError,
    TurbohaulHTTPError,
)


class _Handler(BaseHTTPRequestHandler):
    records: list[dict[str, Any]] = []
    residents: list[dict[str, Any]] = [{"model_tag": "main"}]

    def log_message(self, _format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self.__class__.records.append({"method": "GET", "path": self.path})
        if self.path == "/status":
            self._json(
                200,
                {
                    "queue": {"depth": 2},
                    "active": self.__class__.residents[0] if self.__class__.residents else None,
                    "residents": self.__class__.residents,
                },
            )
            return
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": "main"}, {"name": "aux"}]})
            return
        if self.path == "/api/show?name=main%2F128k":
            self._json(200, {"model_tag": "main/128k", "context_size": 131072})
            return
        self._json(404, {"detail": "not found"})

    def do_PUT(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.__class__.records.append(
            {
                "method": "PUT",
                "path": self.path,
                "if_match": self.headers.get("If-Match"),
                "body": body,
            }
        )
        if self.path == "/api/manifests/main-128k":
            self._json(200, {"model_tag": "main-128k", "revision": 8}, etag='"8"')
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.__class__.records.append({"method": "POST", "path": self.path, "body": body})
        if self.path == "/v1/chat/completions":
            if body.get("keep_alive") == 0 and body.get("model") != "stubborn":
                self.__class__.residents = []
            self._json(200, {"choices": [{"message": {"role": "assistant", "content": "OK"}}]})
            return
        if self.path == "/api/pull-hf":
            self._json(
                200,
                {
                    "pull_id": "pull-test",
                    "status": "complete",
                    "sha256": body.get("expected_sha256"),
                    "bytes_written": 123,
                },
            )
            return
        self._json(404, {"detail": "not found"})

    def _json(self, status: int, payload: dict[str, Any], *, etag: str | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if etag:
            self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def fake_turbohaul() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type(
        "Handler",
        (_Handler,),
        {"records": [], "residents": [{"model_tag": "main"}]},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_status_returns_live_queue_and_active_snapshot() -> None:
    with fake_turbohaul() as (base_url, handler):
        client = TurbohaulClient(base_url, timeout_s=1)

        status = client.status()

    assert status["queue"]["depth"] == 2
    assert status["active"]["model_tag"] == "main"
    assert handler.records == [{"method": "GET", "path": "/status"}]


def test_list_and_show_models_use_ollama_shape_routes() -> None:
    with fake_turbohaul() as (base_url, handler):
        client = TurbohaulClient(base_url, timeout_s=1)

        models = client.list_models()
        detail = client.show_model("main/128k")

    assert [model["name"] for model in models] == ["main", "aux"]
    assert detail["context_size"] == 131072
    assert handler.records == [
        {"method": "GET", "path": "/api/tags"},
        {"method": "GET", "path": "/api/show?name=main%2F128k"},
    ]


def test_put_manifest_forwards_etag_and_returns_new_etag() -> None:
    manifest = {"model_tag": "main-128k", "revision": 7}
    with fake_turbohaul() as (base_url, handler):
        client = TurbohaulClient(base_url, timeout_s=1)

        result = client.put_manifest("main-128k", manifest, etag='"7"')

    assert result.payload["revision"] == 8
    assert result.etag == '"8"'
    assert handler.records == [
        {
            "method": "PUT",
            "path": "/api/manifests/main-128k",
            "if_match": '"7"',
            "body": manifest,
        }
    ]


def test_pull_hf_forwards_pinned_source_and_expected_hash() -> None:
    payload = {
        "repo_id": "org/repo",
        "filename": "model.gguf",
        "revision": "a" * 40,
        "expected_sha256": "b" * 64,
    }
    with fake_turbohaul() as (base_url, handler):
        client = TurbohaulClient(base_url, timeout_s=1, acquisition_timeout_s=2)

        response = client.pull_hf(**payload)

    assert response["status"] == "complete"
    assert response["sha256"] == "b" * 64
    assert handler.records == [
        {"method": "POST", "path": "/api/pull-hf", "body": payload}
    ]


def test_chat_completion_forwards_openai_payload() -> None:
    payload = {
        "model": "main",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
    }
    with fake_turbohaul() as (base_url, handler):
        client = TurbohaulClient(base_url, timeout_s=1)

        response = client.chat_completion(payload)

    assert response["choices"][0]["message"]["content"] == "OK"
    assert handler.records == [
        {"method": "POST", "path": "/v1/chat/completions", "body": payload}
    ]


def test_chat_completion_uses_activation_timeout(monkeypatch) -> None:
    client = TurbohaulClient("http://127.0.0.1:1", timeout_s=1, activation_timeout_s=37)
    captured = {}

    def fake_request(method, path, *, payload=None, headers=None, timeout_s=None):
        captured.update(method=method, path=path, payload=payload, timeout_s=timeout_s)
        return {"choices": []}

    monkeypatch.setattr(client, "_request_json", fake_request)
    payload = {"model": "main", "messages": [{"role": "user", "content": "go"}]}

    client.chat_completion(payload)

    assert captured == {
        "method": "POST",
        "path": "/v1/chat/completions",
        "payload": payload,
        "timeout_s": 37,
    }


def test_unload_model_uses_keep_alive_zero_and_verifies_residency() -> None:
    with fake_turbohaul() as (base_url, handler):
        client = TurbohaulClient(base_url, timeout_s=1)

        final_status = client.unload_model(
            "main", verification_timeout_s=1, poll_interval_s=0.01
        )

    assert final_status["residents"] == []
    assert handler.records[0]["method"] == "POST"
    assert handler.records[0]["body"]["model"] == "main"
    assert handler.records[0]["body"]["keep_alive"] == 0
    assert handler.records[-1] == {"method": "GET", "path": "/status"}


def test_http_errors_keep_status_method_and_path() -> None:
    with fake_turbohaul() as (base_url, _handler):
        client = TurbohaulClient(base_url, timeout_s=1)

        try:
            client.show_model("missing")
        except TurbohaulHTTPError as error:
            assert error.status == 404
            assert error.method == "GET"
            assert error.path == "/api/show?name=missing"
        else:
            raise AssertionError("expected TurbohaulHTTPError")


def test_unload_model_fails_if_status_still_reports_resident() -> None:
    with fake_turbohaul() as (base_url, handler):
        handler.residents = [{"model_tag": "stubborn"}]
        client = TurbohaulClient(base_url, timeout_s=1)

        try:
            client.unload_model(
                "stubborn", verification_timeout_s=0, poll_interval_s=0.01
            )
        except TurbohaulClientError as error:
            assert "did not unload" in str(error)
        else:
            raise AssertionError("expected unload verification failure")
