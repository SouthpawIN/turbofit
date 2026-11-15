"""Readiness must not report a serving engine as a failed model load.

The campaign row ``arete-27b-...-maple-preview-20b-a1b-tq2-0-64k`` failed with
"aux failed to load model maple-preview-tq2: ... Likely cause: a tokenizer,
architecture or projector mismatch" while the engine log showed the model
loading, serving 512 tokens and shutting down cleanly.  The readiness check
parsed nothing, ignored the port, and treated "the process I spawned is no
longer running" as "the weights are bad".

These tests drive real subprocesses through ``CampaignBackend.wait_ready``.
"""
from __future__ import annotations

import socket
import sys
import textwrap
from pathlib import Path

import pytest

from turbofit_runtime.backend import CampaignBackend
from turbofit_runtime.recipes import ResolvedComponent

ENGINE = textwrap.dedent(
    '''
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port, mode, model = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    served = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            served["n"] += 1
            if self.path == "/health":
                body = {"status": "ok"}
            elif self.path == "/v1/models":
                body = {"data": [{"id": model, "meta": {"n_ctx": 65536}}]}
            else:
                self.send_error(404)
                return
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    if mode == "serve-then-exit":
        server = HTTPServer(("127.0.0.1", port), Handler)
        print("srv load_model: model loaded", flush=True)
        print("srv llama_server: server is listening on http://127.0.0.1:%d" % port, flush=True)
        # One health answer and one model-list answer, then a clean exit: the
        # engine did its job and the harness tore it down.
        for _ in range(2):
            server.handle_request()
        print("slot release: stop processing: n_tokens = 535", flush=True)
        print("srv operator(): cleaning up before exit...", flush=True)
        sys.exit(0)
    elif mode == "health-only-then-exit":
        # Answers /health but not /v1/models, then exits cleanly.
        server = HTTPServer(("127.0.0.1", port), Handler)
        server.handle_request()
        print("srv operator(): cleaning up before exit...", flush=True)
        sys.exit(0)
    elif mode == "die-before-serving":
        print("llama_model_load: error loading model: failed to load model from %s" % model, flush=True)
        print("llama_model_loader: tensor data is not within the file bounds", flush=True)
        sys.exit(3)
    elif mode == "port-conflict":
        print("srv start: binding port with default address family", flush=True)
        print("E srv start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: %d" % port, flush=True)
        print("srv operator(): operator(): cleaning up before exit...", flush=True)
        sys.exit(1)
    else:
        raise SystemExit("unknown mode")
    '''
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def engine_component(port: int, *, alias: str = "maple-preview-tq2") -> ResolvedComponent:
    return ResolvedComponent(
        role="aux",
        family="maple-preview",
        alias=alias,
        kind="process",
        method="tq2",
        gpu="0",
        port=port,
        command=("llama-server",),
        model_path="",
    )


@pytest.fixture()
def backend(tmp_path: Path) -> CampaignBackend:
    return CampaignBackend(
        gateway_script=tmp_path / "gateway.py",
        result_dir=tmp_path,
        runtime_state=tmp_path / "runtime-state.json",
        campaign_lease_path=tmp_path / "lease.json",
    )


def spawn(tmp_path: Path, port: int, mode: str, model: str = "maple-preview-tq2",
          alias: str = "maple-preview-tq2"):
    import subprocess

    script = tmp_path / "fake_engine.py"
    script.write_text(ENGINE, encoding="utf-8")
    log = tmp_path / f"campaign-aux-{port}.log"
    handle_file = log.open("w")
    process = subprocess.Popen(
        [sys.executable, str(script), str(port), mode, model],
        stdout=handle_file, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    handle_file.close()
    component = engine_component(port, alias=alias)
    return component, {"kind": "process", "pid": process.pid, "process": process,
                       "port": port, "log": str(log)}


def test_engine_that_serves_then_exits_cleanly_is_ready(backend, tmp_path) -> None:
    port = free_port()
    component, handle = spawn(tmp_path, port, "serve-then-exit")

    check = backend.wait_ready(component, handle)

    assert check["model"] == "maple-preview-tq2"
    assert check["context"] == 65536
    assert check["health"]["status"] == "ok"


def test_engine_that_answers_health_then_exits_cleanly_is_not_a_load_failure(backend, tmp_path) -> None:
    port = free_port()
    component, handle = spawn(tmp_path, port, "health-only-then-exit")

    check = backend.wait_ready(component, handle)

    assert check["model"] == "maple-preview-tq2"
    assert check["exited_cleanly"] is True


def test_engine_that_exits_before_serving_is_a_load_failure_with_its_log_tail(backend, tmp_path) -> None:
    port = free_port()
    component, handle = spawn(tmp_path, port, "die-before-serving")

    with pytest.raises(RuntimeError) as caught:
        backend.wait_ready(component, handle)

    message = str(caught.value)
    assert "failed to load model" in message
    # The engine's own last lines, not a guess about tokenizers.
    assert "tensor data is not within the file bounds" in message
    assert "error loading model" in message
    assert "tokenizer" not in message.lower()
    assert "exit code 3" in message


def test_port_conflict_is_reported_as_a_port_conflict_not_a_load_failure(backend, tmp_path) -> None:
    port = free_port()
    component, handle = spawn(tmp_path, port, "port-conflict")

    with pytest.raises(RuntimeError) as caught:
        backend.wait_ready(component, handle)

    message = str(caught.value)
    assert "could not bind port" in message
    assert str(port) in message
    assert "pkill -f llama-server" in message
    assert "failed to load model" not in message
    assert "couldn't bind HTTP server socket" in message


def test_port_serving_another_model_is_reported_honestly(backend, tmp_path) -> None:
    port = free_port()
    component, handle = spawn(tmp_path, port, "serve-then-exit")
    # Simulate a stray engine from an earlier run owning the port: stop the
    # spawned process, then serve a different model id on the same port.
    handle["process"].kill()
    handle["process"].wait(timeout=10)

    import subprocess

    foreign_log = (tmp_path / "foreign.log").open("w")
    foreign = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(
            f'''
            import json
            from http.server import BaseHTTPRequestHandler, HTTPServer
            class H(BaseHTTPRequestHandler):
                def do_GET(self):
                    body = json.dumps({{"status": "ok"}} if self.path == "/health"
                                      else {{"data": [{{"id": "cyber-tiel-coder-35b-a3b"}}]}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def log_message(self, *a): pass
            HTTPServer(("127.0.0.1", {port}), H).serve_forever()
            '''
        )],
        stdout=foreign_log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    foreign_log.close()
    try:
        import time
        for _ in range(50):
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        with pytest.raises(RuntimeError) as caught:
            backend.wait_ready(component, handle)
        message = str(caught.value)
        assert "cyber-tiel-coder-35b-a3b" in message
        assert "stray engine" in message
        assert "failed to load model" not in message
    finally:
        foreign.kill()
