"""Port reclaim policy: never die on a busy port, never kill a foreign process."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from turbofit_runtime.backend import CampaignBackend
from turbofit_runtime.recipes import ResolvedComponent


LISTENER = (
    "import socket,sys,time\n"
    "port=int(sys.argv[2])\n"
    "s=socket.socket()\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', port))\n"
    "s.listen()\n"
    "print('listening', flush=True)\n"
    "time.sleep(120)\n"
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_listener(port: int) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", LISTENER, "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return process
        time.sleep(0.05)
    process.kill()
    raise AssertionError(f"listener never bound port {port}")


def listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def component(port: int, binary: str) -> ResolvedComponent:
    return ResolvedComponent(
        role="main", family="Test Family", alias="test-alias", kind="process",
        method="baseline", gpu="0", port=port,
        command=(binary, "-m", "/models/m.gguf", "--host", "127.0.0.1",
                 "--port", str(port), "--alias", "test-alias"),
    )


def backend_at(tmp_path: Path) -> CampaignBackend:
    return CampaignBackend(
        gateway_script=tmp_path / "gateway.py",
        result_dir=tmp_path / "results",
        runtime_state=tmp_path / "runtime.json",
    )


def test_free_port_is_used_unchanged(tmp_path: Path) -> None:
    port = free_port()
    backend = backend_at(tmp_path)
    original = component(port, str(tmp_path / "llama-server"))

    claimed = backend.claim_port(original)

    assert claimed.port == port
    assert claimed.command == original.command


def test_owned_residual_engine_is_reclaimed(tmp_path: Path) -> None:
    """Same binary, our log for the port: prove it is ours, terminate, reuse."""
    port = free_port()
    results = tmp_path / "results"
    results.mkdir()
    # The log the earlier, killed run left behind for this port.
    (results / f"campaign-main-{port}.log").write_text("llama_model_load: ok\n")
    listener = start_listener(port)
    backend = backend_at(tmp_path)
    original = component(port, sys.executable)

    try:
        claimed = backend.claim_port(original)
    finally:
        if listener.poll() is None:
            listener.kill()
        listener.wait()

    assert claimed.port == port
    assert claimed.command == original.command
    assert not listening(port)
    assert listener.poll() is not None


def test_runtime_state_pid_identifies_owned_engine(tmp_path: Path) -> None:
    port = free_port()
    listener = start_listener(port)
    backend = backend_at(tmp_path)
    backend.runtime_state.write_text(json.dumps({
        "active": "campaign:test",
        "components": [{"role": "main", "pid": listener.pid, "port": port}],
    }))
    # A different binary still counts as ours: the campaign recorded the pid.
    original = component(port, str(tmp_path / "other-engine"))

    try:
        claimed = backend.claim_port(original)
    finally:
        if listener.poll() is None:
            listener.kill()
        listener.wait()

    assert claimed.port == port
    assert not listening(port)


def test_foreign_listener_moves_to_next_free_port_and_survives(tmp_path: Path) -> None:
    port = free_port()
    listener = start_listener(port)
    backend = backend_at(tmp_path)
    original = component(port, str(tmp_path / "llama-server"))

    try:
        claimed = backend.claim_port(original)
        assert claimed.port != port
        assert backend.PORT_RANGE_START <= claimed.port <= backend.PORT_RANGE_END
        assert str(claimed.port) in claimed.command
        assert claimed.command[claimed.command.index("--port") + 1] == str(claimed.port)
        # The foreign owner is untouched.
        assert listener.poll() is None
        assert listening(port)
    finally:
        listener.kill()
        listener.wait()


def test_reserved_ports_are_skipped(tmp_path: Path) -> None:
    port = free_port()
    listener = start_listener(port)
    backend = backend_at(tmp_path)
    original = component(port, str(tmp_path / "llama-server"))

    try:
        first = backend.claim_port(original)
        second = backend.claim_port(original, reserved={first.port})
        assert second.port != first.port
    finally:
        listener.kill()
        listener.wait()


def test_claim_port_rewrites_equals_form_and_keeps_non_port_tokens(tmp_path: Path) -> None:
    port = free_port()
    listener = start_listener(port)
    backend = backend_at(tmp_path)
    original = ResolvedComponent(
        role="aux", family="Test Family", alias="aux-alias", kind="process",
        method="baseline", gpu="0", port=port,
        command=("/opt/llama-server", "-m", "/models/a.gguf", f"--port={port}"),
    )

    try:
        claimed = backend.claim_port(original)
    finally:
        listener.kill()
        listener.wait()

    assert claimed.port != port
    assert f"--port={claimed.port}" in claimed.command
    assert claimed.command[0] == "/opt/llama-server"
    assert claimed.command[2] == "/models/a.gguf"
