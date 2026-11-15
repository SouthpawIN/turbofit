"""Federation contract: socket-aware CLI, probe classification, bands, bind.

These cover the defects that made a live tailnet peer disappear from TurboFit
with no error on either side:

* the CLI ignored a non-default tailscaled socket (rootless/userspace daemons),
* a refused port (peer up, gateway down) was reported as "peer down",
* discovery probed peers serially with no wall-clock bound,
* the gateway advertised model IDs but never its RAM/VRAM bands,
* the managed launch bound 127.0.0.1, so no peer could ever probe it.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

import turbofit_runtime.tailnet as tailnet
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint

SPEC = importlib.util.spec_from_file_location(
    "turbofit_gateway_federation",
    Path(__file__).resolve().parents[1] / "scripts" / "turbofit-gateway.py",
)
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


def completed(command: list[str], stdout: str = "{}", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def peer_status(*hosts: str) -> str:
    return json.dumps({
        "BackendState": "Running",
        "Self": {"DNSName": "self.example.ts.net."},
        "Peer": {
            host: {"Online": True, "DNSName": f"{host}."}
            for host in hosts
        },
    })


# --- tailscaled socket -------------------------------------------------------

def test_tailscale_command_uses_the_configured_socket(monkeypatch) -> None:
    monkeypatch.setenv("TURBOFIT_TAILSCALE_SOCKET", "/opt/data/state/tailscale/tailscaled.sock")
    assert tailnet.tailscale_command("status", "--json") == [
        "tailscale",
        "--socket=/opt/data/state/tailscale/tailscaled.sock",
        "status",
        "--json",
    ]


def test_tailscale_command_falls_back_to_tailscale_socket_and_default(monkeypatch) -> None:
    monkeypatch.delenv("TURBOFIT_TAILSCALE_SOCKET", raising=False)
    monkeypatch.setenv("TAILSCALE_SOCKET", "/run/tailscale.sock")
    assert tailnet.tailscale_command("serve", "status", "--json")[:2] == [
        "tailscale", "--socket=/run/tailscale.sock",
    ]
    monkeypatch.delenv("TAILSCALE_SOCKET")
    assert tailnet.tailscale_command("status", "--json") == ["tailscale", "status", "--json"]


def test_tailnet_status_and_discovery_pass_the_socket(monkeypatch) -> None:
    """A rootless daemon on a non-default socket must still be usable."""
    monkeypatch.setenv("TURBOFIT_TAILSCALE_SOCKET", "/tmp/ts.sock")
    seen: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        assert command[1] == "--socket=/tmp/ts.sock"
        if command[-2:] == ["status", "--json"]:
            return completed(command, peer_status("omarchy.example.ts.net"))
        return completed(command, "{}")

    status = tailnet.tailnet_status(command_runner=run)
    assert status["connected"] is True

    peers = tailnet.discover_turbofit_peers(
        command_runner=run, probe=lambda host, timeout: True
    )
    assert [peer["host"] for peer in peers] == ["omarchy.example.ts.net"]
    assert all(call[1] == "--socket=/tmp/ts.sock" for call in seen)


# --- probe classification ----------------------------------------------------

def refused(*_a, **_k):
    raise ConnectionRefusedError(111, "Connection refused")


def timed_out(*_a, **_k):
    raise socket.timeout("timed out")


def unresolved(*_a, **_k):
    raise socket.gaierror(-2, "Name or service not known")


def opened(*_a, **_k):
    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Ctx()


def test_probe_gateway_separates_open_refused_and_unreachable() -> None:
    assert tailnet.probe_gateway("a", 8091, connector=opened)["status"] == tailnet.PROBE_OPEN
    refused_result = tailnet.probe_gateway("b", 8091, connector=refused)
    assert refused_result["status"] == tailnet.PROBE_REFUSED
    assert "refused" in refused_result["error"]
    assert tailnet.probe_gateway("c", 8091, connector=timed_out)["status"] == tailnet.PROBE_UNREACHABLE
    assert tailnet.probe_gateway("d", 8091, connector=unresolved)["status"] == tailnet.PROBE_UNRESOLVED


def test_scan_reports_gateway_down_peer_up_instead_of_dropping_it() -> None:
    """omarchy is online with nothing on :8091 -> gateway_down, not 'no peers'."""

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return completed(command, peer_status("omarchy.example.ts.net"))

    scan = tailnet.scan_turbofit_peers(
        command_runner=run,
        probe=lambda host, timeout: {"host": host, "status": tailnet.PROBE_REFUSED, "error": "refused"},
    )

    assert scan["peers"] == []
    assert [peer["host"] for peer in scan["gateway_down"]] == ["omarchy.example.ts.net"]
    assert scan["gateway_down"][0]["os"] is None
    assert tailnet.last_peer_scan()["gateway_down"][0]["host"] == "omarchy.example.ts.net"


def test_discover_returns_only_gateways_that_answered_but_keeps_the_rest_for_diagnostics() -> None:
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return completed(command, peer_status("up.example.ts.net", "down.example.ts.net", "blackhole.example.ts.net"))

    def probe(host: str, timeout: float):
        return {
            "up.example.ts.net": tailnet.PROBE_OPEN,
            "down.example.ts.net": tailnet.PROBE_REFUSED,
            "blackhole.example.ts.net": tailnet.PROBE_UNREACHABLE,
        }[host]

    peers = tailnet.discover_turbofit_peers(command_runner=run, probe=probe)
    assert [peer["host"] for peer in peers] == ["up.example.ts.net"]
    assert peers[0]["url"] == "http://up.example.ts.net:8091"

    scan = tailnet.last_peer_scan()
    assert [peer["host"] for peer in scan["gateway_down"]] == ["down.example.ts.net"]
    assert [peer["host"] for peer in scan["unreachable"]] == ["blackhole.example.ts.net"]


def test_scan_bounds_a_hung_probe_by_the_deadline() -> None:
    """A black-holed peer must not cost a serial timeout per caller."""

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return completed(command, peer_status(*(f"peer{i}.example.ts.net" for i in range(4))))

    started = time.monotonic()
    scan = tailnet.scan_turbofit_peers(
        command_runner=run,
        # Blocks far longer than the scan deadline, as a hung getaddrinfo would.
        probe=lambda host, timeout: time.sleep(5) or True,
        timeout_s=0.2,
        deadline_s=0.3,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"scan took {elapsed:.1f}s; probes are not running concurrently"
    assert scan["peers_seen"] == 4
    assert len(scan["unreachable"]) == 4
    assert "deadline" in scan["unreachable"][0]["error"]


def test_scan_surfaces_tailscale_failure_instead_of_returning_silence() -> None:
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return completed(command, returncode=1, stderr="failed to connect to local tailscaled")

    scan = tailnet.scan_turbofit_peers(command_runner=run)
    assert scan["peers"] == []
    assert "failed to connect to local tailscaled" in scan["error"]
    assert tailnet.discover_turbofit_peers(command_runner=run) == []
    assert tailnet.last_peer_scan()["error"]


# --- band advertisement ------------------------------------------------------

def fingerprint() -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=64 * 1024,
        devices=(
            AcceleratorDevice(
                index=0, uuid="GPU-1", name="RTX 4090", vendor="nvidia",
                backend="cuda", memory_total_mb=24 * 1024,
                compute_capability="8.9", bus_id="0000:01:00.0",
            ),
        ),
    )


def test_node_advertisement_includes_actual_ram_and_vram() -> None:
    advertised = tailnet.node_advertisement(hardware=fingerprint(), provider_port=8091)

    assert advertised["bands"]["ram_gb"] == 64.0
    assert advertised["bands"]["vram_gb"] == 24.0
    assert advertised["bands"]["memory_pool"] == "dedicated"
    assert advertised["bands"]["total_usable_gb"] == pytest.approx(60.8 + 24.0, abs=0.2)
    assert advertised["accelerators"] == [{
        "index": 0, "name": "RTX 4090", "vendor": "nvidia",
        "backend": "cuda", "memory_gb": 24.0,
    }]
    assert advertised["provider_port"] == 8091


def test_gateway_catalog_envelope_carries_the_node_bands(monkeypatch) -> None:
    monkeypatch.setattr(GATEWAY, "node_advertisement_payload", lambda: {"bands": {"ram_gb": 64.0}})
    monkeypatch.setattr(GATEWAY, "provider_models", lambda: [])
    sent: dict = {}

    class Handler(GATEWAY.GatewayHandler):
        def __init__(self):  # noqa: D401 - no BaseHTTPRequestHandler setup needed
            pass

        def _send_json(self, status, payload, extra_headers=None):
            sent["status"] = status
            sent["payload"] = payload

    Handler()._send_provider_models()
    assert sent["payload"]["node"] == {"bands": {"ram_gb": 64.0}}


# --- remote residency + bands on the receiving side -------------------------

class FakeHandle:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_remote_models_keep_peer_bands_and_are_not_relabelled_as_local(monkeypatch) -> None:
    GATEWAY._remote_cache.clear()
    peer_payload = {
        "object": "list",
        "node": {"bands": {"ram_gb": 128.0, "vram_gb": 24.0}, "host": "omarchy"},
        "data": [{"id": "quality-24gb", "description": "peer model", "metadata": {"role": "main"}}],
    }
    monkeypatch.setattr(GATEWAY, "urlopen", lambda url, timeout=None: FakeHandle(peer_payload))
    monkeypatch.setenv("TURBOFIT_PEERS", "omarchy.example.ts.net:8091")
    monkeypatch.setattr(GATEWAY, "REMOTE_CACHE_TTL_S", 30.0)

    remote = GATEWAY._discover_remote_models()
    assert [m["id"] for m in remote] == ["omarchy.example.ts.net/quality-24gb"]
    assert remote[0]["metadata"]["peer_bands"]["vram_gb"] == 24.0
    assert remote[0]["metadata"]["mode"] == "tailnet"
    assert remote[0]["metadata"]["residency"] == "remote"

    # The local role metadata must not overwrite the remote entry.
    monkeypatch.setattr(
        GATEWAY, "provider_model_metadata",
        lambda: {"main": {"role": "main", "residency": "ready", "mode": "local"}},
    )
    monkeypatch.setattr(GATEWAY, "_discover_local_models", lambda: [])
    catalog = GATEWAY.provider_models()
    remote_entry = next(m for m in catalog if m.get("source") == "remote")
    assert remote_entry["metadata"]["residency"] == "remote"
    assert remote_entry["metadata"]["mode"] == "tailnet"
    assert remote_entry["metadata"]["peer_bands"]["vram_gb"] == 24.0


# --- admission ---------------------------------------------------------------

@pytest.mark.parametrize(
    "address, allowed",
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("192.168.1.20", True),
        ("10.0.0.5", True),
        ("100.68.42.68", True),          # tailnet CGNAT
        ("fd7a:115c:a1e0::1", True),
        ("8.8.8.8", False),              # public internet
        ("not-an-ip", False),
    ],
)
def test_gateway_admits_loopback_lan_and_tailnet_only(address, allowed) -> None:
    assert GATEWAY.client_address_allowed(address) is allowed
