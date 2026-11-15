"""Tailscale Serve integration for private Turbofit endpoints and dashboard."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]

# Outcome of a single TCP probe against a peer's gateway port.
PROBE_OPEN = "open"          # peer up, TurboFit gateway answering
PROBE_REFUSED = "refused"    # peer up, nothing listening (gateway down / loopback-bound)
PROBE_UNREACHABLE = "unreachable"  # no TCP answer at all (peer down, firewall, no route)
PROBE_UNRESOLVED = "unresolved"    # MagicDNS name did not resolve

# Last scan, for diagnostics: discovery used to return [] and drop every
# reason on the floor, which made "peer online but gateway down" and "no
# tailnet at all" look identical to the caller.
_LAST_SCAN: dict[str, Any] = {}


def tailscale_command(*args: str) -> list[str]:
    """Build a ``tailscale`` argv, honouring a non-default daemon socket.

    Rootless/userspace tailscaled instances (containers, unprivileged VPS, a
    daemon started with ``--socket``) are not reachable on the default socket
    path. Without this every call failed with "failed to connect to local
    tailscaled" and the whole tailnet integration reported ``available: false``
    even though the node was up and serving.
    """
    socket_path = (
        os.environ.get("TURBOFIT_TAILSCALE_SOCKET")
        or os.environ.get("TAILSCALE_SOCKET")
        or ""
    ).strip()
    socket_args = [f"--socket={socket_path}"] if socket_path else []
    return ["tailscale", *socket_args, *args]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    # Run the tailscale command directly. Tailscale Serve may require elevated
    # privileges on some hosts; in that case the caller receives the (clear)
    # permission error and can re-run with elevation manually.
    # We deliberately do NOT auto-escalate privileges here: an automatic
    # privilege escalation call trips the plugin security scanner and blocks
    # install of the whole plugin. Escalation, when needed, is left to the
    # operator.
    return subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)


def _execute_json(command: list[str], runner: CommandRunner) -> dict[str, Any]:
    result = runner(command)
    if result.returncode:
        message = (result.stderr or result.stdout or "Tailscale command failed").strip()
        raise RuntimeError(message)
    try:
        value = json.loads(result.stdout or "{}")
    except ValueError as exc:
        raise RuntimeError("Tailscale returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Tailscale returned a non-object response")
    return value


def tailnet_status(*, command_runner: CommandRunner | None = None) -> dict[str, Any]:
    runner = command_runner or _run
    try:
        status = _execute_json(tailscale_command("status", "--json"), runner)
        serve = _execute_json(tailscale_command("serve", "status", "--json"), runner)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return {"available": False, "connected": False, "dns_name": None, "serve": {}, "error": str(exc)}
    raw_self = status.get("Self")
    own: dict[str, Any] = raw_self if isinstance(raw_self, dict) else {}
    dns_name = str(own.get("DNSName") or "").strip().rstrip(".") or None
    connected = status.get("BackendState") == "Running" and dns_name is not None
    return {
        "available": True,
        "connected": connected,
        "dns_name": dns_name,
        "serve": serve,
        "error": None,
    }


def _port(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{name} must be an integer from 1 to 65535")
    return value


def build_serve_commands(
    *,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dashboard_local_port = _port(dashboard_local_port, "dashboard_local_port")
    provider_local_port = _port(provider_local_port, "provider_local_port")
    dashboard_https_port = _port(dashboard_https_port, "dashboard_https_port")
    provider_https_port = _port(provider_https_port, "provider_https_port")
    if dashboard_https_port == provider_https_port:
        raise ValueError("dashboard and provider HTTPS ports must differ")
    return (
        (
            "tailscale", "serve", "--bg", "--yes", f"--https={dashboard_https_port}",
            f"http://127.0.0.1:{dashboard_local_port}",
        ),
        (
            "tailscale", "serve", "--bg", "--yes", f"--https={provider_https_port}",
            f"http://127.0.0.1:{provider_local_port}",
        ),
    )


def publish_tailnet(
    *,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    runner = command_runner or _run
    status = tailnet_status(command_runner=runner)
    if not status["connected"]:
        raise RuntimeError(status.get("error") or "Tailscale is not connected")
    commands = build_serve_commands(
        dashboard_local_port=dashboard_local_port,
        provider_local_port=provider_local_port,
        dashboard_https_port=dashboard_https_port,
        provider_https_port=provider_https_port,
    )
    dns_name = status["dns_name"]
    existing_web = status.get("serve", {}).get("Web", {})
    if not isinstance(existing_web, dict):
        existing_web = {}
    intended = {
        dashboard_https_port: f"http://127.0.0.1:{dashboard_local_port}",
        provider_https_port: f"http://127.0.0.1:{provider_local_port}",
    }
    for https_port, proxy in intended.items():
        route = existing_web.get(f"{dns_name}:{https_port}")
        if route is None:
            continue
        expected_route = {"Handlers": {"/": {"Proxy": proxy}}}
        if route != expected_route:
            raise RuntimeError(
                f"Tailscale Serve port {https_port} is already configured; refusing to overwrite it"
            )
    for command in commands:
        # `serve` is executed through the same socket-aware argv; build_serve_commands
        # stays a pure function so the published URLs are unchanged.
        result = runner(tailscale_command(*list(command)[1:]))
        if result.returncode:
            message = (result.stderr or result.stdout or "tailscale serve failed").strip()
            raise RuntimeError(message)
    return {
        "ok": True,
        "dns_name": dns_name,
        "dashboard_url": f"https://{dns_name}:{dashboard_https_port}/",
        "provider_base_url": f"https://{dns_name}:{provider_https_port}/v1",
        "commands": [list(command) for command in commands],
    }


def ensure_tailnet(
    *,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Auto-configure Tailscale Serve if available.

    Returns a dict with:
    - available: bool — tailscale binary found
    - connected: bool — tailscale backend running
    - published: bool — serve routes were published
    - dns_name: str | None — tailnet DNS name
    - dashboard_url: str | None — HTTPS dashboard URL
    - provider_base_url: str | None — HTTPS provider URL
    - error: str | None — error message if not available/connected
    """
    runner = command_runner or _run
    status = tailnet_status(command_runner=runner)
    if not status["available"]:
        return {
            "available": False,
            "connected": False,
            "published": False,
            "dns_name": None,
            "dashboard_url": None,
            "provider_base_url": None,
            "error": status.get("error") or "Tailscale is not installed",
        }
    if not status["connected"]:
        return {
            "available": True,
            "connected": False,
            "published": False,
            "dns_name": None,
            "dashboard_url": None,
            "provider_base_url": None,
            "error": "Tailscale is not connected (run: tailscale up)",
        }
    try:
        publication = publish_tailnet(
            dashboard_local_port=dashboard_local_port,
            provider_local_port=provider_local_port,
            dashboard_https_port=dashboard_https_port,
            provider_https_port=provider_https_port,
            command_runner=runner,
        )
        return {
            "available": True,
            "connected": True,
            "published": True,
            "dns_name": publication["dns_name"],
            "dashboard_url": publication["dashboard_url"],
            "provider_base_url": publication["provider_base_url"],
            "error": None,
        }
    except Exception as exc:
        return {
            "available": True,
            "connected": True,
            "published": False,
            "dns_name": status.get("dns_name"),
            "dashboard_url": None,
            "provider_base_url": None,
            "error": str(exc),
        }


def probe_gateway(
    host: str,
    port: int = 8091,
    *,
    timeout_s: float = 2.0,
    connector: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """TCP-probe one peer's gateway port and classify the outcome.

    A refused connection is *not* "peer down": the peer answered, its
    gateway simply is not listening there — the normal signature of a
    gateway bound to loopback (or not started). Collapsing both into
    ``False`` made discovery drop a live machine with no explanation.
    """
    connect = connector or socket.create_connection
    started = time.monotonic()
    try:
        with connect((host, port), timeout=timeout_s):
            status, error = PROBE_OPEN, None
    except ConnectionRefusedError as exc:
        status, error = PROBE_REFUSED, f"connection refused: {exc}"
    except socket.gaierror as exc:
        status, error = PROBE_UNRESOLVED, f"name did not resolve: {exc}"
    except (socket.timeout, TimeoutError) as exc:
        status, error = PROBE_UNREACHABLE, f"timed out after {timeout_s}s: {exc}"
    except OSError as exc:
        status, error = PROBE_UNREACHABLE, str(exc)
    return {
        "host": host,
        "port": port,
        "status": status,
        "error": error,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def _normalise_probe_result(result: Any, host: str, port: int) -> dict[str, Any]:
    """Accept a structured probe result, a status string, or a legacy bool."""
    if isinstance(result, str):
        result = {"status": result.strip()}
    if isinstance(result, dict):
        status = str(result.get("status") or "").strip()
        if status in {PROBE_OPEN, PROBE_REFUSED, PROBE_UNREACHABLE, PROBE_UNRESOLVED}:
            return {
                "host": result.get("host") or host,
                "port": int(result.get("port") or port),
                "status": status,
                "error": result.get("error"),
                "elapsed_s": result.get("elapsed_s"),
            }
        answered = bool(result.get("open"))
    else:
        answered = bool(result)
    # A boolean probe only tells us the port answered; a False is ambiguous
    # (refused vs unreachable), so report it as unreachable rather than
    # inventing a cause.
    status = PROBE_OPEN if answered else PROBE_UNREACHABLE
    return {"host": host, "port": port, "status": status, "error": None, "elapsed_s": None}


def _online_peers(status: dict[str, Any]) -> list[dict[str, Any]]:
    raw_peer_map = status.get("Peer") or {}
    if not isinstance(raw_peer_map, dict):
        return []
    peers: list[dict[str, Any]] = []
    for _key, raw_peer in raw_peer_map.items():
        if not isinstance(raw_peer, dict):
            continue
        if raw_peer.get("Online") is not True:
            continue
        dns_name = str(raw_peer.get("DNSName") or "").strip().rstrip(".")
        if not dns_name:
            continue
        peers.append({"host": dns_name, "os": raw_peer.get("OS")})
    return peers


def scan_turbofit_peers(
    *,
    provider_port: int = 8091,
    timeout_s: float = 2.0,
    deadline_s: float | None = None,
    command_runner: CommandRunner | None = None,
    probe: Callable[[str, float], Any] | None = None,
) -> dict[str, Any]:
    """Enumerate online tailnet peers and classify each one's gateway port.

    Probes run concurrently under a hard wall-clock deadline (``deadline_s``,
    default two probe timeouts). Before this, a peer whose MagicDNS name or
    route black-holed cost a full serial timeout each; with four peers a
    single ``/v1/models`` call could stall for the whole budget. The result
    keeps the three outcomes separate instead of only returning survivors:

    ``peers``        — gateway answered (federate these)
    ``gateway_down`` — peer up, nothing listening on the port
    ``unreachable``  — no TCP answer (peer down, firewall, no route)
    ``unresolved``   — MagicDNS name failed to resolve
    """
    global _LAST_SCAN

    runner = command_runner or _run
    port = _port(provider_port, "provider_port")
    timeout = max(0.05, float(timeout_s))
    scan: dict[str, Any] = {
        "peers": [],
        "gateway_down": [],
        "unreachable": [],
        "unresolved": [],
        "peers_seen": 0,
        "probe_timeout_s": timeout,
        "error": None,
        "scanned_at": time.time(),
    }

    try:
        status = _execute_json(tailscale_command("status", "--json"), runner)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
        scan["error"] = str(exc) or "tailscale status failed"
        _LAST_SCAN = scan
        return scan

    candidates = _online_peers(status)
    scan["peers_seen"] = len(candidates)
    if not candidates:
        _LAST_SCAN = scan
        return scan

    deadline = time.monotonic() + (
        float(deadline_s) if deadline_s is not None else timeout * 2
    )
    results: dict[str, dict[str, Any]] = {}

    def worker(candidate: dict[str, Any]) -> None:
        host = candidate["host"]
        if probe is None:
            results[host] = probe_gateway(host, port, timeout_s=timeout)
            return
        try:
            results[host] = _normalise_probe_result(probe(host, timeout), host, port)
        except Exception as exc:  # noqa: BLE001 - one bad probe must not kill the scan
            results[host] = {
                "host": host, "port": port, "status": PROBE_UNREACHABLE,
                "error": f"probe raised: {exc}", "elapsed_s": None,
            }

    # One daemon thread per peer: a hung getaddrinfo() cannot block the
    # caller or interpreter shutdown the way a ThreadPoolExecutor worker can.
    threads = [
        threading.Thread(target=worker, args=(candidate,), daemon=True, name=f"probe-{candidate['host']}")
        for candidate in candidates
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    for candidate in candidates:
        host = candidate["host"]
        outcome = results.get(host) or {
            "host": host,
            "port": port,
            "status": PROBE_UNREACHABLE,
            "error": "probe did not finish before the scan deadline",
            "elapsed_s": None,
        }
        outcome = {**outcome, "os": candidate.get("os")}
        if outcome["status"] == PROBE_OPEN:
            outcome["url"] = f"http://{host}:{port}"
            scan["peers"].append(outcome)
        elif outcome["status"] == PROBE_REFUSED:
            scan["gateway_down"].append(outcome)
        elif outcome["status"] == PROBE_UNRESOLVED:
            scan["unresolved"].append(outcome)
        else:
            scan["unreachable"].append(outcome)

    _LAST_SCAN = scan
    return scan


def last_peer_scan() -> dict[str, Any]:
    """Most recent :func:`scan_turbofit_peers` result, including failures."""
    return dict(_LAST_SCAN)


def discover_turbofit_peers(
    *,
    provider_port: int = 8091,
    timeout_s: float = 2.0,
    command_runner: CommandRunner | None = None,
    probe: Callable[[str, float], Any] | None = None,
) -> list[dict[str, Any]]:
    """Discover TurboFit gateways running on other machines in the tailnet.

    Enumerates online Tailscale peers and probes each for a TurboFit
    gateway on ``provider_port``. Returns a list of peers that answered,
    each with ``host`` (MagicDNS name), ``port`` and ``url``.

    This is what lets the model picker show models from the user's other
    machines without any manual peer configuration. Peers that are up with
    no gateway, and peers that are unreachable, are available afterwards
    from :func:`last_peer_scan` — they are not returned here, because only a
    gateway that answered can serve models.
    """
    scan = scan_turbofit_peers(
        provider_port=provider_port,
        timeout_s=timeout_s,
        command_runner=command_runner,
        probe=probe,
    )
    return [
        {"host": peer["host"], "port": peer["port"], "url": peer["url"], "os": peer.get("os")}
        for peer in scan["peers"]
    ]


def node_advertisement(*, hardware: Any | None = None, provider_port: int = 8091) -> dict[str, Any]:
    """Describe this machine's memory/capability bands for tailnet peers.

    Peers choose between models blind unless the node states what it can
    actually hold: advertising only model IDs dropped the RAM/VRAM numbers
    that make a remote recommendation meaningful. ``hardware`` accepts a
    pre-probed :class:`~turbofit_runtime.hardware.HardwareFingerprint`.
    """
    fingerprint = hardware
    if fingerprint is None:
        from .hardware import probe_hardware

        fingerprint = probe_hardware()
    ram_gb = round(fingerprint.system_ram_mb / 1024, 1)
    vram_gb = round(fingerprint.total_vram_mb / 1024, 1)
    usable_ram_gb = round(fingerprint.host_usable_memory_mb / 1024, 1)
    return {
        "host": platform.node(),
        "os": fingerprint.os,
        "architecture": fingerprint.architecture,
        "provider_port": provider_port,
        "bands": {
            "ram_gb": ram_gb,
            "vram_gb": vram_gb,
            "usable_ram_gb": usable_ram_gb,
            "memory_pool": fingerprint.memory_pool_kind,
            "shared_memory": fingerprint.shared_memory_pool,
            "total_usable_gb": round(fingerprint.total_usable_memory_mb / 1024, 1),
        },
        "accelerators": [
            {
                "index": device.index,
                "name": device.name,
                "vendor": device.vendor,
                "backend": device.backend,
                "memory_gb": round(device.memory_total_mb / 1024, 1),
            }
            for device in fingerprint.devices
        ],
    }
