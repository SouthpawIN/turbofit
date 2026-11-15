"""Gateway service installation across platforms.

The gateway must be reachable on the tailnet for federated model
discovery, and on macOS the LaunchAgent must carry an explicit PATH or
the tailscale CLI is invisible and discovery silently returns nothing.
"""
from __future__ import annotations

import importlib.util
import plistlib
from importlib.machinery import SourceFileLoader
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/install-controller-service"
    loader = SourceFileLoader("install_controller_service", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_gateway_unit_binds_for_tailnet_reachability() -> None:
    unit = _module().gateway_unit_text()
    # Without this no peer can ever discover the gateway.
    assert "TURBOFIT_GATEWAY_HOST=0.0.0.0" in unit
    assert "Restart=on-failure" in unit


def test_launchd_plist_sets_path_for_tailscale_cli() -> None:
    plist = _module().gateway_launchd_plist()
    env = plist["EnvironmentVariables"]
    # launchd's default PATH omits /usr/local/bin, where tailscale lives.
    assert "/usr/local/bin" in env["PATH"]
    assert env["TURBOFIT_PEER_DISCOVERY"] == "1"
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True


def test_launchd_plist_binds_for_tailnet_reachability() -> None:
    # The systemd unit binds 0.0.0.0; a loopback-only launchd bind hides
    # the Mac from tailnet peer discovery (fleet can't list its models).
    plist = _module().gateway_launchd_plist()
    assert plist["EnvironmentVariables"]["TURBOFIT_GATEWAY_HOST"] == "0.0.0.0"


def test_launchd_plist_is_serializable_and_targets_the_gateway() -> None:
    plist = _module().gateway_launchd_plist()
    dumped = plistlib.dumps(plist)
    assert plistlib.loads(dumped)["Label"] == "com.hermes.turbofit-gateway"
    assert any("turbofit-gateway.py" in part for part in plist["ProgramArguments"])


def test_controller_launchd_plist_targets_controller_and_keeps_alive() -> None:
    # Fresh macOS installs must come up with BOTH agents: without the
    # controller agent, no local rung ever starts and the fleet degrades
    # to API-only.
    module = _module()
    plist = module.controller_launchd_plist()
    assert plist["Label"] == "com.hermes.turbofit-controller"
    assert any("turbofit-controller" in part for part in plist["ProgramArguments"])
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    env = plist["EnvironmentVariables"]
    assert "/usr/local/bin" in env["PATH"]
    assert "src" in env["PYTHONPATH"]
    dumped = plistlib.dumps(plist)
    assert plistlib.loads(dumped)["Label"] == "com.hermes.turbofit-controller"


def test_ensure_linger_reports_and_warns() -> None:
    import subprocess as _sp

    module = _module()

    def completed(command, stdout="", returncode=0, err=""):
        return _sp.CompletedProcess(command, returncode, stdout, err)

    # enable succeeds -> True
    calls = []
    def ok(command):
        calls.append(list(command))
        return completed(command)
    assert module.ensure_linger(command_runner=ok) is True
    assert calls[0][0] == "loginctl" and calls[0][1] == "enable-linger"

    # enable fails but state already Linger=yes -> True
    def already_on(command):
        if command[1] == "enable-linger":
            return completed(command, returncode=1, err="polkit: not authorized")
        return completed(command, stdout="Linger=yes")
    assert module.ensure_linger(command_runner=already_on) is True

    # both fail -> False + actionable warning
    def denied(command):
        if command[1] == "enable-linger":
            return completed(command, returncode=1, err="polkit: not authorized")
        return completed(command, stdout="Linger=no")
    assert module.ensure_linger(command_runner=denied) is False

    # loginctl missing entirely -> False, no crash
    def no_loginctl(command):
        raise FileNotFoundError("loginctl")
    assert module.ensure_linger(command_runner=no_loginctl) is False


def test_autoconfigure_tailnet_is_best_effort_and_never_raises() -> None:
    import json as _json
    import subprocess as _sp

    module = _module()

    def completed(command, stdout=""):
        return _sp.CompletedProcess(command, 0, stdout=stdout, stderr="")

    # tailscale connected and healthy -> publishes and reports URLs
    def healthy(command):
        if command == ["tailscale", "status", "--json"]:
            return completed(command, _json.dumps({
                "BackendState": "Running",
                "Self": {"DNSName": "host.ts.net."},
            }))
        if command == ["tailscale", "serve", "status", "--json"]:
            return completed(command, "{}")
        return completed(command)
    result = module.autoconfigure_tailnet(command_runner=healthy)
    assert result["available"] is True
    assert result["connected"] is True
    assert result["published"] is True
    assert result["provider_base_url"] == "https://host.ts.net:9443/v1"

    # tailscale binary missing -> warning, install continues
    def missing(command):
        raise FileNotFoundError("tailscale")
    result = module.autoconfigure_tailnet(command_runner=missing)
    assert result["available"] is False
    assert result["published"] is False

    # serve command fails mid-publish -> published=False, no crash
    def flaky(command):
        if command == ["tailscale", "status", "--json"]:
            return completed(command, _json.dumps({
                "BackendState": "Running",
                "Self": {"DNSName": "host.ts.net."},
            }))
        if command == ["tailscale", "serve", "status", "--json"]:
            return completed(command, "{}")
        return _sp.CompletedProcess(command, 1, stdout="", stderr="serve: already in use")
    result = module.autoconfigure_tailnet(command_runner=flaky)
    assert result["published"] is False
    assert "serve: already in use" in (result.get("error") or "")
