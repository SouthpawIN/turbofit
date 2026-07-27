from __future__ import annotations

import json
import subprocess

import pytest

from turbofit_runtime.tailnet import build_serve_commands, publish_tailnet, tailnet_status


def completed(command: list[str], stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def test_tailnet_status_reports_dns_and_existing_serve_routes() -> None:
    calls: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command == ["tailscale", "status", "--json"]:
            return completed(command, json.dumps({"BackendState": "Running", "Self": {"DNSName": "host.example.ts.net."}}))
        return completed(command, json.dumps({"TCP": {"443": {"HTTPS": True}}, "Web": {}}))

    status = tailnet_status(command_runner=run)

    assert status["available"] is True
    assert status["connected"] is True
    assert status["dns_name"] == "host.example.ts.net"
    assert status["serve"]["TCP"]["443"]["HTTPS"] is True
    assert calls == [["tailscale", "status", "--json"], ["tailscale", "serve", "status", "--json"]]


def test_build_serve_commands_uses_distinct_https_ports_without_shell() -> None:
    commands = build_serve_commands(
        dashboard_local_port=9127,
        provider_local_port=8091,
        dashboard_https_port=9444,
        provider_https_port=9443,
    )

    assert commands == (
        ("tailscale", "serve", "--bg", "--yes", "--https=9444", "http://127.0.0.1:9127"),
        ("tailscale", "serve", "--bg", "--yes", "--https=9443", "http://127.0.0.1:8091"),
    )


def test_publish_refuses_to_overwrite_an_unrelated_existing_serve_route() -> None:
    calls: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command == ["tailscale", "status", "--json"]:
            return completed(command, json.dumps({"BackendState": "Running", "Self": {"DNSName": "host.example.ts.net."}}))
        if command == ["tailscale", "serve", "status", "--json"]:
            return completed(command, json.dumps({
                "Web": {"host.example.ts.net:9444": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}}}
            }))
        return completed(command)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        publish_tailnet(command_runner=run)

    assert len(calls) == 2


def test_publish_tailnet_returns_shareable_dashboard_and_provider_urls() -> None:
    calls: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command == ["tailscale", "status", "--json"]:
            return completed(command, json.dumps({"BackendState": "Running", "Self": {"DNSName": "host.example.ts.net."}}))
        if command == ["tailscale", "serve", "status", "--json"]:
            return completed(command, "{}")
        return completed(command)

    result = publish_tailnet(command_runner=run)

    assert result["dashboard_url"] == "https://host.example.ts.net:9444/"
    assert result["provider_base_url"] == "https://host.example.ts.net:9443/v1"
    assert calls[-2:] == [
        ["tailscale", "serve", "--bg", "--yes", "--https=9444", "http://127.0.0.1:9127"],
        ["tailscale", "serve", "--bg", "--yes", "--https=9443", "http://127.0.0.1:8091"],
    ]
