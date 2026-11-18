"""Serving on the tailnet is the default state, not a separate opt-in step.

Every TurboFit install publishes its endpoint so other nodes on the network can
reach the model without bespoke configuration. When Tailscale is missing or down
the provider still ends up configured on loopback — connected either way, with
the reason recorded rather than a silent failure.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_apply_configuration_publishes_by_default() -> None:
    import plugin_tools

    signature = inspect.signature(plugin_tools.apply_configuration)
    assert signature.parameters["publish_tailnet_routes"].default is True


def test_cli_configure_publishes_by_default() -> None:
    source = (ROOT / "plugin_tools.py").read_text(encoding="utf-8")
    assert 'args.get("publish_tailnet", True)' in source


def test_dashboard_api_publishes_by_default() -> None:
    source = (ROOT / "dashboard" / "plugin_api.py").read_text(encoding="utf-8")
    assert 'body.get("publish_tailnet", True)' in source


def test_dashboard_toggle_starts_enabled() -> None:
    source = (ROOT / "dashboard" / "dist" / "index.js").read_text(encoding="utf-8")
    assert "const [publishTailnet, setPublishTailnet] = useState(true);" in source


def test_unpublished_tailnet_falls_back_to_loopback_not_none() -> None:
    """A missing Tailscale must not leave the provider without a base URL."""
    source = (ROOT / "plugin_tools.py").read_text(encoding="utf-8")
    assert '(publication or {}).get("provider_base_url") or base_url' in source


def test_ensure_tailnet_reports_why_it_did_not_publish() -> None:
    from turbofit_runtime.tailnet import ensure_tailnet

    def runner(*args, **kwargs):  # tailscale binary absent
        raise FileNotFoundError("tailscale")

    result = ensure_tailnet(command_runner=runner)
    assert result["published"] is False
    assert result["provider_base_url"] is None
    assert result["error"]


def test_explicit_opt_out_is_still_honoured() -> None:
    """Default-on must not become forced-on: a user can still say no."""
    source = (ROOT / "tests" / "test_plugin.py").read_text(encoding="utf-8")
    assert '"publish_tailnet": "false"' in source
