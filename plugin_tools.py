"""Hermes-facing configuration and status helpers for the Turbofit plugin."""
from __future__ import annotations

import copy
import ipaddress
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import urlopen

from turbofit_runtime.tailnet import publish_tailnet, tailnet_status


DEFAULT_BASE_URL = "http://127.0.0.1:8091/v1"
PLUGIN_ROOT = Path(__file__).resolve().parent
SELECTION_PATH = Path(
    os.getenv(
        "TURBOFIT_SELECTION_STATE",
        Path.home() / ".config" / "turbofit" / "selection.json",
    )
)
RUNTIME_STATE_PATH = Path(
    os.getenv(
        "TURBOFIT_RUNTIME_STATE",
        Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "turbofit"
        / "runtime-state.json",
    )
)
_CONFIG_LOCK = threading.RLock()


def _is_local_or_tailnet(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".ts.net"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    tailnet = ipaddress.ip_network("100.64.0.0/10")
    return address.is_loopback or address in tailnet


def _validated_base_url(value: str | None) -> str:
    base_url = str(value or DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    if parsed.scheme == "http" and not _is_local_or_tailnet(parsed.hostname):
        raise ValueError(
            "plain HTTP base_url must use loopback or a Tailscale address; use HTTPS otherwise"
        )
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def provider_definition(base_url: str | None = None) -> dict[str, Any]:
    """Return the canonical named custom-provider definition."""
    return {
        "name": "turbofit",
        "base_url": _validated_base_url(base_url),
        "api_key": "not-needed",
        "api_mode": "chat_completions",
        "models": {"auto": {}, "active:main": {}, "active:aux": {}},
    }


def configure_hermes(
    config: Mapping[str, Any],
    *,
    primary: bool = False,
    fallback: bool | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return a copied Hermes config with Turbofit registered idempotently."""
    updated = copy.deepcopy(dict(config))
    provider = provider_definition(base_url)

    existing = updated.get("custom_providers")
    providers = list(existing) if isinstance(existing, list) else []
    providers = [
        item
        for item in providers
        if not (isinstance(item, Mapping) and str(item.get("name", "")).lower() == "turbofit")
    ]
    providers.append(provider)
    updated["custom_providers"] = providers

    if primary:
        raw_model = updated.get("model")
        model: dict[str, Any] = copy.deepcopy(dict(raw_model)) if isinstance(raw_model, Mapping) else {}
        model["provider"] = "custom:turbofit"
        model["default"] = "auto"
        updated["model"] = model

    if fallback is not None:
        raw_chain = updated.get("fallback_providers")
        chain = list(raw_chain) if isinstance(raw_chain, list) else []
        chain = [
            item
            for item in chain
            if not (
                isinstance(item, Mapping)
                and str(item.get("provider", "")).lower() in {"custom:turbofit", "turbofit"}
            )
        ]
        if fallback:
            chain.append(
                {
                    "provider": "custom:turbofit",
                    "model": "auto",
                    "base_url": provider["base_url"],
                    "api_mode": "chat_completions",
                }
            )
        updated["fallback_providers"] = chain

    return updated


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _provider_flags(config: Mapping[str, Any]) -> tuple[bool, bool, str]:
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    primary = str(model.get("provider", "")).lower() == "custom:turbofit"
    fallback = any(
        isinstance(item, Mapping)
        and str(item.get("provider", "")).lower() in {"custom:turbofit", "turbofit"}
        for item in (config.get("fallback_providers") or [])
    )
    providers = config.get("custom_providers") or []
    endpoint = next(
        (
            str(item.get("base_url"))
            for item in providers
            if isinstance(item, Mapping) and str(item.get("name", "")).lower() == "turbofit"
        ),
        DEFAULT_BASE_URL,
    )
    return primary, fallback, endpoint


def gateway_health(base_url: str = DEFAULT_BASE_URL, timeout: float = 1.5) -> dict[str, Any]:
    endpoint = f"{_validated_base_url(base_url)}/models"
    try:
        with urlopen(endpoint, timeout=timeout) as response:  # noqa: S310 - user-configured local/Tailscale endpoint
            decoded = json.loads(response.read().decode("utf-8"))
        payload = decoded if isinstance(decoded, dict) else {}
        data = payload.get("data")
        ids = [item.get("id") for item in data if isinstance(item, dict)] if isinstance(data, list) else []
        return {"reachable": True, "endpoint": endpoint, "models": ids}
    except Exception as exc:
        return {"reachable": False, "endpoint": endpoint, "error": str(exc)}


def status_snapshot(config: Mapping[str, Any], *, probe: bool = True) -> dict[str, Any]:
    primary, fallback, endpoint = _provider_flags(config)
    selection = _load_json(SELECTION_PATH)
    runtime = _load_json(RUNTIME_STATE_PATH)
    return {
        "ok": True,
        "provider": {
            "registered": any(
                isinstance(item, Mapping) and str(item.get("name", "")).lower() == "turbofit"
                for item in (config.get("custom_providers") or [])
            ),
            "primary": primary,
            "fallback": fallback,
            "base_url": endpoint,
        },
        "selection": selection,
        "runtime": runtime,
        "gateway": gateway_health(endpoint) if probe else {"reachable": None, "endpoint": endpoint},
        "tailnet": tailnet_status(),
        "platform": {"os": os.name, "sys_platform": sys.platform},
    }


def select_profile(profile: str) -> dict[str, Any]:
    requested = str(profile or "auto").strip()
    script = PLUGIN_ROOT / "scripts" / "turbofit-runtime"
    if not script.is_file():
        raise FileNotFoundError(f"missing runtime selector: {script}")
    result = subprocess.run(
        [str(script), "set", requested],
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    try:
        payload = json.loads(result.stdout or result.stderr)
    except ValueError:
        payload = {"output": (result.stdout or result.stderr).strip()}
    if result.returncode:
        raise RuntimeError(payload.get("error") or payload.get("output") or "profile selection failed")
    return payload


def apply_configuration(
    *,
    primary: bool,
    fallback: bool | None,
    profile: str | None,
    base_url: str | None,
    publish_tailnet_routes: bool = False,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
) -> dict[str, Any]:
    from hermes_cli.config import load_config, save_config

    with _CONFIG_LOCK:
        selected = select_profile(profile) if profile else None
        publication = (
            publish_tailnet(
                dashboard_local_port=dashboard_local_port,
                provider_local_port=provider_local_port,
                dashboard_https_port=dashboard_https_port,
                provider_https_port=provider_https_port,
            )
            if publish_tailnet_routes
            else None
        )
        effective_base_url = publication["provider_base_url"] if publication else base_url
        configured = configure_hermes(
            load_config(),
            primary=primary,
            fallback=fallback,
            base_url=effective_base_url,
        )
        save_config(configured, merge_existing=False)
    return {
        "ok": True,
        "configured": True,
        "selection": selected,
        "tailnet": publication,
        "restart_required": True,
    }


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    try:
        from hermes_cli.config import load_config

        return json.dumps(status_snapshot(load_config()))
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def handle_configure(args: dict[str, Any], **_: Any) -> str:
    try:
        primary = args.get("primary", False)
        fallback = args.get("fallback") if "fallback" in args else None
        publish_routes = args.get("publish_tailnet", False)
        if (
            not isinstance(primary, bool)
            or (fallback is not None and not isinstance(fallback, bool))
            or not isinstance(publish_routes, bool)
        ):
            raise ValueError("primary, fallback, and publish_tailnet must be booleans")
        profile = args.get("profile")
        base_url = args.get("base_url")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("profile must be a string")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("base_url must be a string")
        ports = {
            name: args.get(name, default)
            for name, default in {
                "dashboard_local_port": 9127,
                "provider_local_port": 8091,
                "dashboard_https_port": 9444,
                "provider_https_port": 9443,
            }.items()
        }
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ports.values()):
            raise ValueError("Tailscale ports must be integers")
        result = apply_configuration(
            primary=primary,
            fallback=fallback,
            profile=profile,
            base_url=base_url,
            publish_tailnet_routes=publish_routes,
            **ports,
        )
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})
