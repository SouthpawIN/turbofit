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
) -> dict[str, Any]:
    from hermes_cli.config import load_config, save_config

    with _CONFIG_LOCK:
        configured = configure_hermes(
            load_config(),
            primary=primary,
            fallback=fallback,
            base_url=base_url,
        )
        save_config(configured, merge_existing=False)
        selected = select_profile(profile) if profile else None
    return {"ok": True, "configured": True, "selection": selected, "restart_required": True}


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    try:
        from hermes_cli.config import load_config

        return json.dumps(status_snapshot(load_config()))
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def handle_configure(args: dict[str, Any], **_: Any) -> str:
    try:
        result = apply_configuration(
            primary=bool(args.get("primary", False)),
            fallback=args.get("fallback") if "fallback" in args else None,
            profile=str(args["profile"]) if args.get("profile") else None,
            base_url=str(args["base_url"]) if args.get("base_url") else None,
        )
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)})
