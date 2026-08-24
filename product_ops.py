"""Device update and manual ladder-shift operations for Turbofit."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from . import plugin_tools
except ImportError:  # pytest and scripts import this as a top-level module
    import plugin_tools

TURBOFIT_GIT = "https://github.com/SouthpawIN/turbofit.git"
PREFERENCE_ALIASES = {
    "intelligence": "intelligence",
    "quality": "intelligence",
    "balanced": "balanced",
    "context": "balanced",
    "speed": "speed",
}


def update_products(*, hermes_home: Path | None = None) -> dict[str, Any]:
    """Pull latest Turbofit + Sirvir onto this machine and refresh Desktop."""
    executable = shutil.which("hermes")
    if not executable:
        raise FileNotFoundError("hermes executable is not available")
    environment = os.environ.copy()
    if hermes_home is not None:
        environment["HERMES_HOME"] = str(hermes_home)
    plugin = subprocess.run(
        [executable, "plugins", "update", "turbofit"],
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
        env=environment,
    )
    if plugin.returncode:
        plugin = subprocess.run(
            [executable, "plugins", "install", "--enable", TURBOFIT_GIT],
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
            env=environment,
        )
        if plugin.returncode:
            raise RuntimeError((plugin.stderr or plugin.stdout or "Turbofit plugin update failed").strip())
    desktop = plugin_tools.install_desktop_plugin(hermes_home=hermes_home)
    sirvir = plugin_tools.install_sirvir_profile(hermes_home=hermes_home)
    models = plugin_tools.ensure_recommended_models()
    slash = plugin_tools.activate_slash_commands(hermes_home=hermes_home)
    return {
        "ok": True,
        "updated": True,
        "turbofit_plugin": {
            "ok": True,
            "output": (plugin.stdout or plugin.stderr or "").strip(),
        },
        "desktop": desktop,
        "sirvir": sirvir,
        "models": models,
        "slash_commands": slash,
        "message": (
            "Turbofit plugin, Desktop surface, and Sirvir updated. "
            "Reload Desktop plugins and open Turbofit. Start a new session for provider changes."
        ),
    }


def _ladder(preference: str = "intelligence") -> list[dict[str, Any]]:
    snapshot = plugin_tools.recommendation_snapshot(preference)
    rows = list((snapshot.get("recommendations") or {}).get(preference) or [])
    return [dict(row) for row in rows if isinstance(row, Mapping) and row.get("profile")]


def _current_profile_id() -> str:
    selection = plugin_tools._load_json(plugin_tools.SELECTION_PATH) or {}
    requested = str(selection.get("requested") or selection.get("profile_id") or "auto")
    return requested[7:] if requested.startswith("manual-") else requested


def _apply(profile_id: str, *, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = plugin_tools.select_profile(profile_id)
    payload = {
        "ok": True,
        "shifted": True,
        "profile": profile_id,
        "reason": reason,
        "selection": selected,
    }
    if extra:
        payload.update(extra)
    return payload


def shift_configuration(target: str) -> dict[str, Any]:
    """Move one step on the measured ladder, apply a preference, or select a model."""
    raw = str(target or "").strip()
    if not raw:
        raise ValueError("usage: /turbofit shift up|down|<model>|intelligence|balanced|speed")
    key = raw.lower()
    if key in {"up", "+", "smarter", "heavier"}:
        return _shift_ladder("up")
    if key in {"down", "-", "lighter", "smaller"}:
        return _shift_ladder("down")
    if key in PREFERENCE_ALIASES:
        return _shift_preference(PREFERENCE_ALIASES[key])
    return _shift_model(raw)


def _shift_ladder(direction: str) -> dict[str, Any]:
    rows = _ladder("intelligence")
    if not rows:
        raise RuntimeError("no evidence-backed ladder on this machine yet; run /turbofit scan")
    ids = [str(row["profile"]) for row in rows]
    current = _current_profile_id()
    if current in ids:
        index = ids.index(current)
        next_index = index - 1 if direction == "up" else index + 1
        if not 0 <= next_index < len(ids):
            edge = "smartest" if direction == "up" else "lightest"
            return {
                "ok": False,
                "shifted": False,
                "error": f"already at the {edge} measured configuration",
                "profile": current,
                "ladder": ids,
            }
        return _apply(
            ids[next_index],
            reason=f"shift {direction} using the intelligence ladder",
            extra={"from": current, "ladder": ids, "index": next_index},
        )
    chosen = ids[0] if direction == "up" else ids[-1]
    return _apply(
        chosen,
        reason=f"shift {direction} from unlisted selection onto the measured ladder",
        extra={"from": current, "ladder": ids},
    )


def _shift_preference(preference: str) -> dict[str, Any]:
    rows = _ladder(preference)
    if not rows:
        raise RuntimeError(f"no {preference} configuration fits this machine")
    return _apply(
        str(rows[0]["profile"]),
        reason=f"shift to the recommended {preference} combination",
        extra={"preference": preference},
    )


def _shift_model(query: str) -> dict[str, Any]:
    needle = query.lower()
    combinations = plugin_tools.combination_snapshot()["combinations"]
    matches: list[Mapping[str, Any]] = []
    fields = (
        "profile",
        "main",
        "main_name",
        "main_catalog_id",
        "main_quantization",
        "aux",
        "aux_name",
        "aux_catalog_id",
    )
    for row in combinations:
        if not isinstance(row, Mapping) or not row.get("fit"):
            continue
        haystack = " ".join(str(row.get(field) or "") for field in fields).lower()
        if needle in haystack:
            matches.append(row)
    if not matches:
        for row in _ladder("intelligence"):
            haystack = " ".join(str(row.get(field) or "") for field in ("profile", "main", "aux_mode")).lower()
            if needle in haystack:
                matches.append(row)
    if not matches:
        raise ValueError(f"no fitting combination matches {query!r}")
    ladder = [str(row["profile"]) for row in _ladder("intelligence")]

    def rank(row: Mapping[str, Any]) -> int:
        profile = str(row.get("profile") or "")
        return ladder.index(profile) if profile in ladder else 10_000

    chosen = sorted(matches, key=rank)[0]
    return _apply(
        str(chosen["profile"]),
        reason=f"shift to the recommended combination for {query}",
        extra={"match": chosen.get("profile"), "query": query},
    )


def serve_tailnet(
    *,
    dashboard_local_port: int = 9127,
    provider_local_port: int = 8091,
    dashboard_https_port: int = 9444,
    provider_https_port: int = 9443,
) -> dict[str, Any]:
    """Publish the local Turbofit /v1 gateway on the private tailnet."""
    from hermes_cli.config import load_config, save_config

    publication = plugin_tools.publish_tailnet(
        dashboard_local_port=dashboard_local_port,
        provider_local_port=provider_local_port,
        dashboard_https_port=dashboard_https_port,
        provider_https_port=provider_https_port,
    )
    with plugin_tools._CONFIG_LOCK:
        updated = plugin_tools.configure_hermes(
            load_config(),
            base_url=publication["provider_base_url"],
        )
        save_config(updated, merge_existing=False)
    return {
        "ok": True,
        "served": True,
        **publication,
        "message": (
            f"Models are on your tailnet at {publication['provider_base_url']}. "
            "Other Tailscale devices use that OpenAI base URL. Serve is private; Funnel is never used."
        ),
    }


def serve_status() -> dict[str, Any]:
    status = plugin_tools.tailnet_status()
    return {"ok": bool(status.get("connected")), **status}


def smoke_local_runtime(*, timeout_seconds: float = 300.0) -> dict[str, Any]:
    """Slash/Desktop loopback smoke. Does not shift or promote."""
    dashboard = Path(__file__).resolve().parent / "dashboard"
    if str(dashboard) not in sys.path:
        sys.path.insert(0, str(dashboard))
    from smoke_ops import smoke_local_runtime as run_smoke

    return run_smoke(timeout_seconds=timeout_seconds)
