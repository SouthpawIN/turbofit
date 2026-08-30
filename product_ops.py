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
    from hermes_cli.config import load_config

    publication = plugin_tools.publish_tailnet(
        dashboard_local_port=dashboard_local_port,
        provider_local_port=provider_local_port,
        dashboard_https_port=dashboard_https_port,
        provider_https_port=provider_https_port,
    )
    with plugin_tools._CONFIG_LOCK:
        canonical = plugin_tools.configure_hermes(
            load_config(),
            base_url=publication["provider_base_url"],
        )
        # Fan the tailnet URL to every profile home (same fix as
        # plugin_tools.apply_configuration — see _save_configuration_to_all_homes).
        plugin_tools._save_configuration_to_all_homes(
            canonical,
            base_url=publication["provider_base_url"],
        )
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


def _artifact_rows() -> list[dict[str, Any]]:
    import json

    payload = json.loads((Path(__file__).resolve().parent / "references/artifact-manifest.json").read_text())
    return [dict(item) for item in payload.get("artifacts") or [] if isinstance(item, Mapping)]


def _family_files(family: str) -> list[dict[str, Any]]:
    from importlib.machinery import SourceFileLoader
    import importlib.util

    script = Path(__file__).resolve().parent / "scripts" / "download-artifacts"
    loader = SourceFileLoader("turbofit_download_artifacts", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = module.model_root()
    found: list[dict[str, Any]] = []
    for item in _artifact_rows():
        families = [str(value) for value in item.get("families") or []]
        if family not in families:
            continue
        destination = module.safe_destination(root, str(item["destination"]))
        present = destination.is_file()
        found.append(
            {
                "family": family,
                "destination": str(item["destination"]),
                "path": str(destination),
                "present": present,
                "size_bytes": destination.stat().st_size if present else 0,
            }
        )
    return found


def local_model_replacement() -> dict[str, Any]:
    """Offer archive/delete when Check recommends a different installed main."""
    families = plugin_tools.recommended_artifact_families()
    recommended = families[0] if families else ""
    installed: list[str] = []
    seen: set[str] = set()
    for item in _artifact_rows():
        for family in item.get("families") or []:
            name = str(family)
            if name in seen:
                continue
            seen.add(name)
            if any(row["present"] for row in _family_files(name)):
                installed.append(name)
    outgoing = [name for name in installed if name != recommended]
    current = outgoing[0] if outgoing else ""
    present = _family_files(current) if current else []
    present = [row for row in present if row["present"]]
    offer = bool(recommended and current and present)
    return {
        "ok": True,
        "recommended_main": recommended,
        "current_main": current,
        "offer": offer and bool(present),
        "from_family": current if offer else "",
        "to_family": recommended if offer else "",
        "title": "New model recommended",
        "prompt": (
            f"Check now recommends {recommended} instead of {current}. "
            "Archive or delete the old weights to free disk, or keep both."
            if offer
            else ""
        ),
        "files": present,
        "bytes": sum(int(row["size_bytes"]) for row in present),
    }


def retire_local_model(family: str, action: str) -> dict[str, Any]:
    """Archive or delete an installed model family that is no longer the recommendation."""
    if action not in {"archive", "delete"}:
        raise ValueError("action must be archive or delete")
    token = str(family or "").strip()
    if not token:
        raise ValueError("family is required")
    replacement = local_model_replacement()
    if token == replacement.get("recommended_main"):
        raise ValueError("refusing to retire the currently recommended model")
    rows = [row for row in _family_files(token) if row["present"]]
    if not rows:
        raise ValueError(f"no installed files for {token}")
    archive_root = Path.home() / ".local/share/turbofit/archive" / token
    moved: list[str] = []
    deleted: list[str] = []
    for row in rows:
        source = Path(row["path"])
        if action == "delete":
            source.unlink()
            deleted.append(str(source))
            continue
        archive_root.mkdir(parents=True, exist_ok=True)
        target = archive_root / source.name
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        moved.append(str(target))
    return {
        "ok": True,
        "retired": True,
        "action": action,
        "family": token,
        "archived": moved,
        "deleted": deleted,
        "message": (
            f"Archived {token} under {archive_root}."
            if action == "archive"
            else f"Deleted {len(deleted)} file(s) for {token}."
        ),
    }
