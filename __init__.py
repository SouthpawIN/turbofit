"""Turbofit — adaptive local inference provider plugin for Hermes Agent."""
from __future__ import annotations

import json
from pathlib import Path
import sys

_PLUGIN_SRC = Path(__file__).resolve().parent / "src"
if str(_PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_SRC))

from . import schemas
from .plugin_tools import (
    handle_configure,
    handle_status,
    hardware_tier_snapshot,
    launch_setup_screen,
    recommendation_snapshot,
)


def check_available() -> bool:
    return True


def _slash_turbofit(raw_args: str) -> str:
    action = str(raw_args or "").strip().lower()
    if action == "status":
        return handle_status({})
    if action in {"", "scan", "rescan", "recommend"}:
        try:
            return json.dumps(recommendation_snapshot())
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})
    if action in {"intelligence", "quality", "balanced", "context", "speed"}:
        try:
            return json.dumps(recommendation_snapshot(action))
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})
    if action in {"tiers", "hardware", "hardware-tiers"}:
        try:
            return json.dumps(hardware_tier_snapshot())
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})
    if action in {"setup", "configure"}:
        try:
            return json.dumps({
                "ok": True,
                "setup": launch_setup_screen(),
                "message": "Hermes Dashboard launched. Open Turbofit to configure the provider, fallbacks, runtimes, and multimodal models.",
            })
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})
    return json.dumps({
        "ok": False,
        "error": "usage: /turbofit [scan|status|tiers|intelligence|balanced|speed|setup]",
    })


def register(ctx) -> None:
    """Register Turbofit tools, slash command, and bundled operator skill."""
    ctx.register_tool(
        name="turbofit_status",
        toolset="turbofit",
        schema=schemas.TURBOFIT_STATUS,
        handler=handle_status,
        check_fn=check_available,
        emoji="⚡",
    )
    ctx.register_tool(
        name="turbofit_configure",
        toolset="turbofit",
        schema=schemas.TURBOFIT_CONFIGURE,
        handler=handle_configure,
        check_fn=check_available,
        emoji="⚡",
    )
    ctx.register_command(
        "turbofit",
        _slash_turbofit,
        description="Rescan hardware and recommend Turbofit model configurations",
        args_hint="[scan|status|tiers|intelligence|balanced|speed|setup]",
    )
    ctx.register_skill("turbofit", Path(__file__).parent / "SKILL.md")


__all__ = ["register"]
