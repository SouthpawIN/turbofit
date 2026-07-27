"""Turbofit — adaptive local inference provider plugin for Hermes Agent."""
from __future__ import annotations

import json
from pathlib import Path

from . import schemas
from .plugin_tools import handle_configure, handle_status


def check_available() -> bool:
    return True


def _slash_turbofit(raw_args: str) -> str:
    action = str(raw_args or "status").strip().lower()
    if action in {"", "status"}:
        return handle_status({})
    if action in {"setup", "configure"}:
        return json.dumps(
            {
                "ok": True,
                "message": (
                    "Open `hermes dashboard` and select the Turbofit tab, or call "
                    "turbofit_configure with primary/fallback/profile options."
                ),
            }
        )
    return json.dumps({"ok": False, "error": "usage: /turbofit [status|setup]"})


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
        description="Inspect or set up the Turbofit adaptive local provider",
    )
    ctx.register_skill("turbofit", Path(__file__).parent / "SKILL.md")


__all__ = ["register"]
