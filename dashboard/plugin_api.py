"""Backend API for the Turbofit Hermes dashboard extension."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
for candidate in (PLUGIN_ROOT, PLUGIN_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from plugin_tools import apply_configuration, status_snapshot  # noqa: E402


router = APIRouter()


def _config() -> dict[str, Any]:
    from hermes_cli.config import load_config

    return load_config()


def _profile_rows() -> list[dict[str, Any]]:
    from turbofit_runtime.hardware import probe_hardware
    from turbofit_runtime.runtime_cli import run
    from turbofit_runtime.selection import ProfileCatalog

    import io

    catalog = ProfileCatalog.from_paths(sorted((PLUGIN_ROOT / "runtime-profiles").glob("*gb.yaml")))
    output = io.StringIO()
    code = run(
        ["list"],
        catalog=catalog,
        hardware=probe_hardware(),
        selection_path=Path.home() / ".config" / "turbofit" / "selection.json",
        output=output,
    )
    payload = json.loads(output.getvalue())
    if code:
        raise RuntimeError(payload.get("error", "failed to list profiles"))
    return payload


@router.get("/status")
async def get_status() -> dict[str, Any]:
    return await asyncio.to_thread(status_snapshot, _config())


@router.get("/profiles")
async def get_profiles() -> dict[str, Any]:
    try:
        return {"ok": True, "profiles": await asyncio.to_thread(_profile_rows)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/configure")
async def configure(body: dict[str, Any]) -> dict[str, Any]:
    primary = body.get("primary", False)
    fallback = body.get("fallback") if "fallback" in body else None
    profile = body.get("profile")
    base_url = body.get("base_url")
    if not isinstance(primary, bool) or (fallback is not None and not isinstance(fallback, bool)):
        raise HTTPException(status_code=422, detail="primary and fallback must be booleans")
    if profile is not None and not isinstance(profile, str):
        raise HTTPException(status_code=422, detail="profile must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise HTTPException(status_code=422, detail="base_url must be a string")
    try:
        return await asyncio.to_thread(
            apply_configuration,
            primary=primary,
            fallback=fallback,
            profile=profile,
            base_url=base_url,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
