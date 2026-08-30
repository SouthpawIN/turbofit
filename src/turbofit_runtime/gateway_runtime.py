"""Owned loopback launch contract for the Turbofit provider gateway."""
from __future__ import annotations

from pathlib import Path

from .managed_engine import EngineLaunch


def build_gateway_launch(
    *,
    python: str | Path,
    plugin_root: str | Path,
    routes_path: str | Path,
    port: int = 8091,
) -> EngineLaunch:
    root = Path(plugin_root).resolve()
    script = root / "scripts" / "turbofit-gateway.py"
    routes = Path(routes_path).expanduser().resolve()
    command = (
        "/usr/bin/env",
        "TURBOFIT_GATEWAY_HOST=127.0.0.1",
        f"TURBOFIT_GATEWAY_PORT={port}",
        f"TURBOFIT_RUNTIME_STATE={routes}",
        str(Path(python).expanduser().resolve()),
        str(script),
    )
    return EngineLaunch(
        engine_id="turbofit-gateway",
        model_id="auto",
        model_path=root,
        port=port,
        context_length=262_144,
        bind_host="127.0.0.1",
        upstream_model_id="auto",
        command=command,
        required_model_files=(),
    )
