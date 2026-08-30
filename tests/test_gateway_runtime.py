from __future__ import annotations

from pathlib import Path

from turbofit_runtime.gateway_runtime import build_gateway_launch


def test_gateway_launch_is_loopback_only_and_uses_exact_route_state(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugin"
    script = plugin_root / "scripts/turbofit-gateway.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('gateway')")
    python = tmp_path / "python"
    python.write_text("python")
    python.chmod(0o755)
    routes = tmp_path / "runtime-state.json"

    launch = build_gateway_launch(
        python=python,
        plugin_root=plugin_root,
        routes_path=routes,
        port=8091,
    )

    assert launch.engine_id == "turbofit-gateway"
    assert launch.model_id == "auto"
    assert launch.upstream_model_id == "auto"
    assert launch.required_model_files == ()
    assert launch.command == (
        "/usr/bin/env",
        "TURBOFIT_GATEWAY_HOST=127.0.0.1",
        "TURBOFIT_GATEWAY_PORT=8091",
        f"TURBOFIT_RUNTIME_STATE={routes}",
        str(python),
        str(script),
    )
