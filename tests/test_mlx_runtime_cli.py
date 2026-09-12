from __future__ import annotations

import importlib.util
import json
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/turbofit-mlx-runtime"


def load_script():
    loader = SourceFileLoader("turbofit_mlx_runtime", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Runtime:
    def __init__(self) -> None:
        self.started = []
        self.stopped = 0

    def start_owned(self, launch, *, timeout_s):
        self.started.append((launch, timeout_s))
        return SimpleNamespace(pid=42, owned=True)

    def stop(self):
        self.stopped += 1
        return True


def test_start_publishes_routes_only_after_runtime_verifies(tmp_path: Path) -> None:
    module = load_script()
    runtime = Runtime()
    routes = tmp_path / "routes.json"
    model_root = tmp_path / "models"
    runtime_root = tmp_path / "runtimes"

    result = module.run_action(
        "start",
        runtime=runtime,
        model_root=model_root,
        runtime_root=runtime_root,
        routes_path=routes,
        port=18081,
        timeout_s=30,
    )

    assert result == {"ok": True, "action": "start", "pid": 42, "owned": True}
    assert runtime.started[0][1] == 30
    assert json.loads(routes.read_text())["active"] == "apple-mlx"


def test_stop_removes_only_the_apple_mlx_route(tmp_path: Path) -> None:
    module = load_script()
    runtime = Runtime()
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"active": "apple-mlx", "routes": {}}))

    result = module.run_action(
        "stop",
        runtime=runtime,
        model_root=tmp_path / "models",
        runtime_root=tmp_path / "runtimes",
        routes_path=routes,
        port=18081,
        timeout_s=30,
    )

    assert result == {"ok": True, "action": "stop"}
    assert runtime.stopped == 1
    assert not routes.exists()


def test_cli_reexecutes_from_legacy_system_python_before_importing_runtime(tmp_path: Path) -> None:
    system_python = Path("/usr/bin/python3")
    if not system_python.exists():
        return
    result = subprocess.run(
        [str(system_python), str(SCRIPT), "status", "--state", str(tmp_path / "missing.json")],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert "unsupported operand type" not in result.stderr
    assert json.loads(result.stdout)["action"] == "status"
