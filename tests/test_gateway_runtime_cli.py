from __future__ import annotations

import importlib.util
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/turbofit-gateway-runtime"


def load_script():
    loader = SourceFileLoader("turbofit_gateway_runtime", str(SCRIPT))
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
        return SimpleNamespace(pid=73, owned=True)

    def stop(self):
        self.stopped += 1
        return True


def test_gateway_cli_starts_owned_loopback_gateway(tmp_path: Path) -> None:
    module = load_script()
    runtime = Runtime()

    result = module.run_action(
        "start",
        runtime=runtime,
        python=tmp_path / "python",
        plugin_root=ROOT,
        routes_path=tmp_path / "routes.json",
        port=8091,
        timeout_s=30,
    )

    assert result == {"ok": True, "action": "start", "pid": 73, "owned": True}
    assert runtime.started[0][0].engine_id == "turbofit-gateway"


def test_gateway_cli_bootstraps_from_legacy_system_python(tmp_path: Path) -> None:
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
    assert '"action": "status"' in result.stdout
