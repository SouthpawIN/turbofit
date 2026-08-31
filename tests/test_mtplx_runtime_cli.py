from __future__ import annotations

import importlib.util
import json
import subprocess
from dataclasses import replace
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest

from turbofit_runtime.mtplx_engine import MtplxEndpoint


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/turbofit-mtplx-runtime"


def load_script():
    loader = SourceFileLoader("turbofit_mtplx_runtime", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Runtime:
    def __init__(self) -> None:
        self.adopted = []
        self.started = []
        self.stopped = 0

    def adopt_external(self, launch, *, pid):
        self.adopted.append((launch, pid))
        return SimpleNamespace(pid=pid, owned=False)

    def start_owned(self, launch, *, timeout_s):
        self.started.append((launch, timeout_s))
        return SimpleNamespace(pid=77, owned=True)

    def stop(self):
        self.stopped += 1
        return False

    def inspect(self):
        return {
            "ok": True,
            "status": "ready",
            "runtime": {"pid": 4321, "port": 18086, "owned": True},
        }


def endpoint() -> MtplxEndpoint:
    return MtplxEndpoint(
        host="127.0.0.1",
        port=18086,
        model_id="served-speed",
        model_path="/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
        pid=4321,
        app_launch_id="app-launch",
        health={"ok": True, "model": "served-speed"},
    )


def test_adopt_routes_external_mtplx_without_taking_process_ownership(tmp_path: Path) -> None:
    module = load_script()
    runtime = Runtime()
    routes = tmp_path / "routes.json"

    result = module.run_action(
        "adopt",
        runtime=runtime,
        routes_path=routes,
        discover=lambda: endpoint(),
        executable=tmp_path / "mtplx",
        model_path=None,
        model_repo=None,
        model_id=None,
        port=8000,
        timeout_s=30,
    )

    assert result["owned"] is False
    assert result["pid"] == 4321
    assert runtime.adopted[0][1] == 4321
    state = json.loads(routes.read_text())
    assert state["routes"]["main"]["alias"] == "qwen3-8-27b-mtplx-optimized-speed"
    assert state["routes"]["main"]["model_id"] == "served-speed"
    assert state["routes"]["main"]["port"] == 18086


def test_adopt_fails_cleanly_when_external_mtplx_does_not_report_a_pid(tmp_path: Path) -> None:
    module = load_script()
    runtime = Runtime()

    with pytest.raises(RuntimeError, match="does not report a PID"):
        module.run_action(
            "adopt",
            runtime=runtime,
            routes_path=tmp_path / "routes.json",
            discover=lambda: replace(endpoint(), pid=None),
            executable=tmp_path / "mtplx",
            model_path=None,
            model_repo=None,
            model_id=None,
            port=8000,
            timeout_s=30,
        )


def test_stop_never_signals_adopted_external_mtplx(tmp_path: Path) -> None:
    module = load_script()
    runtime = Runtime()
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps({"active": "apple-mtplx", "routes": {}}))

    result = module.run_action(
        "stop",
        runtime=runtime,
        routes_path=routes,
        discover=lambda: endpoint(),
        executable=tmp_path / "mtplx",
        model_path=None,
        model_repo=None,
        model_id=None,
        port=8000,
        timeout_s=30,
    )

    assert runtime.stopped == 1
    assert result == {"ok": True, "action": "detach", "process_stopped": False}
    assert not routes.exists()


def test_status_distinguishes_turbofit_owned_mtplx(tmp_path: Path) -> None:
    module = load_script()

    result = module.run_action(
        "status",
        runtime=Runtime(),
        routes_path=tmp_path / "routes.json",
        discover=lambda: endpoint(),
        executable=tmp_path / "mtplx",
        model_path=None,
        model_repo=None,
        model_id=None,
        port=8000,
        timeout_s=30,
    )

    assert result["endpoint"]["owned_by_turbofit"] is True
    assert result["endpoint"]["lifecycle_status"] == "ready"


def test_cli_reexecutes_from_legacy_system_python_before_runtime_imports(tmp_path: Path) -> None:
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
