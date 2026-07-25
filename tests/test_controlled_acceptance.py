from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/controlled-pressure-acceptance"
LOADER = SourceFileLoader("controlled_pressure_acceptance", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(
    "controlled_pressure_acceptance",
    LOADER,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_route_state_predicates_distinguish_local_and_terminal_api(tmp_path: Path) -> None:
    state = tmp_path / "routes.json"
    local = {
        "active": "hardware-48gb",
        "rung_index": 0,
        "routes": {"main": {"kind": "local"}, "aux": {"kind": "local"}},
    }
    state.write_text(json.dumps(local))
    assert MODULE.wait_for(state, MODULE.is_local, 0.1) == local
    assert not MODULE.is_api(local)

    terminal = {
        **local,
        "rung_index": 1,
        "routes": {
            "main": {"kind": "api-policy"},
            "aux": {"kind": "api-policy"},
        },
    }
    state.write_text(json.dumps(terminal))
    assert MODULE.wait_for(state, MODULE.is_api, 0.1) == terminal


def test_acceptance_only_signals_the_process_handle_it_created() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/controlled-pressure-acceptance").read_text()
    assert "os.kill" not in source
    assert "killpg" not in source
    assert "pkill" not in source
    assert "acceptance_owned_processes_terminated" in source
