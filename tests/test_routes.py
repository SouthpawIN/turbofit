from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.routes import (
    build_route_state,
    load_runtime_resolutions,
    publish_route_state,
)


ROOT = Path(__file__).resolve().parents[1]


def profile(class_gb: int):
    return load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")


def test_builds_shared_main_route_for_24gb_profile() -> None:
    resolutions = load_runtime_resolutions(
        ROOT / "runtime-profiles" / "runtime-resolutions.json"
    )
    state = build_route_state(profile(24), 0, resolutions, manager_port=11401)

    assert state["active"] == "hardware-24gb"
    assert state["rung_id"] == "local-grm-131072"
    assert state["routes"]["main"] == {
        "kind": "local",
        "alias": "grm-2-6-plus-auto-128k-main",
        "port": 11401,
    }
    assert state["routes"]["aux"] == {"kind": "shared-main"}


def test_builds_dedicated_route_for_48gb_profile() -> None:
    resolutions = load_runtime_resolutions(
        ROOT / "runtime-profiles" / "runtime-resolutions.json"
    )
    state = build_route_state(profile(48), 0, resolutions, manager_port=11401)

    assert state["routes"]["main"]["alias"].endswith("-main")
    assert state["routes"]["aux"]["alias"] == "bonsai-27b-1bit-262k-main"
    assert state["routes"]["aux"]["mode"] == "dedicated"
    assert state["routes"]["main"]["port"] == 11401


def test_local_floor_route_contains_no_api_policy_or_credentials() -> None:
    resolutions = load_runtime_resolutions(
        ROOT / "runtime-profiles" / "runtime-resolutions.json"
    )
    item = profile(8)
    state = build_route_state(item, len(item.rungs) - 1, resolutions, manager_port=11401)

    assert state["routes"]["main"]["kind"] == "local"
    assert state["routes"]["aux"] == {"kind": "shared-main"}
    assert "api-policy" not in json.dumps(state)
    assert "api_key" not in json.dumps(state)


def test_route_publication_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "runtime-state.json"
    resolutions = load_runtime_resolutions(
        ROOT / "runtime-profiles" / "runtime-resolutions.json"
    )
    state = build_route_state(profile(8), 0, resolutions, manager_port=11401)

    publish_route_state(path, state)

    assert json.loads(path.read_text()) == state
    assert not list(tmp_path.glob(".runtime-state.json.*"))
