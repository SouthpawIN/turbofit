from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.reconciler import ReconcileError, ReconcilerState
from turbofit_runtime.routes import build_route_state, load_runtime_resolutions, publish_route_state
from turbofit_runtime.turbohaul_backend import TurbohaulBackend


ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self) -> None:
        self.chats: list[dict] = []
        self.unloads: list[str] = []
        self.snapshot: dict = {"active": None, "idle_hot": None, "residents": []}

    def chat_completion(self, payload):
        self.chats.append(payload)
        tag = payload["model"]
        residents = self.snapshot.setdefault("residents", [])
        if not any(item.get("model_tag") == tag for item in residents):
            residents.append({"model_tag": tag, "pid": 1000 + len(residents)})
        return {"choices": [{"message": {"content": "OK"}}]}

    def status(self):
        return self.snapshot

    def unload_model(self, model, **kwargs):
        self.unloads.append(model)
        self.snapshot["residents"] = [
            item for item in self.snapshot.get("residents", []) if item.get("model_tag") != model
        ]
        return self.snapshot


def fixture(tmp_path: Path, class_gb: int, rung_index: int):
    profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
    resolutions = load_runtime_resolutions(ROOT / "runtime-profiles" / "runtime-resolutions.json")
    routes = tmp_path / "routes.json"
    publish_route_state(routes, build_route_state(profile, rung_index, resolutions, manager_port=11401))
    state = ReconcilerState(
        profile_id=profile.id,
        rung_index=rung_index,
        main_target="current-main",
        aux_target="current-aux",
    )
    client = Client()
    backend = TurbohaulBackend(
        profile=profile,
        resolutions=resolutions,
        route_state_path=routes,
        manager_port=11401,
        client=client,
        current_state=state,
        sleep=lambda _: None,
        clock=lambda: 100.0,
    )
    return profile, resolutions, routes, state, client, backend


def test_block_aux_admission_redirects_new_work_before_drain(tmp_path: Path) -> None:
    _, _, routes, _, client, backend = fixture(tmp_path, 48, 0)
    current = json.loads(routes.read_text())
    aux_tag = current["routes"]["aux"]["alias"]
    client.snapshot["active"] = {"model_tag": aux_tag, "pid": 1002}

    backend.block_aux_admission()

    staged = json.loads(routes.read_text())
    assert staged["routes"]["aux"] == {"kind": "shared-main"}
    assert backend.drain_aux(0) is False
    client.snapshot["active"] = None
    assert backend.drain_aux(0) is True


def test_activate_verify_and_publish_dedicated_local_rung(tmp_path: Path) -> None:
    profile, _, routes, _, client, backend = fixture(tmp_path, 48, 1)

    rung_id = profile.rungs[0].id
    backend.activate_local(rung_id)
    backend.route_aux_dedicated()
    assert backend.verify_rung(rung_id) is True
    published = ReconcilerState(profile.id, 0, "local:main", "local:aux")
    backend.publish_routes(published)

    tags = [payload["model"] for payload in client.chats]
    assert tags == [
        "hardware-48gb-grm-128k-main",
        "carwin-nano-1-bit-bonsai-262k-aux",
    ]
    state = json.loads(routes.read_text())
    assert state["rung_index"] == 0
    assert state["routes"]["aux"]["mode"] == "dedicated"


def test_api_activation_publishes_policy_not_credentials(tmp_path: Path) -> None:
    profile, resolutions, routes, _, client, backend = fixture(tmp_path, 24, 0)
    main_tag = resolutions[profile.id]["local-131072"]["main"]["model_tag"]
    client.snapshot["residents"] = [{"model_tag": main_tag, "pid": 1234}]

    backend.activate_api("api:auto", "api:auto")
    guarded = json.loads(routes.read_text())["routes"]
    assert guarded["main"]["kind"] == "api-policy"
    assert guarded["aux"]["kind"] == "api-policy"
    assert client.unloads == [main_tag]
    assert backend.verify_rung("api") is True
    backend.publish_routes(ReconcilerState(profile.id, 1, "api:auto", "api:auto"))

    text = routes.read_text()
    assert "api-policy" in text
    assert "api_key" not in text


def test_unload_is_delegated_to_turbohaul_and_escalation_never_signals(tmp_path: Path) -> None:
    _, _, routes, _, client, backend = fixture(tmp_path, 48, 0)
    aux_tag = json.loads(routes.read_text())["routes"]["aux"]["alias"]
    client.snapshot["residents"] = [{"model_tag": aux_tag, "pid": 4321}]
    backend.block_aux_admission()

    assert backend.owned_pids() == (4321,)
    assert backend.clean_unload_aux() is True
    assert client.unloads == [aux_tag]
    with pytest.raises(ReconcileError, match="Turbohaul Manager owns escalation"):
        backend.escalate_owned((4321,))

    source = (ROOT / "src" / "turbofit_runtime" / "turbohaul_backend.py").read_text()
    assert "os.kill" not in source and "SIGKILL" not in source
