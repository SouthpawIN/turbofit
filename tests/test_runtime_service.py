from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from test_controller import pressure
from test_selection import hardware
from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.routes import load_runtime_resolutions
from turbofit_runtime.runtime_service import RuntimeService
from turbofit_runtime.selection import ProfileCatalog, save_selection


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Backend:
    events: list[object] = field(default_factory=list)

    def block_aux_admission(self): self.events.append("block")
    def drain_aux(self, timeout_s): return True
    def clean_unload_aux(self): return True
    def owned_pids(self): return ()
    def escalate_owned(self, pids): raise AssertionError("not expected")
    def activate_local(self, rung_id): self.events.append(("local", rung_id))
    def activate_api(self, main, aux): self.events.append(("api", main, aux))
    def route_aux_to_main(self): self.events.append("shared")
    def route_aux_dedicated(self): self.events.append("dedicated")
    def verify_rung(self, rung_id): return True
    def publish_routes(self, state): self.events.append(("publish", state.rung_index))
    def restore(self, state): self.events.append(("restore", state.rung_index))
    def verify_restore(self, state): return True


def service(tmp_path: Path):
    profiles = ProfileCatalog(
        load_yaml_profile(ROOT / "runtime-profiles" / f"{size}gb.yaml")
        for size in (8, 16, 24, 48)
    )
    backends: list[Backend] = []

    def factory(profile, state):
        del profile, state
        item = Backend()
        backends.append(item)
        return item

    return RuntimeService(
        catalog=profiles,
        resolutions=load_runtime_resolutions(ROOT / "runtime-profiles" / "runtime-resolutions.json"),
        requirements_path=ROOT / "runtime-profiles" / "rung-requirements.json",
        controller_state_path=tmp_path / "controller.json",
        route_state_path=tmp_path / "routes.json",
        manager_port=11401,
        backend_factory=factory,
    ), backends


def test_fresh_auto_selection_starts_at_terminal_api_before_healing(tmp_path: Path) -> None:
    runtime, _ = service(tmp_path)
    choice = runtime.catalog.select(hardware(24576), requested="auto")
    selection_path = tmp_path / "selection.json"
    save_selection(selection_path, choice)

    controller = runtime.synchronize(selection_path, hardware(24576))

    assert controller.state.adaptive.current_index == 1
    route = json.loads((tmp_path / "routes.json").read_text())
    assert route["rung_id"] == "api"
    assert route["routes"]["main"]["kind"] == "api-policy"


def test_service_persists_healing_and_contraction_state(tmp_path: Path) -> None:
    runtime, backends = service(tmp_path)
    choice = runtime.catalog.select(hardware(24576), requested="auto")
    selection_path = tmp_path / "selection.json"
    save_selection(selection_path, choice)
    runtime.synchronize(selection_path, hardware(24576))

    runtime.tick(pressure(24576), now=0)
    healed = runtime.tick(pressure(24576), now=120)
    assert healed.state.adaptive.current_index == 0

    runtime.tick(pressure(10000), now=151)
    contracted = runtime.tick(pressure(10000), now=156)
    assert contracted.state.adaptive.current_index == 1
    assert ("publish", 0) in backends[-1].events
    assert ("publish", 1) in backends[-1].events

    restarted, _ = service(tmp_path)
    restored = restarted.synchronize(selection_path, hardware(24576))
    assert restored.state.adaptive.current_index == 1
