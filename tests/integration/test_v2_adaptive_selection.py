from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import pytest

from turbofit_runtime.controller import AdaptiveController, ControllerState, load_rung_requirements
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.pressure import CardPressure, PressureSnapshot
from turbofit_runtime.profile_io import load_profile
from turbofit_runtime.routes import load_runtime_resolutions
from turbofit_runtime.selection import ProfileCatalog


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "runtime-profiles" / "rung-requirements.json"
RESOLUTIONS = load_runtime_resolutions(ROOT / "runtime-profiles" / "runtime-resolutions.json")


def hardware(*memory_mb: int) -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=65536,
        devices=tuple(
            AcceleratorDevice(
                index=index,
                uuid=f"GPU-{index}",
                name="GPU",
                vendor="nvidia",
                backend="cuda",
                memory_total_mb=value,
                compute_capability="8.9",
                bus_id=f"0000:{index + 1:02x}:00.0",
            )
            for index, value in enumerate(memory_mb)
        ),
    )


def pressure(*available: int) -> PressureSnapshot:
    return PressureSnapshot(
        cards=tuple(
            CardPressure(index, 24576, 0, 0, 0, 0, 0, 0, value)
            for index, value in enumerate(available)
        ),
        process_data_available=True,
        release_targets=(),
    )


@dataclass
class Backend:
    events: list[tuple | str] = field(default_factory=list)

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


def supported_catalog() -> ProfileCatalog:
    requirements = json.loads(REQUIREMENTS.read_text())["profiles"]
    api_only = {
        profile_id
        for profile_id, rows in requirements.items()
        if all(not row["required_mb_per_card"] for row in rows)
    }
    ids = set(RESOLUTIONS) | api_only
    paths = list((ROOT / "runtime-profiles").glob("*gb.yaml"))
    paths.extend((ROOT / "runtime-profiles" / "migrated").glob("*.json"))
    profiles = [item for item in (load_profile(path) for path in paths) if item.id in ids]
    return ProfileCatalog(profiles)


def make_controller(requested: str, memory: tuple[int, ...]) -> AdaptiveController:
    choice = supported_catalog().select(hardware(*memory), requested=requested)
    return AdaptiveController(
        profile=choice.profile,
        requirements=load_rung_requirements(REQUIREMENTS, choice.profile),
        backend=Backend(),
        state=ControllerState.from_choice(choice),
    )


@pytest.mark.parametrize(
    ("memory", "expected"),
    [
        ((8192,), "hardware-8gb"),
        ((16384,), "hardware-16gb"),
        ((24576,), "hardware-24gb"),
        ((24576, 24576), "hardware-48gb"),
        ((32768, 32768), "hardware-64gb"),
    ],
)
def test_auto_selects_canonical_hardware_profile(memory, expected) -> None:
    choice = supported_catalog().select(hardware(*memory), requested="auto")
    assert choice.profile.id == expected
    assert choice.initial_rung_index == len(choice.profile.rungs) - 1


@pytest.mark.parametrize("memory", [(8192,), (16384,), (32768, 32768)])
def test_api_only_small_cards_remain_safe(memory) -> None:
    controller = make_controller("auto", memory)
    result = controller.tick(pressure(*memory), now=500)
    assert result.transitioned is False
    assert result.state.adaptive.current_index == 0


@pytest.mark.parametrize("memory", [(24576,), (24576, 24576)])
def test_local_auto_contracts_under_pressure_and_heals_after_stable_headroom(memory) -> None:
    controller = make_controller("auto", memory)
    clear = pressure(*memory)
    pressured = pressure(*(10000 for _ in memory))

    controller.tick(clear, now=0)
    assert controller.tick(clear, now=120).state.adaptive.current_index == 0
    controller.tick(pressured, now=151)
    assert controller.tick(pressured, now=156).state.adaptive.current_index == 1
    controller.tick(clear, now=187)
    assert controller.tick(clear, now=307).state.adaptive.current_index == 0


@pytest.mark.parametrize("profile_id", sorted(RESOLUTIONS))
def test_every_offered_manual_local_combination_uses_the_same_contract_and_heal_path(profile_id) -> None:
    item = next(profile for profile in supported_catalog().profiles if profile.id == profile_id)
    count = len(load_rung_requirements(REQUIREMENTS, item).required_mb_by_rung[0])
    controller = make_controller(profile_id, tuple(24576 for _ in range(count)))
    clear = pressure(*(24576 for _ in range(count)))
    pressured = pressure(*(10000 for _ in range(count)))

    controller.tick(clear, now=0)
    assert controller.tick(clear, now=120).state.adaptive.current_index == 0
    controller.tick(pressured, now=151)
    assert controller.tick(pressured, now=156).state.adaptive.current_index == 1
    controller.tick(clear, now=187)
    assert controller.tick(clear, now=307).state.adaptive.current_index == 0
