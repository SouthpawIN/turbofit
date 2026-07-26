from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from turbofit_runtime.controller import AdaptiveController, ControllerState, load_rung_requirements
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.pressure import CardPressure, PressureSnapshot
from turbofit_runtime.routes import load_runtime_resolutions
from turbofit_runtime.selection import ProfileCatalog


ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "runtime-profiles" / "rung-requirements.json"
RESOLUTIONS = {
    key: value
    for key, value in load_runtime_resolutions(
        ROOT / "runtime-profiles" / "runtime-resolutions.json"
    ).items()
    if key.startswith("hardware-")
}


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
    return ProfileCatalog.from_paths(
        sorted((ROOT / "runtime-profiles").glob("*gb.yaml"))
    )


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
        ((24576, 24576, 24576, 24576), "hardware-96gb"),
        ((102400, 102400), "hardware-200gb"),
        ((102400, 102400, 102400), "hardware-300gb"),
    ],
)
def test_auto_selects_canonical_hardware_profile(memory, expected) -> None:
    choice = supported_catalog().select(hardware(*memory), requested="auto")
    assert choice.profile.id == expected
    assert choice.initial_rung_index == len(choice.profile.rungs) - 1


def heal_to_ceiling(controller: AdaptiveController, clear: PressureSnapshot, now: float = 0) -> float:
    while controller.state.adaptive.current_index > 0:
        controller.tick(clear, now=now)
        now += 120
        result = controller.tick(clear, now=now)
        assert result.transitioned is True
        now += 31
    return now


def canonical_memory(profile) -> tuple[int, ...]:
    sizes: list[int] = []
    for part in profile.hardware.topology.split("+"):
        count, size = (int(value) for value in part.lower().removesuffix("gb").split("x"))
        sizes.extend([size * 1024] * count)
    return tuple(sizes)


@pytest.mark.parametrize(
    "memory",
    [
        (8192,),
        (16384,),
        (32768, 32768),
        (24576, 24576, 24576, 24576),
        (102400, 102400),
        (102400, 102400, 102400),
    ],
)
def test_every_hardware_class_starts_on_a_local_floor(memory) -> None:
    controller = make_controller("auto", memory)
    result = controller.tick(pressure(*memory), now=500)
    assert result.transitioned is False
    assert result.state.adaptive.current_index == len(controller.profile.rungs) - 1
    assert result.state.reconciler.main_target.startswith("local:")
    assert all(rung.aux_mode.value != "api" for rung in controller.profile.rungs)


@pytest.mark.parametrize("memory", [(24576,), (24576, 24576)])
def test_local_auto_contracts_under_pressure_and_heals_after_stable_headroom(memory) -> None:
    controller = make_controller("auto", memory)
    clear = pressure(*memory)
    pressured = pressure(*(10000 for _ in memory))

    now = heal_to_ceiling(controller, clear)
    assert controller.state.adaptive.current_index == 0
    controller.tick(pressured, now=now)
    result = controller.tick(pressured, now=now + 5)
    assert result.state.adaptive.current_index > 0
    heal_to_ceiling(controller, clear, now + 36)
    assert controller.state.adaptive.current_index == 0


@pytest.mark.parametrize("profile_id", sorted(RESOLUTIONS))
def test_every_offered_manual_local_combination_uses_the_same_contract_and_heal_path(profile_id) -> None:
    item = next(profile for profile in supported_catalog().profiles if profile.id == profile_id)
    memory = canonical_memory(item)
    controller = make_controller(profile_id, memory)
    clear = pressure(*memory)
    pressured = pressure(*(7000 for _ in memory))

    now = heal_to_ceiling(controller, clear)
    assert controller.state.adaptive.current_index == 0
    controller.tick(pressured, now=now)
    result = controller.tick(pressured, now=now + 5)
    if len(item.rungs) == 1:
        assert result.transitioned is False
        assert result.state.adaptive.current_index == 0
    else:
        assert result.state.adaptive.current_index == len(item.rungs) - 1
        heal_to_ceiling(controller, clear, now + 36)
        assert controller.state.adaptive.current_index == 0
