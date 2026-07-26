from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from test_selection import catalog, hardware
from turbofit_runtime.controller import (
    AdaptiveController,
    ControllerState,
    RungRequirements,
    load_controller_state,
    load_rung_requirements,
    save_controller_state,
)
from turbofit_runtime.pressure import CardPressure, PressureSnapshot
from turbofit_runtime.reconciler import RollbackFailedError
from turbofit_runtime.reconciler import ReconcilerState
from turbofit_runtime.selection import ProfileCatalog


def pressure(*available: int) -> PressureSnapshot:
    return PressureSnapshot(
        cards=tuple(
            CardPressure(
                gpu=index,
                total_mb=24576,
                observed_used_mb=0,
                managed_mb=0,
                external_mb=0,
                desktop_mb=0,
                safety_reserve_mb=0,
                reservation_mb=0,
                available_for_managed_mb=value,
            )
            for index, value in enumerate(available)
        ),
        process_data_available=True,
        release_targets=(),
    )


@dataclass
class Backend:
    events: list[object] = field(default_factory=list)

    def reset_managed(self): self.events.append("reset")
    def block_aux_admission(self): self.events.append("block")
    def drain_aux(self, timeout_s): self.events.append(("drain", timeout_s)); return True
    def clean_unload_aux(self): self.events.append("unload"); return True
    def owned_pids(self): return ()
    def escalate_owned(self, pids): self.events.append(("escalate", pids))
    def activate_local(self, rung_id): self.events.append(("local", rung_id))
    def activate_api(self, main, aux): self.events.append(("api", main, aux))
    def route_aux_to_main(self): self.events.append("shared")
    def route_aux_dedicated(self): self.events.append("dedicated")
    def verify_rung(self, rung_id): self.events.append(("verify", rung_id)); return True
    def publish_routes(self, state): self.events.append(("publish", state.rung_index))
    def restore(self, state): self.events.append(("restore", state.rung_index))
    def verify_restore(self, state): return True


def controller_for(
    profiles: ProfileCatalog, requested: str, memory: tuple[int, ...]
) -> tuple[AdaptiveController, Backend]:
    choice = profiles.select(hardware(*memory), requested=requested)
    requirements = {
        "hardware-8gb": ((),),
        "hardware-16gb": ((),),
        "hardware-24gb": ((21554,), ()),
        "hardware-48gb": ((23028, 22768), ()),
    }[choice.profile.id]
    backend = Backend()
    state = ControllerState.from_choice(choice)
    return AdaptiveController(
        profile=choice.profile,
        requirements=RungRequirements(choice.profile.id, requirements),
        backend=backend,
        state=state,
    ), backend


@pytest.mark.parametrize(
    ("requested", "memory"),
    [("auto", (8192,)), ("hardware-16gb", (24576,))],
)
def test_api_only_selections_remain_safe_without_local_activation(
    requested: str, memory: tuple[int, ...]
) -> None:
    controller, backend = controller_for(catalog(), requested, memory)

    result = controller.tick(pressure(*(value for value in memory)), now=500)

    assert result.state.adaptive.current_index == 0
    assert result.transitioned is False
    assert backend.events == []


def test_controller_repairs_a_missing_persisted_local_rung_after_manager_restart() -> None:
    original, _ = controller_for(catalog(), "hardware-48gb", (24576, 24576))
    state = replace(
        original.state,
        adaptive=replace(
            original.state.adaptive,
            current_index=0,
            last_stable_index=0,
            pending_index=None,
            target_ceiling_index=0,
        ),
        reconciler=ReconcilerState(
            profile_id=original.profile.id,
            rung_index=0,
            main_target="local:main",
            aux_target="local:aux",
        ),
    )

    @dataclass
    class ColdBackend(Backend):
        ready: bool = False

        def activate_local(self, rung_id):
            super().activate_local(rung_id)
            self.ready = True

        def verify_rung(self, rung_id):
            self.events.append(("verify", rung_id))
            return self.ready

    backend = ColdBackend()
    controller = AdaptiveController(
        profile=original.profile,
        requirements=original.requirements,
        backend=backend,
        state=state,
    )

    result = controller.tick(pressure(24576, 24576), now=0)

    assert result.transitioned is True
    assert result.reason == "restored missing current rung"
    assert ("local", original.profile.rungs[0].id) in backend.events
    assert ("publish", 0) in backend.events

    backend.ready = False
    assert controller.tick(pressure(24576, 24576), now=20).transitioned is False
    repaired_again = controller.tick(pressure(24576, 24576), now=31)
    assert repaired_again.transitioned is True
    assert backend.events.count(("local", original.profile.rungs[0].id)) == 2


@pytest.mark.parametrize(
    ("requested", "memory", "clear", "pressured"),
    [
        ("auto", (24576,), (24576,), (10000,)),
        ("hardware-48gb", (24576, 24576), (24576, 24576), (16000, 16000)),
    ],
)
def test_auto_and_manual_profiles_contract_and_heal_with_same_safety_policy(
    requested: str,
    memory: tuple[int, ...],
    clear: tuple[int, ...],
    pressured: tuple[int, ...],
) -> None:
    controller, backend = controller_for(catalog(), requested, memory)

    controller.tick(pressure(*clear), now=0)
    healed = controller.tick(pressure(*clear), now=120)
    assert healed.transitioned is True
    assert healed.state.adaptive.current_index == 0

    controller.tick(pressure(*pressured), now=151)
    contracted = controller.tick(pressure(*pressured), now=156)
    assert contracted.transitioned is True
    assert contracted.state.adaptive.current_index == 1

    controller.tick(pressure(*clear), now=187)
    recovered = controller.tick(pressure(*clear), now=307)
    assert recovered.transitioned is True
    assert recovered.state.adaptive.current_index == 0

    assert [event for event in backend.events if isinstance(event, tuple) and event[0] == "publish"] == [
        ("publish", 0),
        ("publish", 1),
        ("publish", 0),
    ]


def test_requirement_vectors_must_match_profile_and_card_count() -> None:
    profiles = catalog()
    choice = profiles.select(hardware(24576), requested="auto")
    with pytest.raises(ValueError, match="rung count"):
        AdaptiveController(
            profile=choice.profile,
            requirements=RungRequirements(choice.profile.id, ((1,),)),
            backend=Backend(),
            state=ControllerState.from_choice(choice),
        )

    requirements = RungRequirements(choice.profile.id, ((1, 2), ()))
    controller = AdaptiveController(
        profile=choice.profile,
        requirements=requirements,
        backend=Backend(),
        state=ControllerState.from_choice(choice),
    )
    with pytest.raises(ValueError, match="card count"):
        controller.tick(pressure(24576), now=0)


def test_published_manual_requirements_are_evidence_bound_and_can_heal() -> None:
    from pathlib import Path

    from turbofit_runtime.profile_io import load_profile

    root = Path(__file__).resolve().parents[1]
    requirements_path = root / "runtime-profiles" / "rung-requirements.json"
    paths = list((root / "runtime-profiles").glob("*gb.yaml"))
    paths.extend((root / "runtime-profiles" / "migrated").glob("*.json"))
    supported = []
    for path in paths:
        item = load_profile(path)
        try:
            requirements = load_rung_requirements(requirements_path, item)
        except ValueError as exc:
            if "missing rung requirements" in str(exc):
                continue
            raise
        supported.append(item.id)
        assert len(requirements.required_mb_by_rung) == len(item.rungs)
        margin = round(item.policy.expansion_margin_gb_per_card * 1024)
        per_card_total = round(item.hardware.per_device_min_gb * 1024)
        for rung, required in zip(item.rungs, requirements.required_mb_by_rung, strict=True):
            if required:
                assert max(required) + margin <= per_card_total
            else:
                assert rung.aux_mode.value == "api"
    assert "hardware-64gb" in supported
    assert supported
    assert {f"hardware-{value}gb" for value in (8, 16, 24, 48)} <= set(supported)


def test_48gb_class_accounts_managed_residency_on_the_pinned_cards() -> None:
    from pathlib import Path

    from turbofit_runtime.profile_io import load_profile

    root = Path(__file__).resolve().parents[1]
    profile = load_profile(root / "runtime-profiles/48gb.yaml")
    requirements = load_rung_requirements(
        root / "runtime-profiles/rung-requirements.json", profile
    )
    assert requirements.required_mb_by_rung[0] == (11_163, 21_554)


def test_controller_state_round_trips_atomically(tmp_path) -> None:
    controller, _ = controller_for(catalog(), "auto", (24576,))
    controller.tick(pressure(24576), now=0)
    path = tmp_path / "controller.json"

    save_controller_state(path, controller.state)
    restored = load_controller_state(path)

    assert restored == controller.state
    assert not list(tmp_path.glob(".controller.json.*"))


def test_failed_activation_returns_persistable_rollback_state() -> None:
    profiles = catalog()
    choice = profiles.select(hardware(24576), requested="auto")

    class FailingBackend(Backend):
        def verify_rung(self, rung_id):
            return False

    controller = AdaptiveController(
        profile=choice.profile,
        requirements=RungRequirements(choice.profile.id, ((21554,), ())),
        backend=FailingBackend(),
        state=ControllerState.from_choice(choice),
    )
    controller.tick(pressure(24576), now=0)

    result = controller.tick(pressure(24576), now=120)

    assert result.action.value == "rollback"
    assert result.transitioned is False
    assert result.state.adaptive.current_index == 1
    assert result.state.adaptive.failure_times == (120,)


def test_failed_rollback_is_fatal_and_never_claimed_as_restored() -> None:
    profiles = catalog()
    choice = profiles.select(hardware(24576), requested="auto")

    class BrokenRollbackBackend(Backend):
        def verify_rung(self, rung_id):
            return False

        def verify_restore(self, state):
            return False

    controller = AdaptiveController(
        profile=choice.profile,
        requirements=RungRequirements(choice.profile.id, ((21554,), ())),
        backend=BrokenRollbackBackend(),
        state=ControllerState.from_choice(choice),
    )
    controller.tick(pressure(24576), now=0)

    with pytest.raises(RollbackFailedError, match="rollback verification failed"):
        controller.tick(pressure(24576), now=120)
