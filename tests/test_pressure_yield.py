"""External GPU pressure must be able to yield VRAM, not only downshift rungs.

A game or media app on the same card is measured pressure, and
``external_gpu_priority: absolute`` promises that app gets the memory it
needs. When the ladder has no cheaper rung to move to (single-rung profile,
or the resident model is the top rung), the only way to keep that promise is
to release residency and restore it once the pressure is gone.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from test_controller import Backend, controller_for, hardware, pressure
from test_runtime_profile import DIGEST_A, valid_mapping
from test_selection import catalog as selection_catalog
from turbofit_runtime.policy import (
    ActionKind,
    AdaptiveState,
    CapacitySnapshot,
    reconcile,
)
from turbofit_runtime.recipes import RecipeBook, UnknownRecipeError
from turbofit_runtime.runtime_profile import Turbofile

ROOT = Path(__file__).resolve().parents[1]


def single_rung_profile() -> Turbofile:
    """One local rung: the ladder has nothing cheaper to contract to."""
    mapping = valid_mapping()
    mapping["policy"].update(contraction_dwell_s=10, expansion_dwell_s=20, cooldown_s=30)
    mapping["rungs"] = [
        {
            "id": "only-262k",
            "context": 262144,
            "aux_mode": "shared-main",
            "evidence": "sha256:" + "1" * 64,
            "main_manifest": DIGEST_A,
        }
    ]
    return Turbofile.from_mapping(mapping)


def api_rung_profile() -> Turbofile:
    """API-only rung: it holds no VRAM, so there is nothing to yield."""
    mapping = valid_mapping()
    mapping["policy"].update(contraction_dwell_s=10, expansion_dwell_s=20, cooldown_s=30)
    mapping["rungs"] = [
        {
            "id": "api",
            "context": 131072,
            "aux_mode": "api",
            "evidence": "sha256:" + "3" * 64,
            "main_api_policy": "api:auto",
            "aux_api_policy": "api:auto",
        }
    ]
    return Turbofile.from_mapping(mapping)


def capacity(available: int, *, contended: bool, releaseable: bool) -> CapacitySnapshot:
    return CapacitySnapshot(
        available_mb_per_card=(available,),
        required_mb_by_rung=((20000,),),
        contended_per_card=(contended,),
        current_rung_releaseable=releaseable,
    )


def test_sustained_contention_with_no_cheaper_rung_yields_residency() -> None:
    item = single_rung_profile()
    first = reconcile(
        AdaptiveState(current_index=0), capacity(24000, contended=True, releaseable=True), item, now=0
    )
    plan = reconcile(first.state, capacity(24000, contended=True, releaseable=True), item, now=10)

    assert first.action is ActionKind.NONE
    assert plan.action is ActionKind.RELEASE
    assert plan.target_index == 0
    assert plan.state.released is True


def test_yielded_residency_is_not_re_released_while_pressure_persists() -> None:
    item = single_rung_profile()
    yielded = replace(AdaptiveState(current_index=0), released=True)

    plan = reconcile(yielded, capacity(16000, contended=True, releaseable=True), item, now=500)

    assert plan.action is ActionKind.NONE
    assert plan.state.released is True
    assert "yielded" in plan.reason


def test_residency_is_restored_on_the_same_rung_once_pressure_clears() -> None:
    item = single_rung_profile()
    yielded = replace(AdaptiveState(current_index=0), released=True)

    first = reconcile(yielded, capacity(24000, contended=False, releaseable=False), item, now=500)
    plan = reconcile(first.state, capacity(24000, contended=False, releaseable=False), item, now=620)

    assert first.action is ActionKind.NONE
    assert first.state.surplus_since == 500
    assert plan.action is ActionKind.ACTIVATE
    assert plan.target_index == 0
    assert plan.state.pending_index == 0
    assert plan.state.released is False


def test_api_rung_never_yields_because_it_holds_no_vram() -> None:
    item = api_rung_profile()
    first = reconcile(
        AdaptiveState(current_index=0), capacity(24000, contended=True, releaseable=False), item, now=0
    )
    plan = reconcile(first.state, capacity(24000, contended=True, releaseable=False), item, now=10)

    assert plan.action is ActionKind.NONE
    assert plan.state.released is False


class ReleasingBackend(Backend):
    """Records the yield; residency stops being ready once it is released."""

    resident: bool = True

    def verify_rung(self, rung_id):
        self.events.append(("verify", rung_id))
        return self.resident

    def release_current(self) -> str:
        self.events.append("release")
        self.resident = False
        return "released"


def test_controller_releases_residency_and_does_not_reload_while_contended() -> None:
    from turbofit_runtime.controller import AdaptiveController, ControllerState, RungRequirements
    from turbofit_runtime.reconciler import ReconcilerState
    from turbofit_runtime.selection import SelectionMode

    profile = single_rung_profile()
    backend = ReleasingBackend()
    controller = AdaptiveController(
        profile=profile,
        requirements=RungRequirements(profile.id, ((20000,),)),
        backend=backend,
        state=ControllerState(
            selection_mode=SelectionMode.MANUAL,
            profile_id=profile.id,
            profile_revision=profile.revision,
            adaptive=AdaptiveState(current_index=0),
            reconciler=ReconcilerState(
                profile_id=profile.id,
                rung_index=0,
                main_target="local:main",
                aux_target="local:aux",
            ),
        ),
    )
    contended = replace(pressure(24000), cards=(replace(pressure(24000).cards[0], contended=True),))

    controller.tick(contended, now=0)
    result = controller.tick(contended, now=120)

    assert result.transitioned is True
    assert result.action is ActionKind.RELEASE
    assert "release" in backend.events
    assert controller.state.adaptive.released is True

    backend.events.clear()
    held = controller.tick(contended, now=240)
    assert held.action is ActionKind.NONE
    # The tick may re-check that residency is still gone; it must never reload it.
    assert "release" not in backend.events
    assert not [event for event in backend.events if isinstance(event, tuple) and event[0] == "local"], \
        "a yielded rung must not be reloaded under pressure"


def test_declared_alias_and_normalized_spelling_resolve_the_same_recipe(tmp_path) -> None:
    book = RecipeBook.load(ROOT / "references/model-recipes.json")

    by_alias = book._spec("qwen3-8-27b")
    by_display = book._spec("Qwen 3.8 27B")

    assert by_alias == by_display


def test_unknown_recipe_name_reports_close_candidates() -> None:
    book = RecipeBook.load(ROOT / "references/model-recipes.json")

    with pytest.raises(UnknownRecipeError) as excinfo:
        book._spec("qwen38-27b-does-not-exist")

    assert excinfo.value.candidates
    assert any("qwen" in name.lower() for name in excinfo.value.candidates)


def test_a_release_that_does_not_hold_is_repeated() -> None:
    """Something outside TurboFit can revive a released rung; re-yield it."""
    from turbofit_runtime.controller import AdaptiveController, ControllerState, RungRequirements
    from turbofit_runtime.reconciler import ReconcilerState
    from turbofit_runtime.selection import SelectionMode

    class RevivingBackend(ReleasingBackend):
        def release_current(self) -> str:
            self.events.append("release")
            self.resident = True  # an external starter revived it
            return "released"

    profile = single_rung_profile()
    backend = RevivingBackend()
    controller = AdaptiveController(
        profile=profile,
        requirements=RungRequirements(profile.id, ((20000,),)),
        backend=backend,
        state=ControllerState(
            selection_mode=SelectionMode.MANUAL,
            profile_id=profile.id,
            profile_revision=profile.revision,
            adaptive=replace(AdaptiveState(current_index=0), released=True),
            reconciler=ReconcilerState(
                profile_id=profile.id, rung_index=0,
                main_target="local:main", aux_target="local:aux",
            ),
        ),
    )

    held = controller.tick(pressure(24000), now=100)

    assert held.action is ActionKind.RELEASE
    assert held.transitioned is True
    assert "release" in backend.events
