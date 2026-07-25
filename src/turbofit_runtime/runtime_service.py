"""Persistent orchestration shell for one adaptive controller instance."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .controller import (
    AdaptiveController,
    ControllerResult,
    ControllerState,
    load_controller_state,
    load_rung_requirements,
    save_controller_state,
)
from .hardware import HardwareFingerprint
from .pressure import PressureSnapshot
from .reconciler import ReconcilerState, RuntimeBackend, transition
from .routes import RuntimeResolutions, build_route_state, publish_route_state
from .selection import ProfileCatalog, load_selection

BackendFactory = Callable[[object, ReconcilerState], RuntimeBackend]


class RuntimeService:
    """Synchronize persisted selection, controller state, and route publication."""

    def __init__(
        self,
        *,
        catalog: ProfileCatalog,
        resolutions: RuntimeResolutions,
        requirements_path: str | Path,
        controller_state_path: str | Path,
        route_state_path: str | Path,
        manager_port: int,
        backend_factory: BackendFactory,
    ) -> None:
        self.catalog = catalog
        self.resolutions = resolutions
        self.requirements_path = Path(requirements_path)
        self.controller_state_path = Path(controller_state_path)
        self.route_state_path = Path(route_state_path)
        self.manager_port = manager_port
        self.backend_factory = backend_factory
        self.controller: AdaptiveController | None = None

    def synchronize(
        self,
        selection_path: str | Path,
        hardware: HardwareFingerprint,
    ) -> AdaptiveController:
        selection = load_selection(selection_path)
        choice = self.catalog.select(hardware, requested=selection["requested"])
        if choice.profile.id != selection["profile_id"]:
            raise ValueError("persisted automatic selection is stale for current physical hardware")

        existing: ControllerState | None
        try:
            existing = load_controller_state(self.controller_state_path)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            existing.profile_id != choice.profile.id
            or existing.selection_mode is not choice.mode
        ):
            self._retire_previous(existing)
            existing = None

        if existing is None:
            state = ControllerState.from_choice(choice)
            initial_routes = build_route_state(
                choice.profile,
                state.adaptive.current_index,
                self.resolutions,
                manager_port=self.manager_port,
            )
            publish_route_state(self.route_state_path, initial_routes)
            save_controller_state(self.controller_state_path, state)
        else:
            state = existing
            self._validate_state_bounds(state, choice.profile.id, len(choice.profile.rungs))

        backend = self.backend_factory(choice.profile, state.reconciler)
        self.controller = AdaptiveController(
            profile=choice.profile,
            requirements=load_rung_requirements(self.requirements_path, choice.profile),
            backend=backend,
            state=state,
        )
        return self.controller

    def tick(self, pressure: PressureSnapshot, *, now: float) -> ControllerResult:
        if self.controller is None:
            raise RuntimeError("runtime service must synchronize before ticking")
        result = self.controller.tick(pressure, now=now)
        save_controller_state(self.controller_state_path, result.state)
        return result

    def _retire_previous(self, existing: ControllerState) -> None:
        previous = next(
            (profile for profile in self.catalog.profiles if profile.id == existing.profile_id),
            None,
        )
        if previous is None:
            raise ValueError(
                f"cannot safely retire unknown previous profile {existing.profile_id}"
            )
        self._validate_state_bounds(existing, previous.id, len(previous.rungs))
        terminal = len(previous.rungs) - 1
        if existing.reconciler.rung_index == terminal:
            return
        backend = self.backend_factory(previous, existing.reconciler)
        retired = transition(existing.reconciler, terminal, previous, backend)
        safe_state = ControllerState(
            selection_mode=existing.selection_mode,
            profile_id=previous.id,
            adaptive=existing.adaptive,
            reconciler=retired,
        )
        save_controller_state(self.controller_state_path, safe_state)

    @staticmethod
    def _validate_state_bounds(state: ControllerState, profile_id: str, rung_count: int) -> None:
        if state.profile_id != profile_id:
            raise ValueError("controller state profile mismatch")
        for value in (
            state.adaptive.current_index,
            state.adaptive.last_stable_index,
            state.adaptive.pending_index,
            state.adaptive.target_ceiling_index,
            state.reconciler.rung_index,
        ):
            if value is not None and not 0 <= value < rung_count:
                raise ValueError("controller state rung index is outside selected profile")
