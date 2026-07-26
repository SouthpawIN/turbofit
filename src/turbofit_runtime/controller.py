"""Small adaptive controller joining pressure, policy, and verified transitions."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .policy import ActionKind, AdaptiveState, CapacitySnapshot, reconcile
from .pressure import PressureSnapshot
from .reconciler import ReconcilerState, RollbackFailedError, RuntimeBackend, transition
from .runtime_profile import AuxMode, Turbofile
from .selection import ProfileChoice, SelectionMode


@dataclass(frozen=True)
class RungRequirements:
    profile_id: str
    required_mb_by_rung: tuple[tuple[int, ...], ...]

    def __init__(self, profile_id: str, required_mb_by_rung: Sequence[Sequence[int]]) -> None:
        values = tuple(tuple(row) for row in required_mb_by_rung)
        if not profile_id:
            raise ValueError("profile_id must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for row in values
            for value in row
        ):
            raise ValueError("rung requirements must be non-negative integers")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "required_mb_by_rung", values)


def load_rung_requirements(path: str | Path, profile: Turbofile) -> RungRequirements:
    """Load evidence-bound local VRAM requirements for one portable profile."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"schema", "profiles"}:
        raise ValueError("invalid rung requirements root")
    if raw["schema"] != "turbofit.rung-requirements/v1":
        raise ValueError("unsupported rung requirements schema")
    profiles = raw["profiles"]
    if not isinstance(profiles, Mapping) or profile.id not in profiles:
        raise ValueError(f"missing rung requirements for {profile.id}")
    rows = profiles[profile.id]
    if not isinstance(rows, list) or len(rows) != len(profile.rungs):
        raise ValueError("requirement rung count must match profile rung count")
    values: list[tuple[int, ...]] = []
    for index, (row, rung) in enumerate(zip(rows, profile.rungs, strict=True)):
        if not isinstance(row, Mapping) or set(row) != {
            "rung_id",
            "evidence",
            "required_mb_per_card",
        }:
            raise ValueError(f"invalid requirement row {index}")
        if row["rung_id"] != rung.id or row["evidence"] != rung.evidence:
            raise ValueError(f"requirement row {index} does not match profile evidence")
        required = row["required_mb_per_card"]
        if not isinstance(required, list):
            raise ValueError(f"requirement row {index} must contain a list")
        values.append(tuple(required))
    return RungRequirements(profile.id, values)


@dataclass(frozen=True)
class ControllerState:
    selection_mode: SelectionMode
    profile_id: str
    profile_revision: int
    adaptive: AdaptiveState
    reconciler: ReconcilerState

    @classmethod
    def from_choice(cls, choice: ProfileChoice) -> "ControllerState":
        index = choice.initial_rung_index
        rung = choice.profile.rungs[index]
        if rung.aux_mode is AuxMode.API:
            main_target = rung.main_api_policy or choice.profile.roles.fallback
            aux_target = rung.aux_api_policy or choice.profile.roles.fallback
        else:
            main_target = f"local:{rung.id}"
            aux_target = main_target if rung.aux_mode is AuxMode.SHARED_MAIN else f"local:{rung.id}:aux"
        return cls(
            selection_mode=choice.mode,
            profile_id=choice.profile.id,
            profile_revision=choice.profile.revision,
            adaptive=AdaptiveState(
                current_index=index,
                last_stable_index=index,
                target_ceiling_index=choice.target_ceiling_index,
            ),
            reconciler=ReconcilerState(
                profile_id=choice.profile.id,
                rung_index=index,
                main_target=main_target,
                aux_target=aux_target,
            ),
        )


def save_controller_state(path: str | Path, state: ControllerState) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "turbofit.controller-state/v2",
        "selection_mode": state.selection_mode.value,
        "profile_id": state.profile_id,
        "profile_revision": state.profile_revision,
        "adaptive": asdict(state.adaptive),
        "reconciler": asdict(state.reconciler),
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_controller_state(path: str | Path) -> ControllerState:
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("invalid controller state root")
    schema = raw.get("schema")
    required = {"schema", "selection_mode", "profile_id", "adaptive", "reconciler"}
    if schema == "turbofit.controller-state/v2":
        required.add("profile_revision")
    elif schema != "turbofit.controller-state/v1":
        raise ValueError("unsupported controller state schema")
    if set(raw) != required:
        raise ValueError("invalid controller state root")
    adaptive_raw = raw["adaptive"]
    reconciler_raw = raw["reconciler"]
    if not isinstance(adaptive_raw, Mapping) or set(adaptive_raw) != {
        "current_index",
        "last_stable_index",
        "pending_index",
        "target_ceiling_index",
        "deficit_since",
        "surplus_since",
        "cooldown_until",
        "failure_times",
        "quarantine_until",
    }:
        raise ValueError("invalid adaptive controller state")
    if not isinstance(reconciler_raw, Mapping) or set(reconciler_raw) != {
        "profile_id",
        "rung_index",
        "main_target",
        "aux_target",
    }:
        raise ValueError("invalid reconciler controller state")
    adaptive_values = dict(adaptive_raw)
    failures = adaptive_values.get("failure_times")
    if not isinstance(failures, list):
        raise ValueError("controller failure_times must be a list")
    adaptive_values["failure_times"] = tuple(failures)
    state = ControllerState(
        selection_mode=SelectionMode(raw["selection_mode"]),
        profile_id=raw["profile_id"],
        profile_revision=raw["profile_revision"] if schema.endswith("/v2") else 0,
        adaptive=AdaptiveState(**adaptive_values),
        reconciler=ReconcilerState(**dict(reconciler_raw)),
    )
    if state.profile_id != state.reconciler.profile_id:
        raise ValueError("controller state profile identities do not match")
    return state


@dataclass(frozen=True)
class ControllerResult:
    state: ControllerState
    transitioned: bool
    action: ActionKind
    reason: str


class AdaptiveController:
    """Reconcile one selected profile; manual and auto choices share this path."""

    def __init__(
        self,
        *,
        profile: Turbofile,
        requirements: RungRequirements,
        backend: RuntimeBackend,
        state: ControllerState,
    ) -> None:
        if requirements.profile_id != profile.id or state.profile_id != profile.id:
            raise ValueError("controller profile identities must match")
        if len(requirements.required_mb_by_rung) != len(profile.rungs):
            raise ValueError("requirement rung count must match profile rung count")
        for index, (rung, required) in enumerate(
            zip(profile.rungs, requirements.required_mb_by_rung, strict=True)
        ):
            if rung.aux_mode is AuxMode.API and required:
                raise ValueError(f"terminal API rung {index} cannot require local VRAM")
            if rung.aux_mode is not AuxMode.API and not required:
                raise ValueError(f"local rung {index} must declare per-card VRAM")
        self.profile = profile
        self.requirements = requirements
        self.backend = backend
        self.state = state

    def tick(self, pressure: PressureSnapshot, *, now: float) -> ControllerResult:
        available = tuple(card.available_for_managed_mb for card in pressure.cards)
        for index, (rung, required) in enumerate(
            zip(self.profile.rungs, self.requirements.required_mb_by_rung, strict=True)
        ):
            if rung.aux_mode is not AuxMode.API and len(required) != len(available):
                raise ValueError(
                    f"rung {index} requirement card count does not match pressure card count"
                )
        capacity = CapacitySnapshot(
            available_mb_per_card=available,
            required_mb_by_rung=self.requirements.required_mb_by_rung,
        )
        plan = reconcile(self.state.adaptive, capacity, self.profile, now)
        if plan.action is not ActionKind.ACTIVATE:
            self.state = replace(self.state, adaptive=plan.state)
            return ControllerResult(self.state, False, plan.action, plan.reason)

        target = plan.target_index
        assert target is not None
        try:
            reconciled = transition(
                self.state.reconciler,
                target,
                self.profile,
                self.backend,
            )
        except Exception as exc:
            if isinstance(exc, RollbackFailedError):
                raise
            failed = replace(capacity, activation_failed=True)
            rollback = reconcile(plan.state, failed, self.profile, now)
            self.state = replace(self.state, adaptive=rollback.state)
            return ControllerResult(
                self.state,
                False,
                ActionKind.ROLLBACK,
                f"activation failed and previous rung was restored: {exc}",
            )

        committed = reconcile(
            plan.state,
            replace(capacity, activation_succeeded=True),
            self.profile,
            now,
        )
        self.state = replace(
            self.state,
            adaptive=committed.state,
            reconciler=reconciled,
        )
        return ControllerResult(self.state, True, ActionKind.ACTIVATE, plan.reason)
