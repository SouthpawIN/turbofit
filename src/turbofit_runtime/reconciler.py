"""Verified, rollback-capable runtime rung effect reconciler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .runtime_profile import AuxMode, RuntimeRung, Turbofile


class ReconcileError(RuntimeError):
    pass


class RollbackFailedError(ReconcileError):
    """The target failed and the previously published rung could not be restored."""


@dataclass(frozen=True)
class ReconcilerState:
    profile_id: str
    rung_index: int
    main_target: str
    aux_target: str


class RuntimeBackend(Protocol):
    def block_aux_admission(self) -> None: ...
    def drain_aux(self, timeout_s: float) -> bool: ...
    def clean_unload_aux(self) -> bool: ...
    def owned_pids(self) -> tuple[int, ...]: ...
    def escalate_owned(self, pids: tuple[int, ...]) -> None: ...
    def activate_local(self, rung_id: str) -> None: ...
    def activate_api(self, main_policy: str, aux_policy: str) -> None: ...
    def route_aux_to_main(self) -> None: ...
    def route_aux_dedicated(self) -> None: ...
    def verify_rung(self, rung_id: str) -> bool: ...
    def publish_routes(self, state: ReconcilerState) -> None: ...
    def restore(self, state: ReconcilerState) -> None: ...
    def verify_restore(self, state: ReconcilerState) -> bool: ...


def transition(
    current: ReconcilerState,
    target_index: int,
    profile: Turbofile,
    backend: RuntimeBackend,
    *,
    drain_timeout_s: float = 60,
) -> ReconcilerState:
    if current.profile_id != profile.id:
        raise ValueError("current state profile does not match Turbofile")
    if not 0 <= current.rung_index < len(profile.rungs):
        raise ValueError("current rung index is outside profile")
    if not 0 <= target_index < len(profile.rungs):
        raise ValueError("target rung index is outside profile")
    if drain_timeout_s < 0:
        raise ValueError("drain_timeout_s must be non-negative")
    if target_index == current.rung_index:
        return current

    source = profile.rungs[current.rung_index]
    target = profile.rungs[target_index]
    try:
        if source.aux_mode is AuxMode.DEDICATED and target.aux_mode is not AuxMode.DEDICATED:
            _retire_dedicated_aux(backend, drain_timeout_s)

        if target.aux_mode is AuxMode.API:
            assert target.main_api_policy is not None
            assert target.aux_api_policy is not None
            backend.activate_api(target.main_api_policy, target.aux_api_policy)
            main_target = target.main_api_policy
            aux_target = target.aux_api_policy
        else:
            if not _kv_preserving_seam(source, target):
                backend.activate_local(target.id)
            if target.aux_mode is AuxMode.SHARED_MAIN:
                backend.route_aux_to_main()
                aux_target = "local:main"
            else:
                backend.route_aux_dedicated()
                aux_target = "local:aux"
            main_target = "local:main"

        if not backend.verify_rung(target.id):
            raise ReconcileError(f"target rung verification failed: {target.id}")
        published = ReconcilerState(
            profile_id=profile.id,
            rung_index=target_index,
            main_target=main_target,
            aux_target=aux_target,
        )
        backend.publish_routes(published)
        return published
    except Exception as exc:
        _restore_or_raise(current, backend, exc)
        raise AssertionError("unreachable")


def _retire_dedicated_aux(backend: RuntimeBackend, timeout_s: float) -> None:
    backend.block_aux_admission()
    if not backend.drain_aux(timeout_s):
        raise ReconcileError("auxiliary stream drain timed out")
    if backend.clean_unload_aux():
        return
    owned = tuple(sorted(set(backend.owned_pids())))
    if not owned:
        raise ReconcileError("clean auxiliary unload failed and no owned process is available")
    backend.escalate_owned(owned)
    if not backend.clean_unload_aux():
        raise ReconcileError("owned-process escalation did not unload auxiliary")


def _kv_preserving_seam(source: RuntimeRung, target: RuntimeRung) -> bool:
    return (
        source.main_manifest is not None
        and source.main_manifest == target.main_manifest
        and source.context == target.context
    )


def _restore_or_raise(
    current: ReconcilerState, backend: RuntimeBackend, original: Exception
) -> None:
    try:
        backend.restore(current)
        restored = backend.verify_restore(current)
    except Exception as rollback_error:
        raise RollbackFailedError(
            f"transition failed ({original}); rollback raised: {rollback_error}"
        ) from rollback_error
    if not restored:
        raise RollbackFailedError(
            f"transition failed ({original}); rollback verification failed"
        ) from original
    if isinstance(original, ReconcileError):
        raise original
    raise ReconcileError(f"transition failed; previous state restored: {original}") from original
