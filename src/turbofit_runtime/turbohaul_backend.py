"""Turbohaul-owned effect backend for adaptive Turbofit transitions."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from .reconciler import ReconcileError, ReconcilerState
from .routes import RuntimeResolutions, build_route_state, publish_route_state
from .runtime_profile import AuxMode, Turbofile


class TurbohaulLike(Protocol):
    def status(self) -> dict[str, Any]: ...
    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def unload_model(self, model: str, **kwargs: Any) -> dict[str, Any]: ...


class ModelAcquirerLike(Protocol):
    def ensure_tags(self, tags: tuple[str, ...]) -> None: ...


class TurbohaulBackend:
    """All model lifecycle effects flow through Turbohaul; no direct signals."""

    def __init__(
        self,
        *,
        profile: Turbofile,
        resolutions: RuntimeResolutions,
        route_state_path: str | Path,
        manager_port: int,
        client: TurbohaulLike,
        acquirer: ModelAcquirerLike,
        current_state: ReconcilerState,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        verification_timeout_s: float = 30.0,
    ) -> None:
        if current_state.profile_id != profile.id:
            raise ValueError("backend state profile does not match profile")
        if verification_timeout_s < 0:
            raise ValueError("verification_timeout_s must be non-negative")
        self.profile = profile
        self.resolutions = resolutions
        self.route_state_path = Path(route_state_path)
        self.manager_port = manager_port
        self.client = client
        self.acquirer = acquirer
        self.current_state = current_state
        self.sleep = sleep
        self.clock = clock
        self.verification_timeout_s = verification_timeout_s
        self._target_rung_id: str | None = None
        self._target_aux_mode: AuxMode | None = None
        self._blocked_previous: dict[str, Any] | None = None
        self._retiring_aux_tag: str | None = None

    def reset_managed(self) -> None:
        """Unload only model tags owned by Turbofit before a profile revision migration."""
        managed = {
            str(role["model_tag"])
            for rungs in self.resolutions.values()
            for roles in rungs.values()
            for role in roles.values()
        }
        try:
            published = json.loads(self.route_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            published = {}
        routes = published.get("routes")
        if isinstance(routes, dict):
            for route in routes.values():
                if isinstance(route, dict) and route.get("kind") == "local":
                    alias = route.get("alias")
                    if isinstance(alias, str) and alias:
                        managed.add(alias)

        status = self.client.status()
        for tag in sorted(managed):
            if not _model_resident(status, tag):
                continue
            try:
                status = self.client.unload_model(tag, verification_timeout_s=30.0)
            except Exception as exc:
                raise ReconcileError(f"Turbohaul failed to reset managed model {tag}") from exc
            if _model_resident(status, tag):
                raise ReconcileError(f"Turbohaul kept managed model resident: {tag}")
        self._blocked_previous = None
        self._retiring_aux_tag = None

    def block_aux_admission(self) -> None:
        state = json.loads(self.route_state_path.read_text(encoding="utf-8"))
        routes = state.get("routes")
        if not isinstance(routes, dict) or not isinstance(routes.get("main"), dict):
            raise ReconcileError("cannot block auxiliary admission without valid routes")
        aux = routes.get("aux")
        if isinstance(aux, dict) and aux.get("kind") == "local":
            alias = aux.get("alias")
            if isinstance(alias, str) and alias:
                self._retiring_aux_tag = alias
        self._blocked_previous = state
        staged = json.loads(json.dumps(state))
        if staged["routes"]["main"].get("kind") == "local":
            staged["routes"]["aux"] = {"kind": "shared-main"}
        else:
            staged["routes"]["aux"] = {"kind": "api-policy", "policy": "api:auto"}
        publish_route_state(self.route_state_path, staged)

    def drain_aux(self, timeout_s: float) -> bool:
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        tag = self._retiring_aux_tag
        if not tag:
            return True
        deadline = self.clock() + timeout_s
        while True:
            if not _model_in_flight(self.client.status(), tag):
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(min(0.25, max(0.0, deadline - self.clock())))

    def clean_unload_aux(self) -> bool:
        tag = self._retiring_aux_tag
        if not tag:
            return True
        try:
            self.client.unload_model(tag, verification_timeout_s=30.0)
        except Exception:
            return False
        return not _model_resident(self.client.status(), tag)

    def owned_pids(self) -> tuple[int, ...]:
        tag = self._retiring_aux_tag
        if not tag:
            return ()
        return tuple(sorted(set(_matching_pids(self.client.status(), tag))))

    def escalate_owned(self, pids: tuple[int, ...]) -> None:
        del pids
        raise ReconcileError(
            "Turbohaul Manager owns escalation; Turbofit will not signal model processes"
        )

    def _retire_conflicting_split_residents(
        self, target_tags: set[str], *, target_uses_split: bool
    ) -> None:
        managed_tags = {
            str(item["model_tag"])
            for roles in self.resolutions.get(self.profile.id, {}).values()
            for item in roles.values()
        }
        status = self.client.status()
        for resident in tuple(status.get("residents") or ()):
            tag = str(resident.get("model_tag") or "")
            split_mode = str(resident.get("split_mode") or "none").lower()
            if (
                tag not in managed_tags
                or tag in target_tags
                or (split_mode == "none" and not target_uses_split)
            ):
                continue
            try:
                status = self.client.unload_model(tag, verification_timeout_s=30.0)
            except Exception as exc:
                raise ReconcileError(f"Turbohaul failed to unload split resident {tag}") from exc
            if _model_resident(status, tag):
                raise ReconcileError(f"Turbohaul kept split resident active: {tag}")

    def activate_local(self, rung_id: str) -> None:
        roles = self._roles(rung_id)
        tags = tuple(str(roles[role]["model_tag"]) for role in ("main", "aux") if role in roles)
        self.acquirer.ensure_tags(tags)
        self._retire_conflicting_split_residents(
            set(tags),
            target_uses_split=any(
                str(item.get("split_mode", "none")) != "none" for item in roles.values()
            ),
        )
        for role in ("main", "aux"):
            item = roles.get(role)
            if item is None:
                continue
            self.client.chat_completion(
                {
                    "model": item["model_tag"],
                    "messages": [{"role": "user", "content": "Reply exactly OK"}],
                    "max_tokens": 1,
                    "temperature": 0,
                    "keep_alive": 600,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
        self._target_rung_id = rung_id

    def activate_api(self, main_policy: str, aux_policy: str) -> None:
        if main_policy != "api:auto" or aux_policy != "api:auto":
            raise ReconcileError("unsupported API policy")
        if self._blocked_previous is None and self.route_state_path.exists():
            self._blocked_previous = json.loads(
                self.route_state_path.read_text(encoding="utf-8")
            )
        terminal_index = len(self.profile.rungs) - 1
        publish_route_state(
            self.route_state_path,
            build_route_state(
                self.profile,
                terminal_index,
                self.resolutions,
                manager_port=self.manager_port,
            ),
        )
        current = self.profile.rungs[self.current_state.rung_index]
        if current.aux_mode is not AuxMode.API:
            status = self.client.status()
            for item in self._roles(current.id).values():
                tag = str(item["model_tag"])
                if not _model_resident(status, tag):
                    continue
                try:
                    status = self.client.unload_model(tag, verification_timeout_s=30.0)
                except Exception as exc:
                    raise ReconcileError(f"Turbohaul failed to unload local model {tag}") from exc
                if _model_resident(status, tag):
                    raise ReconcileError(f"Turbohaul kept local model resident: {tag}")
        self._target_rung_id = self.profile.rungs[-1].id
        self._target_aux_mode = AuxMode.API

    def route_aux_to_main(self) -> None:
        self._target_aux_mode = AuxMode.SHARED_MAIN

    def route_aux_dedicated(self) -> None:
        self._target_aux_mode = AuxMode.DEDICATED

    def verify_rung(self, rung_id: str) -> bool:
        rung = next((item for item in self.profile.rungs if item.id == rung_id), None)
        if rung is None:
            return False
        if rung.aux_mode is AuxMode.API:
            return self._target_aux_mode is AuxMode.API
        try:
            roles = self._roles(rung_id)
        except ReconcileError:
            return False
        deadline = self.clock() + self.verification_timeout_s
        while True:
            status = self.client.status()
            if all(_model_resident(status, str(item["model_tag"])) for item in roles.values()):
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(min(0.25, max(0.0, deadline - self.clock())))

    def publish_routes(self, state: ReconcilerState) -> None:
        route_state = build_route_state(
            self.profile,
            state.rung_index,
            self.resolutions,
            manager_port=self.manager_port,
        )
        publish_route_state(self.route_state_path, route_state)
        self.current_state = state
        self._blocked_previous = None
        self._retiring_aux_tag = None

    def restore(self, state: ReconcilerState) -> None:
        rung = self.profile.rungs[state.rung_index]
        if rung.aux_mode is AuxMode.API:
            assert rung.main_api_policy and rung.aux_api_policy
            self.activate_api(rung.main_api_policy, rung.aux_api_policy)
        else:
            self.activate_local(rung.id)
            if rung.aux_mode is AuxMode.SHARED_MAIN:
                self.route_aux_to_main()
            else:
                self.route_aux_dedicated()
        if not self.verify_rung(rung.id):
            raise ReconcileError(f"could not restore rung {rung.id}")
        self.publish_routes(state)

    def verify_restore(self, state: ReconcilerState) -> bool:
        try:
            published = json.loads(self.route_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            published.get("active") == state.profile_id
            and published.get("rung_index") == state.rung_index
            and self.verify_rung(self.profile.rungs[state.rung_index].id)
        )

    def _roles(self, rung_id: str) -> dict[str, dict[str, int | str]]:
        try:
            return self.resolutions[self.profile.id][rung_id]
        except KeyError as exc:
            raise ReconcileError(
                f"missing runtime resolution for {self.profile.id}/{rung_id}"
            ) from exc


def _model_in_flight(status: dict[str, Any], tag: str) -> bool:
    for key in ("active", "loading", "grace"):
        if _contains_model(status.get(key), tag):
            return True
    queue = status.get("queue")
    return _contains_model(queue, tag)


def _model_resident(status: dict[str, Any], tag: str) -> bool:
    return any(
        _contains_model(status.get(key), tag)
        for key in ("active", "loading", "grace", "idle_hot", "residents")
    )


def _contains_model(value: Any, tag: str) -> bool:
    if isinstance(value, str):
        return value == tag
    if isinstance(value, list):
        return any(_contains_model(item, tag) for item in value)
    if isinstance(value, dict):
        if any(value.get(key) == tag for key in ("model_tag", "model", "name", "tag")):
            return True
        return any(_contains_model(item, tag) for item in value.values())
    return False


def _matching_pids(value: Any, tag: str) -> list[int]:
    found: list[int] = []
    if isinstance(value, list):
        for item in value:
            found.extend(_matching_pids(item, tag))
    elif isinstance(value, dict):
        if any(value.get(key) == tag for key in ("model_tag", "model", "name", "tag")):
            pid = value.get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
                found.append(pid)
        for item in value.values():
            found.extend(_matching_pids(item, tag))
    return found
