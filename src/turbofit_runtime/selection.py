"""Manual and hardware-auto selection of adaptive Turbofile profiles."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from .hardware import HardwareFingerprint
from .profile_io import load_profile
from .recommend import hardware_satisfies
from .runtime_profile import AuxMode, Turbofile


class SelectionMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class ProfileChoice:
    mode: SelectionMode
    profile: Turbofile
    initial_rung_index: int
    target_ceiling_index: int = 0


def save_selection(path: str | Path, choice: ProfileChoice) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "turbofit.selection/v1",
        "mode": choice.mode.value,
        "requested": "auto" if choice.mode is SelectionMode.AUTO else choice.profile.id,
        "profile_id": choice.profile.id,
        "target_ceiling_index": choice.target_ceiling_index,
        "initial_rung_index": choice.initial_rung_index,
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


def load_selection(path: str | Path) -> dict[str, Any]:
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {
        "schema",
        "mode",
        "requested",
        "profile_id",
        "target_ceiling_index",
        "initial_rung_index",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("invalid selection record")
    if raw["schema"] != "turbofit.selection/v1":
        raise ValueError("unsupported selection schema")
    try:
        mode = SelectionMode(raw["mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid selection mode") from exc
    requested = raw["requested"]
    profile_id = raw["profile_id"]
    if not isinstance(requested, str) or not requested:
        raise ValueError("selection requested value must be non-empty")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("selection profile_id must be non-empty")
    if mode is SelectionMode.AUTO and requested != "auto":
        raise ValueError("automatic selection must request auto")
    for field in ("target_ceiling_index", "initial_rung_index"):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"selection {field} must be a non-negative integer")
    return dict(raw)


class ProfileCatalog:
    """Immutable catalog that applies one safety contract to auto and manual choices."""

    def __init__(self, profiles: Iterable[Turbofile]) -> None:
        ordered = tuple(sorted(profiles, key=lambda item: item.id))
        if not ordered:
            raise ValueError("profile catalog must not be empty")
        ids = [profile.id for profile in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate profile id")
        self._profiles = ordered
        self._by_id = {profile.id: profile for profile in ordered}

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "ProfileCatalog":
        return cls(load_profile(path) for path in paths)

    @property
    def profiles(self) -> tuple[Turbofile, ...]:
        return self._profiles

    def without_deferred_models(
        self,
        resolutions: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int | str]]]],
        deferred_model_ids: set[str],
    ) -> "ProfileCatalog":
        """Remove local rungs routed to deferred models while retaining safe rungs."""
        if not deferred_model_ids:
            return self
        filtered: list[Turbofile] = []
        for profile in self.profiles:
            kept = []
            for rung in profile.rungs:
                roles = resolutions.get(profile.id, {}).get(rung.id, {})
                routed_models = {
                    str(role.get(field))
                    for role in roles.values()
                    for field in ("model_tag", "family")
                    if role.get(field) is not None
                }
                if routed_models & deferred_model_ids:
                    continue
                kept.append(rung)
            if not kept:
                raise ValueError(f"deferring models removes every rung from {profile.id}")
            rungs = tuple(kept)
            selected = replace(
                profile,
                rungs=rungs,
                revision=profile.revision + (rungs != profile.rungs),
            )
            selected.validate()
            filtered.append(selected)
        return ProfileCatalog(filtered)

    def select(self, hardware: HardwareFingerprint, *, requested: str = "auto") -> ProfileChoice:
        if requested == "auto":
            profile = self._auto(hardware)
            mode = SelectionMode.AUTO
        else:
            profile = self._by_id.get(requested)
            if profile is None:
                raise ValueError(f"unknown profile: {requested}")
            if self._has_local_rung(profile) and not hardware_satisfies(hardware, profile.hardware):
                raise ValueError(
                    f"manual profile {requested} does not fit physical hardware {hardware.topology_key}"
                )
            mode = SelectionMode.MANUAL
        return ProfileChoice(
            mode=mode,
            profile=profile,
            initial_rung_index=len(profile.rungs) - 1,
            target_ceiling_index=0,
        )

    def _auto(self, hardware: HardwareFingerprint) -> Turbofile:
        exact = [
            profile
            for profile in self._profiles
            if hardware_satisfies(hardware, profile.hardware)
        ]
        if exact:
            return max(exact, key=self._auto_rank)

        total_gb = hardware.total_usable_memory_mb / 1024
        safe_api = [
            profile
            for profile in self._profiles
            if not self._has_local_rung(profile)
            and profile.hardware.class_vram_gb <= total_gb
        ]
        if safe_api:
            return max(safe_api, key=lambda item: (item.hardware.class_vram_gb, item.id))
        raise ValueError(
            f"no safe automatic profile for physical hardware {hardware.topology_key}"
        )

    @staticmethod
    def _has_local_rung(profile: Turbofile) -> bool:
        return any(rung.aux_mode is not AuxMode.API for rung in profile.rungs)

    @staticmethod
    def _auto_rank(profile: Turbofile) -> tuple[bool, bool, float, str]:
        canonical = profile.id == f"hardware-{int(profile.hardware.class_vram_gb)}gb"
        return (
            canonical,
            ProfileCatalog._has_local_rung(profile),
            profile.hardware.class_vram_gb,
            profile.id,
        )
