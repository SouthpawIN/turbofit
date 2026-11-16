"""Manual and hardware-auto selection of adaptive Turbofile profiles."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

from .hardware import HardwareFingerprint
from .profile_io import load_profile
from .recommend import hardware_satisfies
from .runtime_profile import AuxMode, Turbofile


class SelectionMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


def _gib(value: int) -> str:
    """Short human size used in selection reasons (``41.2 GiB``)."""
    return f"{value / (1024 ** 3):.1f} GiB"


@dataclass(frozen=True)
class ArtifactPresence:
    """What the artifact machinery knows about one rung's weights on this disk.

    Automatic selection may not pick a configuration it cannot launch: this is
    the answer it consults before ranking, so "fits the tier" and "is on disk"
    are one decision rather than two.
    """

    present: bool
    missing_bytes: int = 0
    model: str = ""
    detail: str = ""


ArtifactProbe = Callable[["Turbofile", int], ArtifactPresence]


class WaitingForDownload(ValueError):
    """No automatic candidate has its artifacts on disk.

    The controller reports this as ``waiting-download`` with the missing byte
    count instead of retrying a launch that cannot succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        recommended_profile_id: str = "",
        recommended_rung_id: str = "",
        model: str = "",
        missing_bytes: int = 0,
    ) -> None:
        super().__init__(message)
        self.reason = message
        self.recommended_profile_id = recommended_profile_id
        self.recommended_rung_id = recommended_rung_id
        self.model = model
        self.missing_bytes = missing_bytes


@dataclass(frozen=True)
class ProfileChoice:
    mode: SelectionMode
    profile: Turbofile
    initial_rung_index: int
    target_ceiling_index: int = 0
    reason: str = ""
    recommended_profile_id: str = ""
    recommended_rung_id: str = ""
    missing_bytes: int = 0


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


def load_or_create_selection(
    path: str | Path,
    catalog: "ProfileCatalog",
    hardware: HardwareFingerprint,
    *,
    probe: "ArtifactProbe | None" = None,
) -> tuple[dict[str, Any], bool]:
    """Return the persisted selection, persisting a default one on first run.

    A first run has no record at all. Synthesis is chosen over idling because
    the controller can only serve a profile it has selected: treating absence
    as "no selection yet" would leave a fresh install permanently unable to
    serve, which reads as an operational failure forever. The synthesised
    record is the same automatic choice ``turbofit-runtime set auto`` writes,
    so a later explicit ``set`` simply overwrites it.

    Returns ``(selection, created)``. Raises only when no profile fits the
    hardware, which is a real failure rather than an unconfigured machine.
    """
    try:
        return load_selection(path), False
    except FileNotFoundError:
        choice = catalog.select(hardware, requested="auto", probe=probe)
        save_selection(path, choice)
        return load_selection(path), True


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

    def without_rungs(
        self, drop: Mapping[str, set[str]]
    ) -> tuple["ProfileCatalog", dict[str, str]]:
        """Drop named rungs from profiles, reporting why each one went.

        A ladder may not be emptied: a profile that loses every local rung is
        kept intact so the caller can report a real configuration fault
        instead of silently serving an empty ladder.
        """
        problems: dict[str, str] = {}
        filtered: list[Turbofile] = []
        for profile in self._profiles:
            dropped = drop.get(profile.id, set())
            if not dropped:
                filtered.append(profile)
                continue
            kept = [rung for rung in profile.rungs if rung.id not in dropped]
            if not kept or not any(rung.aux_mode is not AuxMode.API for rung in kept):
                filtered.append(profile)
                continue
            for rung_id in dropped:
                problems[f"{profile.id}/{rung_id}"] = "unavailable"
            rungs = tuple(kept)
            selected = replace(
                profile,
                rungs=rungs,
                revision=profile.revision + (rungs != profile.rungs),
            )
            selected.validate()
            filtered.append(selected)
        return ProfileCatalog(filtered), problems

    def select(
        self,
        hardware: HardwareFingerprint,
        *,
        requested: str = "auto",
        probe: ArtifactProbe | None = None,
    ) -> ProfileChoice:
        if requested == "auto":
            return self._auto_choice(hardware, probe)
        profile = self._by_id.get(requested)
        if profile is None:
            raise ValueError(f"unknown profile: {requested}")
        if self._has_local_rung(profile) and not hardware_satisfies(hardware, profile.hardware):
            raise ValueError(
                f"manual profile {requested} does not fit physical hardware {hardware.topology_key}"
            )
        return ProfileChoice(
            mode=SelectionMode.MANUAL,
            profile=profile,
            initial_rung_index=len(profile.rungs) - 1,
            target_ceiling_index=0,
            reason=f"manually requested {profile.id}",
        )

    def _auto_choice(
        self,
        hardware: HardwareFingerprint,
        probe: ArtifactProbe | None,
    ) -> ProfileChoice:
        """Pick the best tier fit whose weights are actually on this disk.

        Ranking by fit alone is what made a fresh host select a model it had
        never downloaded and fail every tick. With a probe, the recommended
        (best-fit) configuration is still named -- so the record can say what
        is missing -- but only a present one is ever returned.
        """
        exact = [
            profile
            for profile in self._profiles
            if hardware_satisfies(hardware, profile.hardware)
        ]
        if not exact:
            return self._auto_api_choice(hardware)
        ranked = sorted(exact, key=self._auto_rank, reverse=True)
        recommended = next(
            (profile for profile in ranked if self._has_local_rung(profile)), None
        )
        if recommended is None:
            # Every fitting profile is API-only by design: no weights to want.
            return self._auto_api_choice(hardware, exact)
        if probe is None:
            return ProfileChoice(
                mode=SelectionMode.AUTO,
                profile=recommended,
                initial_rung_index=len(recommended.rungs) - 1,
                target_ceiling_index=0,
                reason=(
                    f"highest tier fit for {hardware.topology_key}: "
                    f"{recommended.id}/{recommended.rungs[len(recommended.rungs) - 1].id}"
                ),
            )
        recommended_index = self._top_local_rung_index(recommended)

        recommended_presence = probe(recommended, recommended_index)
        recommended_model = (
            recommended_presence.model or recommended.rungs[recommended_index].id
        )
        for profile in ranked:
            index = self._present_rung_index(profile, probe)
            if index is None:
                continue
            presence = probe(profile, index)
            chosen_model = presence.model or profile.rungs[index].id
            reason = f"best present artifact at this tier: {chosen_model}"
            if profile.id != recommended.id or index != recommended_index:
                short = (
                    f"{_gib(recommended_presence.missing_bytes)} short"
                    if recommended_presence.missing_bytes
                    else "not on disk"
                )
                reason += f"; recommended {recommended_model} is {short}"
            else:
                reason += f" ({profile.id}/{profile.rungs[index].id})"
            return ProfileChoice(
                mode=SelectionMode.AUTO,
                profile=profile,
                initial_rung_index=index,
                target_ceiling_index=0,
                reason=reason,
                recommended_profile_id=recommended.id,
                recommended_rung_id=recommended.rungs[recommended_index].id,
                missing_bytes=recommended_presence.missing_bytes,
            )

        short = (
            f"{_gib(recommended_presence.missing_bytes)} short"
            if recommended_presence.missing_bytes
            else "of unknown size"
        )
        raise WaitingForDownload(
            f"waiting-download: no tier-fitting artifact is on disk; "
            f"recommended {recommended_model} is {short}",
            recommended_profile_id=recommended.id,
            recommended_rung_id=recommended.rungs[recommended_index].id,
            model=recommended_model,
            missing_bytes=recommended_presence.missing_bytes,
        )

    def _present_rung_index(
        self, profile: Turbofile, probe: ArtifactProbe
    ) -> int | None:
        """Highest local rung of *profile* whose artifacts are complete.

        API rungs are skipped: they need no weights, so treating them as
        "present" would answer a missing-download with a cloud fallback instead
        of the local model one tier down that is actually installed.
        """
        for index in range(len(profile.rungs) - 1, -1, -1):
            if profile.rungs[index].aux_mode is AuxMode.API:
                continue
            if probe(profile, index).present:
                return index
        return None

    @staticmethod
    def _top_local_rung_index(profile: Turbofile) -> int:
        """Index of the highest local (weights-bearing) rung of *profile*."""
        for index in range(len(profile.rungs) - 1, -1, -1):
            if profile.rungs[index].aux_mode is not AuxMode.API:
                return index
        return len(profile.rungs) - 1

    def _auto_api_choice(
        self, hardware: HardwareFingerprint, candidates: Iterable[Turbofile] | None = None
    ) -> ProfileChoice:
        pool = list(candidates) if candidates is not None else list(self._profiles)
        total_gb = hardware.total_usable_memory_mb / 1024
        safe_api = [
            profile
            for profile in pool
            if not self._has_local_rung(profile)
            and profile.hardware.class_vram_gb <= total_gb
        ]
        if safe_api:
            profile = max(safe_api, key=lambda item: (item.hardware.class_vram_gb, item.id))
            return ProfileChoice(
                mode=SelectionMode.AUTO,
                profile=profile,
                initial_rung_index=len(profile.rungs) - 1,
                target_ceiling_index=0,
                reason=f"no local profile fits; API-only fallback {profile.id}",
            )
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
