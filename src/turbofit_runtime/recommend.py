"""Evidence-backed recommendations based only on physical hardware capacity."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .hardware import HardwareFingerprint
from .runtime_profile import HardwareConstraint, Turbofile

MIN_CONTEXT = 131_072
WORKING_CONTEXT = 262_144
MAX_CONTEXT = 1_048_576
INTERACTIVE_TPS = 30.0
FAST_TPS = 50.0
# Windows Vulkan drivers can report a dedicated 8 GiB device as 8,176 MiB
# (16 MiB below the binary tier). Keep the physical tier boundary strict
# apart from that documented reporting delta.
DEDICATED_VRAM_REPORTING_TOLERANCE_MB = 16
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOPOLOGY_PART_RE = re.compile(r"^(\d+)x(\d+)(?:gb)?$", re.IGNORECASE)


class NoRecommendation(LookupError):
    pass


@dataclass(frozen=True)
class EvidenceCandidate:
    profile: Turbofile
    quality_rank: int
    min_tps: float
    evidence_digest: str | None
    policy_variant: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.quality_rank, bool)
            or not isinstance(self.quality_rank, int)
            or self.quality_rank < 0
        ):
            raise ValueError("quality_rank must be a non-negative integer")
        if (
            isinstance(self.min_tps, bool)
            or not isinstance(self.min_tps, (int, float))
            or not math.isfinite(self.min_tps)
            or self.min_tps < 0
        ):
            raise ValueError("min_tps must be finite and non-negative")
        if not isinstance(self.policy_variant, str) or not self.policy_variant.strip():
            raise ValueError("policy_variant must be non-empty")

    @property
    def max_context(self) -> int:
        return max(rung.context for rung in self.profile.rungs)

    @property
    def has_evidence(self) -> bool:
        return bool(
            isinstance(self.evidence_digest, str)
            and _DIGEST_RE.fullmatch(self.evidence_digest)
        )


@dataclass(frozen=True)
class LiveRuntimeState:
    hardware: HardwareFingerprint
    current_profile_id: str | None
    current_rung_id: str | None
    free_vram_mb_by_uuid: Mapping[str, int]

    def __post_init__(self) -> None:
        copied: dict[str, int] = {}
        for uuid, value in self.free_vram_mb_by_uuid.items():
            if not isinstance(uuid, str) or not uuid:
                raise ValueError("free VRAM UUID keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("free VRAM values must be non-negative integers")
            copied[uuid] = value
        object.__setattr__(self, "free_vram_mb_by_uuid", MappingProxyType(copied))


@dataclass(frozen=True)
class RecommendationResult:
    recommended: EvidenceCandidate
    eligible: tuple[EvidenceCandidate, ...]
    current_profile_id: str | None
    current_rung_id: str | None
    policy_variant: str | None


def priority_key(
    context: int, min_tps: float, quality_rank: int
) -> tuple[int, bool, float, bool, float, bool, float]:
    """Quality → 128K → 30 tok/s → 262K → 50 tok/s → 1M → speed."""
    return (
        quality_rank,
        context >= MIN_CONTEXT,
        min(min_tps, INTERACTIVE_TPS),
        context >= WORKING_CONTEXT,
        min(min_tps, FAST_TPS),
        context >= MAX_CONTEXT,
        min_tps,
    )


def recommend(
    state: LiveRuntimeState,
    candidates: Sequence[EvidenceCandidate],
    *,
    policy_variant: str | None = None,
) -> RecommendationResult:
    """Recommend from immutable physical capacity; current free VRAM is metadata only."""
    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.has_evidence
        and (policy_variant is None or candidate.policy_variant == policy_variant)
        and hardware_satisfies(state.hardware, candidate.profile.hardware)
    )
    if not eligible:
        raise NoRecommendation("no evidence-backed profile matches physical hardware")
    ranked = tuple(
        sorted(
            eligible,
            key=lambda item: (
                priority_key(item.max_context, item.min_tps, item.quality_rank),
                item.profile.id,
            ),
            reverse=True,
        )
    )
    return RecommendationResult(
        recommended=ranked[0],
        eligible=ranked,
        current_profile_id=state.current_profile_id,
        current_rung_id=state.current_rung_id,
        policy_variant=policy_variant,
    )


def hardware_satisfies(
    hardware: HardwareFingerprint, constraint: HardwareConstraint
) -> bool:
    devices = hardware.devices
    shared_local = hardware.shared_memory_pool and constraint.accelerator.lower() == "llama.cpp-local"
    effective_device_count = constraint.min_devices if shared_local else len(devices)
    if effective_device_count < constraint.min_devices:
        return False
    if hardware.total_usable_memory_mb < _gb_to_mb(constraint.total_vram_gb):
        return False
    minimum_mb = _gb_to_mb(constraint.per_device_min_gb)
    if not shared_local:
        minimum_mb = max(0, minimum_mb - DEDICATED_VRAM_REPORTING_TOLERANCE_MB)
    capable_devices = (
        constraint.min_devices if shared_local and hardware.total_usable_memory_mb >= minimum_mb
        else sum(device.memory_total_mb >= minimum_mb for device in devices)
    )
    if capable_devices < constraint.min_devices:
        return False
    if constraint.system_ram_gb is not None:
        if hardware.system_ram_mb < _gb_to_mb(constraint.system_ram_gb):
            return False
    if not _accelerator_matches(hardware, constraint.accelerator):
        return False
    if constraint.compute_capability_min is not None:
        if shared_local:
            return False
        required = _capability_tuple(constraint.compute_capability_min)
        if any(
            _capability_tuple(device.compute_capability) < required
            for device in devices
            if device.compute_capability is not None
        ):
            return False
        if any(device.compute_capability is None for device in devices):
            return False
    if constraint.topology != "any" and not shared_local:
        expected = _parse_topology(constraint.topology)
        actual = Counter(round(device.memory_total_mb / 1024) for device in devices)
        if not _topology_satisfies(actual, expected):
            return False
    return True


def _topology_satisfies(actual: Counter[int], expected: Counter[int]) -> bool:
    """Require the advertised card set while allowing extra/larger cards."""
    available = sorted(actual.elements())
    for required in sorted(expected.elements(), reverse=True):
        candidates = [index for index, size in enumerate(available) if size >= required]
        if not candidates:
            return False
        available.pop(candidates[0])
    return True


def _accelerator_matches(hardware: HardwareFingerprint, expected: str) -> bool:
    normalized = expected.lower()
    if normalized == "llama.cpp-local":
        return not hardware.devices or bool(set(hardware.backends) & {"cuda", "metal", "rocm", "vulkan"})
    accepted: set[str] = set()
    for device in hardware.devices:
        accepted.update((device.vendor, device.backend, f"{device.vendor}-{device.backend}"))
    return normalized in accepted


def _parse_topology(value: str) -> Counter[int]:
    result: Counter[int] = Counter()
    for part in value.split("+"):
        match = _TOPOLOGY_PART_RE.fullmatch(part.strip())
        if not match:
            raise ValueError(f"invalid portable topology: {value}")
        count, memory_gb = (int(group) for group in match.groups())
        if count <= 0 or memory_gb <= 0:
            raise ValueError(f"invalid portable topology: {value}")
        result[memory_gb] += count
    return result


def _capability_tuple(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def _gb_to_mb(value: float) -> int:
    return round(value * 1024)
