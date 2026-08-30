"""Live-refit fit check: f(available_memory) -> Fit List tier profile.

Authority: Fit List v2.4 tier bands.

Shared total memory:
    Maple 8-15 GB, Ornith 16-23 GB, Unleashed UD-Q3_K_XL 24 GB+
Dedicated VRAM:
    Maple TQ2 8-15 GB, Unleashed UD-IQ3_XXS 16-23 GB,
    Unleashed UD-Q3_K_XL 24-95 GB, bf16 96 GB+

The fit function is the single target-selection input for contraction
(live re-fit under pressure) and direct expansion; the pre-baked rung
ladder is only a fallback for profiles whose rung ids carry no tier
identity. The ban list (no Bonsai, no 9B) applies to every selection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_BAN_LIST = frozenset({"bonsai", "9b"})
_TOKEN_SPLIT = re.compile(r"[^a-z0-9.]+")


@dataclass(frozen=True)
class TierProfile:
    """One Fit List tier band: usable-memory range plus model identity."""

    tier_id: str
    memory_class: str  # "shared" | "dedicated"
    min_gb: int
    max_gb: int | None  # inclusive upper bound; None = unbounded
    model_tokens: tuple[str, ...]

    def banned(self, ban_list: frozenset[str]) -> bool:
        tokens = set(self.model_tokens)
        return bool(tokens & ban_list)


def _tier(tier_id: str, memory_class: str, min_gb: int, max_gb: int | None, *tokens: str) -> TierProfile:
    return TierProfile(tier_id, memory_class, min_gb, max_gb, tuple(tokens))


SHARED_TIERS: tuple[TierProfile, ...] = (
    _tier("maple-8gb", "shared", 8, 15, "maple"),
    _tier("ornith-16gb", "shared", 16, 23, "ornith"),
    _tier("unleashed-ud-q3-k-xl-24gb", "shared", 24, None, "unleashed"),
)
DEDICATED_TIERS: tuple[TierProfile, ...] = (
    _tier("maple-tq2-8gb", "dedicated", 8, 15, "maple", "tq2"),
    _tier("unleashed-ud-iq3-xxs-16gb", "dedicated", 16, 23, "unleashed", "iq3-xxs"),
    _tier("unleashed-ud-q3-k-xl-24gb", "dedicated", 24, 95, "unleashed", "q3-k-xl"),
    _tier("bf16-96gb", "dedicated", 96, None, "bf16"),
)
TIERS_BY_CLASS = {"shared": SHARED_TIERS, "dedicated": DEDICATED_TIERS}


def rung_tokens(rung_id: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_SPLIT.split(rung_id.lower()) if token)


def rung_is_banned(rung_id: str, ban_list: frozenset[str] = DEFAULT_BAN_LIST) -> bool:
    """Ban-list check for a rung id (no Bonsai, no 9B by default)."""
    tokens = rung_tokens(rung_id)
    for banned in ban_list:
        if banned in tokens:
            return True
        if banned == "9b" and any(token.startswith("9b") for token in tokens):
            return True
    return False


def fit_tier(
    available_memory_gb: float,
    *,
    memory_class: str = "shared",
    ban_list: frozenset[str] = DEFAULT_BAN_LIST,
) -> TierProfile | None:
    """f(available_memory) -> tier profile, or None when no tier fits."""
    if not isinstance(available_memory_gb, (int, float)) or available_memory_gb < 0:
        raise ValueError("available_memory_gb must be a non-negative number")
    tiers = TIERS_BY_CLASS.get(memory_class)
    if tiers is None:
        raise ValueError(f"unknown memory class: {memory_class}")
    usable = int(available_memory_gb)
    for tier in tiers:
        if tier.min_gb <= usable and (tier.max_gb is None or usable <= tier.max_gb):
            if tier.banned(ban_list):
                return None
            return tier
    return None


def select_rung_for_tier(
    rung_ids: tuple[str, ...],
    tier: TierProfile | None,
    ban_list: frozenset[str] = DEFAULT_BAN_LIST,
) -> int | None:
    """First rung index whose id carries the tier's model identity (ban-aware).

    Returns None when no rung matches the tier; callers fall back to the
    pre-baked rung ladder in that case.
    """
    if tier is None:
        return None
    tier_tokens = set(tier.model_tokens)
    for index, rung_id in enumerate(rung_ids):
        if rung_is_banned(rung_id, ban_list):
            continue
        if tier_tokens & rung_tokens(rung_id):
            return index
    return None
