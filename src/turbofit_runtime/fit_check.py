"""Live-refit fit check: f(available_memory) -> Fit List tier profile.

Authority: Fit Lists v1 (two separate lists).

Dedicated (GPU VRAM) — dense models preferred:
    8-11 GB   Bonsai 2 PTQ1_0 / PQ2_0 (prism-llama.cpp)
    12-15 GB  Underdog 27B PQ2_0 (prism-llama.cpp)
    16-23 GB  Qwen 3.8 27B GSQ-RCO IQ3_S (llama.cpp, 131K)
    24-63 GB  Qwen 3.8 27B GSQ-RCO IQ3_S (llama.cpp, 262K)
    64+ GB    Strata Flash Next full residency

Unified / CPU — MoE models preferred (dense 27B crawls on CPU):
    8-15 GB   Maple Preview TQ2_0 (20B-A1B ternary MoE)
    16-23 GB  Cyber-Tiel Coder UD-IQ3_XXS (35B-A3B)
    24-31 GB  Cyber-Tiel Coder UD-Q4_K_XL
    32-47 GB  Cyber-Tiel Coder UD-Q5_K_XL
    48-63 GB  Cyber-Tiel Coder UD-Q6_K_XL
    64+ GB    Strata Flash Next

The fit function is the single target-selection input for contraction
(live re-fit under pressure) and direct expansion. The ban list
(no 9B) applies to every selection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_BAN_LIST = frozenset({"9b"})
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


# Dedicated VRAM ladder (dense models preferred):
#   8-11 GB   Bonsai 2 PTQ1_0 / PQ2_0
#   12-15 GB  Underdog 27B PQ2_0
#   16-63 GB  Qwen 3.8 27B GSQ-RCO IQ3_S (llama.cpp)
#             16-23 GB at 131K, 24-63 GB at 262K (measured: 49.2 tok/s @ 48GB)
#   64+ GB    Strata Flash Next (125B MoE, needs 64+ GB for expert residency)
DEDICATED_TIERS: tuple[TierProfile, ...] = (
    _tier("bonsai-2-ptq1-8gb", "dedicated", 8, 11, "bonsai-2", "ptq1"),
    _tier("bonsai-2-pq2-8gb", "dedicated", 8, 11, "bonsai-2", "pq2"),
    _tier("underdog-pq2-12gb", "dedicated", 12, 15, "underdog", "pq2"),
    _tier("gsq-rco-iq3s-16gb", "dedicated", 16, 23, "gsq-rco", "iq3-s"),
    _tier("gsq-rco-iq3s-24gb", "dedicated", 24, 63, "gsq-rco", "iq3-s"),
    _tier("strata-flash-next-64gb", "dedicated", 64, None, "strata", "flash-next"),
)
# Unified / CPU ladder (MoE models preferred — dense 27B crawls on CPU):
#   8-15 GB   Maple Preview TQ2_0
#   16-23 GB  Cyber-Tiel Coder UD-IQ3_XXS
#   24-31 GB  Cyber-Tiel Coder UD-Q4_K_XL
#   32-47 GB  Cyber-Tiel Coder UD-Q5_K_XL
#   48-63 GB  Cyber-Tiel Coder UD-Q6_K_XL
#   64+ GB    Strata Flash Next
SHARED_TIERS: tuple[TierProfile, ...] = (
    _tier("maple-tq2-8gb", "shared", 8, 15, "maple", "tq2"),
    _tier("cyber-tiel-iq3-xxs-16gb", "shared", 16, 23, "cyber-tiel", "iq3-xxs"),
    _tier("cyber-tiel-q4-k-xl-24gb", "shared", 24, 31, "cyber-tiel", "q4-k-xl"),
    _tier("cyber-tiel-q5-k-xl-32gb", "shared", 32, 47, "cyber-tiel", "q5-k-xl"),
    _tier("cyber-tiel-q6-k-xl-48gb", "shared", 48, 63, "cyber-tiel", "q6-k-xl"),
    _tier("strata-flash-next-64gb", "shared", 64, None, "strata", "flash-next"),
)
TIERS_BY_CLASS = {"shared": SHARED_TIERS, "dedicated": DEDICATED_TIERS}


def rung_tokens(rung_id: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_SPLIT.split(rung_id.lower()) if token)


def rung_matches_tier(rung_id: str, model_tokens: tuple[str, ...]) -> bool:
    """True when the rung id carries this tier's model identity.

    Substring match on the normalized rung id, not token-set intersection:
    the tokenizer splits on '-', which breaks hyphenated quant names
    ("cyber-tiel", "iq3-xxs", "q4-k-xl", "gsq-rco", "iq3-s", "tq2", "pq2").

    All tokens must match (family AND quant), otherwise any cyber-tiel rung
    matches any cyber-tiel tier regardless of quantization.
    """
    normalized = rung_id.lower()
    return all(token in normalized for token in model_tokens)


def rung_is_banned(rung_id: str, ban_list: frozenset[str] = DEFAULT_BAN_LIST) -> bool:
    """Ban-list check for a rung id (retired families only, no 9B by default).

    Token match on the rung id. Retired Bonsai 1 rungs carry the token
    "bonsai"; Bonsai 2 rungs carry "bonsai-2" which tokenizes to
    {bonsai, 2} — an exact token hit on "bonsai" still matches Bonsai 2.
    The caller must not put "bonsai" in the ban list alongside a live
    Bonsai 2 tier; use "bonsai-1" for the retired family instead.
    """
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
        if rung_matches_tier(rung_id, tier.model_tokens):
            return index
    return None
