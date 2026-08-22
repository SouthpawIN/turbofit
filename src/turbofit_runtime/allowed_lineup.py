"""Canonical TurboFit Check / Auto lineup.

A 9B must never be recommended. Low-VRAM machines run Bonsai 27B or
Ornith 1.5 35A3B with experts offloaded to CPU. The 700B+ MoE class is
out of scope for Auto.
"""
from __future__ import annotations

import re
from typing import Iterable

ALLOWED_MAIN = (
    "bonsai-27b",
    "bonsai-27b-q1",
    "bonsai-27b-1bit",
    "qwen3-8-27b-unleashed-ud-iq3-xxs",
    "qwen3-8-27b-unleashed-ud-q3-k-xl",
    "qwen3-8-27b-unleashed",
    "qwen3-8-27b-bf16",
    "qwen3-8-27b-q8",
)

ALLOWED_AUX = (
    "ornith-1-5-35a3b",
    "carwin-nano",
    "carwin-moe-nano",
    "auto",
)

_BANNED = re.compile(
    r"(?:^|[-_])9b(?:$|[-_])|ornith-9|qwen3\.5-9b|qwen3\.8-9b",
    re.IGNORECASE,
)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def is_banned(alias: str) -> bool:
    return bool(_BANNED.search(normalize(alias)))


def is_allowed_main(alias: str) -> bool:
    token = normalize(alias)
    if is_banned(token):
        return False
    allowed = {normalize(item) for item in ALLOWED_MAIN}
    return token in allowed or any(token.startswith(item) and "9b" not in token for item in allowed)


def is_allowed_aux(alias: str) -> bool:
    token = normalize(alias)
    if is_banned(token):
        return False
    allowed = {normalize(item) for item in ALLOWED_AUX}
    return token in allowed or token.startswith("ornith-1-5") or token.startswith("carwin")


def local_main_for_vram_gb(vram_gb: float) -> str:
    if vram_gb < 16:
        return "bonsai-27b"
    if vram_gb < 24:
        return "qwen3-8-27b-unleashed-ud-iq3-xxs"
    if vram_gb < 96:
        return "qwen3-8-27b-unleashed-ud-q3-k-xl"
    return "qwen3-8-27b-bf16"


def local_aux_for_host(*, vram_gb: float, host_ram_gb: float) -> str:
    """Ornith 1.5 35A3B with CPU expert offload whenever host RAM can hold experts."""
    if host_ram_gb >= 32:
        return "ornith-1-5-35a3b"
    return "auto"


def filter_aliases(aliases: Iterable[str], *, role: str) -> list[str]:
    checker = is_allowed_main if role == "main" else is_allowed_aux
    return [alias for alias in aliases if checker(alias)]
