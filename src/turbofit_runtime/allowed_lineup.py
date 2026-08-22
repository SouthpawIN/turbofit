"""Canonical TurboFit Check / Auto lineup.

A 9B must never be recommended. Low-VRAM machines run Bonsai 27B or
Ornith 1.5 35A3B. Ornith experts stay off the GPU (`--cpu-moe`) and are
mmapped so an 8 GB card with 8-16 GB RAM streams cold experts from disk
instead of requiring 32 GB of host RAM. The 700B+ MoE class is out of
Auto.
"""
from __future__ import annotations

import re
from typing import Iterable

ALLOWED_MAIN = (
    "bonsai-27b",
    "bonsai-27b-q1",
    "bonsai-27b-1bit",
    "ornith-1-5-35a3b",
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


def expert_residency(*, host_ram_gb: float) -> dict[str, object]:
    """How Ornith 1.5 experts live on a small box.

    ram: host can hold the 21 GB Q4 experts.
    disk: mmap + no mlock; cold experts stream from the drive.
    """
    if host_ram_gb >= 32:
        return {
            "mode": "ram",
            "cpu_moe": True,
            "mmap": True,
            "mlock": False,
        }
    return {
        "mode": "disk",
        "cpu_moe": True,
        "mmap": True,
        "mlock": False,
    }


def local_main_for_vram_gb(vram_gb: float) -> str:
    if vram_gb < 16:
        return "bonsai-27b"
    if vram_gb < 24:
        return "qwen3-8-27b-unleashed-ud-iq3-xxs"
    if vram_gb < 96:
        return "qwen3-8-27b-unleashed-ud-q3-k-xl"
    return "qwen3-8-27b-bf16"


def local_aux_for_host(*, vram_gb: float, host_ram_gb: float) -> str:
    """Ornith 1.5 is the aux MoE. On 8 GB VRAM it cannot share the card with
    Bonsai, so aux becomes shared-main. On more VRAM, Ornith streams experts
    from RAM or disk — never shrinks to a 9B.
    """
    if vram_gb < 16:
        return "auto"
    return "ornith-1-5-35a3b"


def low_vram_moe_main() -> str:
    """8 GB MoE option: Ornith 1.5 with experts mmapped from disk or RAM."""
    return "ornith-1-5-35a3b"


def check_local_options(*, vram_gb: float, host_ram_gb: float) -> list[dict[str, object]]:
    """Ranked Check options. 8 GB boxes get Bonsai and disk-streamed Ornith."""
    residency = expert_residency(host_ram_gb=host_ram_gb)
    if vram_gb < 16:
        return [
            {
                "role": "main",
                "alias": "bonsai-27b",
                "why": "Dense 27B that fits an 8 GB card without offload.",
            },
            {
                "role": "main",
                "alias": "ornith-1-5-35a3b",
                "why": (
                    "35A3B MoE. Experts stay off GPU (--cpu-moe) and mmap "
                    f"from {residency['mode']} so 8 GB VRAM + <32 GB RAM still works."
                ),
                "residency": residency,
            },
        ]
    return [
        {
            "role": "main",
            "alias": local_main_for_vram_gb(vram_gb),
            "why": "Unleashed / Qwen band for this VRAM.",
        },
        {
            "role": "aux",
            "alias": "ornith-1-5-35a3b",
            "why": "35A3B aux with CPU/disk expert stream.",
            "residency": residency,
        },
    ]


def filter_aliases(aliases: Iterable[str], *, role: str) -> list[str]:
    checker = is_allowed_main if role == "main" else is_allowed_aux
    return [alias for alias in aliases if checker(alias)]
