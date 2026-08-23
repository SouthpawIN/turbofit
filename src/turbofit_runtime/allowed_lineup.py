"""Canonical TurboFit Check / Auto lineup.

A 9B must never be recommended. Routing:

- Apple / Metal: OrcaRouter Qwen3.8-27B-Uncensored MLX (4/6/8-bit). Never the
  MLX 2-bit build (uploader: archival, quality collapse).
- Integrated / unified (non-Apple): Ornith 1.5 35A3B with --cpu-moe + mmap.
- Low-memory dedicated: Bonsai 27B Q1 with safe host spill and/or Ornith 1.5 streamed from disk.
- Dedicated 16 GB+: Unleashed GGUF + Ornith aux.

The Asha/Escha mixed 2-bit (EschaLabs/Qwen3.8-27B-Escha-W2) is a research
candidate only: custom SGLang kernels, no llama.cpp recipe, no TurboFit TPS.
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
    "qwen3-8-27b-uncensored-mlx-4bit",
    "qwen3-8-27b-uncensored-mlx-6bit",
    "qwen3-8-27b-uncensored-mlx-8bit",
)

ALLOWED_AUX = (
    "ornith-1-5-35a3b",
    "carwin-nano",
    "carwin-moe-nano",
    "auto",
)

MLX_REPO = "orcarouter/Qwen3.8-27B-Uncensored-MLX"
MLX_REVISION = "b4603df5fd2a51e7fed2560ee7090caa4e13e4b7"

_BANNED = re.compile(
    r"(?:^|[-_])9b(?:$|[-_])|ornith-9|qwen3\.5-9b|qwen3\.8-9b|mlx-2bit|uncensored-mlx-2",
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
    return (
        token in allowed
        or token.startswith("auto-")
        or token.startswith("ornith-1-5")
        or token.startswith("carwin")
    )


def expert_residency(*, host_ram_gb: float) -> dict[str, object]:
    if host_ram_gb >= 32:
        return {"mode": "ram", "cpu_moe": True, "mmap": True, "mlock": False}
    return {"mode": "disk", "cpu_moe": True, "mmap": True, "mlock": False}


def apple_mlx_main(*, unified_ram_gb: float) -> str:
    """OrcaRouter Uncensored MLX. 2-bit is banned. Sub-24 GB Mac stays Ornith."""
    if unified_ram_gb >= 48:
        return "qwen3-8-27b-uncensored-mlx-8bit"
    if unified_ram_gb >= 32:
        return "qwen3-8-27b-uncensored-mlx-6bit"
    if unified_ram_gb >= 24:
        return "qwen3-8-27b-uncensored-mlx-4bit"
    return "ornith-1-5-35a3b"


def local_main_for_vram_gb(vram_gb: float) -> str:
    if vram_gb < 16:
        return "bonsai-27b"
    if vram_gb < 24:
        return "qwen3-8-27b-unleashed-ud-iq3-xxs"
    if vram_gb < 96:
        return "qwen3-8-27b-unleashed-ud-q3-k-xl"
    return "qwen3-8-27b-bf16"


def local_aux_for_host(*, vram_gb: float, host_ram_gb: float) -> str:
    if vram_gb < 16:
        return "auto"
    return "ornith-1-5-35a3b"


def low_vram_moe_main() -> str:
    return "ornith-1-5-35a3b"


def check_local_options(
    *,
    vram_gb: float,
    host_ram_gb: float,
    memory_pool: str = "dedicated",
    backend: str = "cuda",
    vendor: str = "nvidia",
) -> list[dict[str, object]]:
    residency = expert_residency(host_ram_gb=host_ram_gb)
    apple = vendor == "apple" or backend == "metal"
    unified = memory_pool == "unified" or apple

    if apple:
        main = apple_mlx_main(unified_ram_gb=host_ram_gb)
        return [
            {
                "role": "main",
                "alias": main,
                "engine": "mlx" if main.startswith("qwen3-8-27b-uncensored-mlx") else "llama.cpp",
                "repo": MLX_REPO if main.startswith("qwen3-8-27b-uncensored-mlx") else None,
                "revision": MLX_REVISION if main.startswith("qwen3-8-27b-uncensored-mlx") else None,
                "why": (
                    "Apple Silicon: OrcaRouter Uncensored MLX (4/6/8-bit). "
                    "MLX 2-bit is banned."
                    if main.startswith("qwen3-8-27b-uncensored-mlx")
                    else "Apple under 24 GB unified: Ornith 1.5, not MLX 2-bit."
                ),
                "residency": residency if main == "ornith-1-5-35a3b" else None,
            }
        ]

    if unified:
        return [
            {
                "role": "main",
                "alias": "ornith-1-5-35a3b",
                "why": (
                    "Integrated/unified memory: Ornith 1.5 35A3B with --cpu-moe "
                    f"and mmap from {residency['mode']}."
                ),
                "residency": residency,
            }
        ]

    if vram_gb < 16:
        return [
            {
                "role": "main",
                "alias": "bonsai-27b",
                "why": "Bonsai 27B Q1 shared-main lane with safe host spill when full GPU residency is unavailable.",
            },
            {
                "role": "main",
                "alias": "ornith-1-5-35a3b",
                "why": (
                    "35A3B MoE. Experts stay off GPU (--cpu-moe) and mmap "
                    f"from {residency['mode']} so constrained VRAM and host RAM remain a testable lane."
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
