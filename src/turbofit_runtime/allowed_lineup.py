"""Canonical TurboFit Check / Auto lineup.

A 9B must never be recommended. Routing:

- Apple / Metal: OrcaRouter Qwen3.8-27B-Uncensored MLX (4/6/8-bit). Never the
  MLX 2-bit build (uploader: archival, quality collapse).
- Integrated / RAM-only (non-Apple): Maple at 8 GB total, Ornith at 16 GB,
  and Qwen 3.8 27B Unleashed at 24 GB+.
- Dedicated 8 GB: Maple, plus Ornith when host RAM can hold offloaded experts.
- Dedicated 16 GB+: Unleashed GGUF + Ornith aux.

The Asha/Escha mixed 2-bit (EschaLabs/Qwen3.8-27B-Escha-W2) is a research
candidate only: custom SGLang kernels, no llama.cpp recipe, no TurboFit TPS.
"""
from __future__ import annotations

import re
from typing import Iterable

ALLOWED_MAIN = (
    "maple-preview-tq2",
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
        return "maple-preview-tq2"
    if vram_gb < 24:
        return "qwen3-8-27b-unleashed-ud-iq3-xxs"
    if vram_gb < 96:
        return "qwen3-8-27b-unleashed-ud-q3-k-xl"
    return "qwen3-8-27b-bf16"


def shared_main_for_total_memory_gb(total_memory_gb: float) -> str:
    """Select by the one shared pool; never interpret it as dedicated VRAM."""
    if total_memory_gb < 16:
        return "maple-preview-tq2"
    if total_memory_gb < 24:
        return "ornith-1-5-35a3b"
    return "qwen3-8-27b-unleashed-ud-q3-k-xl"


def local_aux_for_host(*, vram_gb: float, host_ram_gb: float) -> str:
    if vram_gb < 16:
        return "auto"
    return "ornith-1-5-35a3b"


def low_vram_moe_main() -> str:
    return "ornith-1-5-35a3b"


def recommendable_mains_for_hardware(hardware: object) -> frozenset[str]:
    """Mains Check may offer. Host-spill of a dense 27B is not a recommendation."""
    vram_gb = float(getattr(hardware, "total_vram_mb", 0) or 0) / 1024
    host_ram_gb = float(getattr(hardware, "system_ram_mb", 0) or 0) / 1024
    pool = str(getattr(hardware, "memory_pool_kind", "dedicated") or "dedicated")
    devices = getattr(hardware, "devices", ()) or ()
    backend = devices[0].backend if devices else "cpu"
    vendor = devices[0].vendor if devices else ""
    if pool == "cpu":
        pool = "unified"
        vram_gb = 0
    return frozenset(
        normalize(str(item["alias"]))
        for item in check_local_options(
            vram_gb=vram_gb,
            host_ram_gb=host_ram_gb,
            memory_pool=pool,
            backend=backend,
            vendor=vendor,
        )
    )


def main_matches_recommendable(alias: str, allowed: Iterable[str]) -> bool:
    token = normalize(alias)
    return any(token == item or token.startswith(f"{item}-") for item in allowed)


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
        main = shared_main_for_total_memory_gb(host_ram_gb)
        return [
            {
                "role": "main",
                "alias": main,
                "why": (
                    "Integrated/unified memory uses total-memory bands: "
                    "Maple at 8 GB, Ornith at 16 GB, Unleashed at 24 GB+."
                ),
                "residency": residency if main == "ornith-1-5-35a3b" else None,
            }
        ]

    if vram_gb < 16:
        options: list[dict[str, object]] = [
            {
                "role": "main",
                "alias": "maple-preview-tq2",
                "why": "20B-A1B Maple TQ2_0 fits within the dedicated 8 GB VRAM band.",
            },
        ]
        if host_ram_gb >= 16:
            options.append(
                {
                    "role": "main",
                    "alias": "ornith-1-5-35a3b",
                    "why": (
                        "35A3B MoE alternative when host RAM can hold offloaded "
                        f"experts via {residency['mode']}-backed mmap."
                    ),
                    "residency": residency,
                }
            )
        return options
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
