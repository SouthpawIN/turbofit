"""Canonical TurboFit Check / Auto lineup.

A 9B must never be recommended. Routing:

- Apple / Metal: Qwen 3.8 27B GSQ-RCO UD-Q3_K_XL via llama.cpp at 24 GB+
  unified memory (Fit List winner, evidence
  qwen-3-8-27b-gsq-rco-iq3s-auto-262k); Ornith below 24 GB. The
  OrcaRouter Uncensored MLX band and the MLX 2-bit build (uploader: archival,
  quality collapse) are retired from the default Apple route.
- Integrated / RAM-only (non-Apple): Maple at 8 GB total, Ornith at 16 GB,
  and Qwen 3.8 27B GSQ-RCO at 24 GB+.
- Dedicated 8 GB: Maple, plus Ornith when host RAM can hold offloaded experts.
- Dedicated 16 GB+: GSQ-RCO GGUF + Ornith aux.

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
    "ornith-1-5-oq4e-mtp",
    "ornith-1-5-oq8e-mtp",
    "qwen3-8-27b-gsq-rco-iq3xxs",
    "qwen3-8-27b-gsq-rco-iq3s",
    "qwen3-8-27b-gsq-rco",
    "qwen3-8-27b-bf16",
    "qwen3-8-27b-q8",
    "qwen3-8-27b-uncensored-mlx-4bit",
    "qwen3-8-27b-uncensored-mlx-6bit",
    "qwen3-8-27b-uncensored-mlx-8bit",
    "cyber-tiel-coder-35b-a3b",
    "qwen3-8-flash-next-strata",
    "qwen3-8-flash-next-iq3xxs",
    "darwin-180b-rsi-strata",
    "arete-27b-q3",
    "arete-27b-q4",
    "arete-27b-q5",
    "arete-27b-q8",
    "nanbeige-33b-iq3xs",
    "nanbeige-33b-iq4xs",
    "flint-40a2-5b-mix12g",
    "flint-40a2-5b-mix24g",
    "deepseek-v4-1-flash-1017-iq1m",
)

ALLOWED_AUX = (
    "maple-preview-tq2",
    "ornith-1-5-35a3b",
    "ornith-1-5-oq4e-mtp",
    "ornith-1-5-oq8e-mtp",
    "carwin-nano",
    "carwin-moe-nano",
    "qwen3-8-flash-next-strata",
    "auto",
)

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
    """Fit List route for Apple unified memory.

    24 GB+ shared memory runs Qwen 3.8 27B GSQ-RCO UD-Q3_K_XL
    (evidence: qwen-3-8-27b-gsq-rco-iq3s-auto-262k), not the
    retired OrcaRouter Uncensored MLX band. Sub-24 GB Macs stay Ornith;
    the MLX 2-bit build is banned.
    """
    if unified_ram_gb >= 24:
        return "qwen3-8-27b-gsq-rco-iq3s"
    return "ornith-1-5-35a3b"


def local_main_for_vram_gb(vram_gb: float) -> str:
    if vram_gb < 16:
        return "maple-preview-tq2"
    if vram_gb < 24:
        return "qwen3-8-27b-gsq-rco-iq3xxs"
    if vram_gb < 96:
        return "qwen3-8-27b-gsq-rco-iq3s"
    return "qwen3-8-27b-bf16"


def speed_main_for_vram_gb(vram_gb: float) -> str:
    """Faster MoE over dense 27B when the user picks speed."""
    if vram_gb < 16:
        return "maple-preview-tq2"
    if vram_gb < 24:
        return "ornith-1-5-35a3b"
    if vram_gb < 48:
        return "ornith-1-5-oq4e-mtp"
    return "ornith-1-5-oq8e-mtp"


def speed_family_rank(alias: str) -> int:
    token = normalize(alias)
    if "oq8e" in token:
        return 50
    if "oq4e" in token:
        return 45
    if token.startswith("ornith"):
        return 40
    if token.startswith("maple"):
        return 30
    if "gsq-rco" in token or token.startswith("qwen3-8-27b"):
        return 5
    return 10


def shared_main_for_total_memory_gb(total_memory_gb: float) -> str:
    """Select by the one shared pool; never interpret it as dedicated VRAM."""
    if total_memory_gb < 16:
        return "maple-preview-tq2"
    if total_memory_gb < 24:
        return "ornith-1-5-35a3b"
    return "qwen3-8-27b-gsq-rco-iq3s"


def local_aux_for_host(*, vram_gb: float, host_ram_gb: float) -> str:
    """Default aux is Maple Preview (fast RAM/disk-streaming 20B-A1B MoE).

    Maple TQ2_0 is tiny (~6 GB on disk), streams from host RAM / disk, and
    never blocks the main model's VRAM — ideal as a lightweight auxiliary
    across every VRAM tier. Ornith MTP variants remain available as an
    explicit opt-in for boxes that can stack the extra model alongside main.
    """
    return "maple-preview-tq2"


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
        is_gsq_rco = main == "qwen3-8-27b-gsq-rco-iq3s"
        return [
            {
                "role": "main",
                "alias": main,
                "engine": "llama.cpp",
                "repo": None,
                "revision": None,
                "why": (
                    "Apple 24 GB+ unified: GSQ-RCO IQ3_S per the Fit List "
                    "(OrcaRouter Uncensored MLX band is retired)."
                    if is_gsq_rco
                    else "Apple under 24 GB unified: Ornith 1.5, not MLX 2-bit."
                ),
                "residency": residency if main == "ornith-1-5-35a3b" else None,
            },
            {
                "role": "aux",
                "alias": "maple-preview-tq2",
                "engine": "llama.cpp",
                "why": (
                    "Maple Preview 20B-A1B streams from host RAM / disk and "
                    "never blocks the main model's VRAM — default auxiliary."
                ),
                "residency": {"mode": "ram", "cpu_moe": True, "mmap": True, "mlock": False},
            },
        ]

    if unified:
        main = shared_main_for_total_memory_gb(host_ram_gb)
        options = [
            {
                "role": "main",
                "alias": main,
                "engine": "llama.cpp",
                "why": (
                    "Integrated/unified memory uses total-memory bands: "
                    "Maple at 8 GB, Ornith at 16 GB, GSQ-RCO at 24 GB+."
                ),
                "residency": residency if main == "ornith-1-5-35a3b" else None,
            },
        ]
        # Dense exception: a 27B at q3+ fits a 16 GB unified pool, but it is not
        # the default because MoE decodes far faster on shared memory.
        if host_ram_gb >= 16:
            dense = "arete-27b-q4" if host_ram_gb >= 24 else "arete-27b-q3"
            options.append(
                {
                    "role": "main",
                    "alias": dense,
                    "engine": "llama.cpp",
                    "why": (
                        "Dense-exception option: Arete 27B at q3+ fits the unified "
                        "pool, but expect fewer tokens/sec than the MoE default."
                    ),
                }
            )
        return options + [
            {
                "role": "aux",
                "alias": "maple-preview-tq2",
                "engine": "llama.cpp",
                "why": (
                    "Maple Preview 20B-A1B streams from host RAM / disk and "
                    "never blocks the main model's VRAM — default auxiliary."
                ),
                "residency": {"mode": "ram", "cpu_moe": True, "mmap": True, "mlock": False},
            },
        ]

    if vram_gb < 16:
        options: list[dict[str, object]] = [
            {
                "role": "main",
                "alias": "maple-preview-tq2",
                "engine": "llama.cpp",
                "why": "20B-A1B Maple TQ2_0 fits within the dedicated 8 GB VRAM band.",
            },
        ]
        if vram_gb >= 12:
            options.append(
                {
                    "role": "main",
                    "alias": "arete-27b-q3",
                    "engine": "llama.cpp",
                    "why": (
                        "Arete 27B Q3_K_S (~11.6 GiB weights) fits a 12 GB card at "
                        "reduced context; the smallest tier that clears q3."
                    ),
                }
            )
        if host_ram_gb >= 16:
            options.append(
                {
                    "role": "main",
                    "alias": "ornith-1-5-35a3b",
                    "engine": "llama.cpp",
                    "why": (
                        "35A3B MoE alternative when host RAM can hold offloaded "
                        f"experts via {residency['mode']}-backed mmap."
                    ),
                    "residency": residency,
                }
            )
        return options
    options = [
        {
            "role": "main",
            "alias": local_main_for_vram_gb(vram_gb),
            "engine": "llama.cpp",
            "why": "Intelligence / balanced: GSQ-RCO / Qwen band for this VRAM.",
        },
        {
            "role": "main",
            "alias": speed_main_for_vram_gb(vram_gb),
            "engine": "llama.cpp",
            "why": "Speed: Ornith MoE / MTP instead of dense 27B.",
        },
    ]
    # Named alternatives across the matrix. These never displace the band default;
    # they are the "every model this machine can run" set from the Fit List.
    if vram_gb < 24:
        options.append(
            {
                "role": "main",
                "alias": "arete-27b-q4",
                "engine": "llama.cpp",
                "why": "Arete 27B Q4_K_M (~16 GiB) at 131K context on a 16-23 GB card.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "flint-40a2-5b-mix12g",
                "engine": "llama.cpp",
                "why": "Flint 40A2.5B Mix-12G: MoE decode speed at a 12 GB footprint.",
            }
        )
    elif vram_gb < 64:
        options.append(
            {
                "role": "main",
                "alias": "arete-27b-q5",
                "engine": "llama.cpp",
                "why": "Arete 27B Q5_K_M (~19 GiB) at 262K context: the dense quality pick.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "nanbeige-33b-iq4xs",
                "engine": "llama.cpp",
                "why": "Nanbeige 4.3 33B IQ4_XS (~17 GiB): a 33B dense alternative.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "flint-40a2-5b-mix24g",
                "engine": "llama.cpp",
                "why": "Flint 40A2.5B Mix-24G for the larger card in this band.",
            }
        )
    else:
        options.append(
            {
                "role": "main",
                "alias": "arete-27b-q8",
                "engine": "llama.cpp",
                "why": "Arete 27B Q8_0 (~26.7 GiB): near-lossless dense at 64 GB+.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "qwen3-8-flash-next-iq3xxs",
                "engine": "strata",
                "why": "Qwen 3.8 Flash Next IQ3_XXS (~70 GiB) served through Strata expert tiering.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "flint-40a2-5b-mix24g",
                "engine": "llama.cpp",
                "why": "Flint 40A2.5B Mix-24G for the larger card in this band.",
            }
        )
    options.append(
        {
            "role": "aux",
            "alias": local_aux_for_host(vram_gb=vram_gb, host_ram_gb=host_ram_gb),
            "engine": "llama.cpp",
            "why": "Default aux is MTP Ornith when total VRAM can hold it next to main.",
            "residency": residency,
        }
    )
    return options


def filter_aliases(aliases: Iterable[str], *, role: str) -> list[str]:
    checker = is_allowed_main if role == "main" else is_allowed_aux
    return [alias for alias in aliases if checker(alias)]
