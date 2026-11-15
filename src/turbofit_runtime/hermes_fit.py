"""Bridge TurboFit onto the built-in Hermes local-model recommendation.

The built-in ``hermes_cli.local_runtime`` (``hardware`` + ``catalog`` +
``estimator``) is the authoritative local-model recommender: it scans the
machine into a ``HardwareBudget``, physics-checks every catalog entry, and
returns ``recommended_entry`` (the fastest model that fits without spill).
This module *uses that recommendation* and layers TurboFit's doctrine on top:

* **UMA / unified-memory compensation** -- the built-in budget zeroes
  ``ram_available_bytes`` on Apple Silicon and caps ``usable_vram_bytes``
  low, so a ~22 GB MoE gets physics-refused on a 32 GB unified box even
  though the unified pool *is* the host pool. We credit the full unified
  pool (minus a headroom) so the built-in pick actually lands on the
  hardware.
* **Ban filter** -- the built-in pick is dropped if TurboFit's allowlist has
  banned the model; otherwise the first allowed built-in fit is primary.
* **Ladder label** -- ``unified/moe`` (integrated) vs ``dedicated/dense``
  (discrete-GPU) so the supervisor live-refit picks the right ladder.
* **Convergence target** -- always surface ``qwen3.8-flash-next`` (the
  q=95 "Flash Next" rung) with the GB it would need, so the next rung up is
  visible on every scan.
* **Engine priority** -- llama.cpp Metal/CUDA per the allowlist + engine
  matrix.
* **Fallback** -- if the Hermes catalog can't be imported (standalone
  TurboFit on a host without Hermes), fall back to TurboFit's own ladders +
  ``check_local_options`` so the command never breaks.

Import of the built-in runtime is lazy, so the TurboFit runtime never fails
on a machine that doesn't ship Hermes.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from . import allowed_lineup

# --- built-in (Hermes local runtime) ---------------------------------------
_HERMES_PKG = "hermes_cli"
_GIB = 1024**3
_HERMES_HOME_ENV = "TURBOFIT_HERMES_HOME"
_DEFAULT_HERMES_HOME = "~/.hermes/hermes-agent"

# Headroom left off the unified pool before it is treated as usable VRAM.
# Apple leaves a fraction of unified memory to the OS / other GPU clients.
UMA_HEADROOM = 0.12

# The "next rung up" convergence target the user keeps naming.
FLASH_NEXT_ID = "qwen3.8-flash-next"

# Built-in catalog id -> TurboFit allowlist alias (for the ban / allow gate).
_ID_TO_ALIAS = {
    "qwen3.6-35b-a3b": "qwen36-35b-a3b",
    "qwen3.6-27b": "qwen36-27b",
    "qwen3.8-27b": "qwen36-27b",
    "qwen3.8-flash-next": "qwen38-flash-next",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-flash-95b": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4-pro-preview": "deepseek-v4-pro",
    "deepseek-v4-1.5": "deepseek-v4-1.5",
    "dflash2-28b": "dflash2-28b",
    "dflash2-24b": "dflash2-24b",
    "dspark-550b": "dspark-550b",
}


def _alias_for(model_id: str) -> str:
    if model_id in _ID_TO_ALIAS:
        return _ID_TO_ALIAS[model_id]
    # Fallback: normalize (lowercase, drop dots/spaces) so the ban prefixes
    # (e.g. ``qwen35``) still catch new catalog entries.
    return model_id.lower().replace(".", "").replace(" ", "")


# --- dataclasses -----------------------------------------------------------
@dataclass
class ModelFit:
    model: str
    quality: int
    variant: str | None
    quant: str | None
    context_window: int | None
    zero_spill: bool
    tok_per_sec: float
    required_gb: float
    allowed: bool
    banned: bool
    alias: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "quality": self.quality,
            "variant": self.variant,
            "quant": self.quant,
            "context_window": self.context_window,
            "zero_spill": self.zero_spill,
            "tok_per_sec": round(self.tok_per_sec, 1),
            "required_gb": round(self.required_gb, 1),
            "allowed": self.allowed,
            "banned": self.banned,
            "alias": self.alias,
        }


@dataclass
class Recommendation:
    source: str  # "hermes-builtin" | "turbofit-fallback"
    uma: bool
    pool_kind: str  # "unified" | "dedicated" | "cpu"
    backend: str  # "metal" | "cuda" | "rocm" | "vulkan" | "auto"
    ladder: str  # "unified/moe" | "dedicated/dense"
    vram_gb: float
    host_ram_gb: float
    usable_vram_gb: float
    compensated: bool
    primary: ModelFit | None
    speed: ModelFit | None
    stretch: dict[str, Any] | None
    fits: list[ModelFit] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    engine: str | None = None
    turbofit_canonical: dict[str, Any] | None = None
    suggested_profile: str | None = None
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "uma": self.uma,
            "pool_kind": self.pool_kind,
            "backend": self.backend,
            "ladder": self.ladder,
            "vram_gb": round(self.vram_gb, 1),
            "host_ram_gb": round(self.host_ram_gb, 1),
            "usable_vram_gb": round(self.usable_vram_gb, 1),
            "uma_compensated": self.compensated,
            "primary": self.primary.to_dict() if self.primary else None,
            "speed": self.speed.to_dict() if self.speed else None,
            "stretch": self.stretch,
            "fits": [m.to_dict() for m in self.fits],
            "refused": self.refused,
            "engine": self.engine,
            "turbofit_canonical": self.turbofit_canonical,
            "suggested_profile": self.suggested_profile,
            "why": self.why,
        }


# --- built-in access -------------------------------------------------------
def _ensure_hermes_importable() -> None:
    """Ensure the built-in (``hermes_cli``) is importable.

    In production the running agent *is* ``hermes_cli``, so it is already on
    ``sys.path``. When running from the standalone repo/venv it is not, so we
    add the Hermes Agent checkout (``TURBOFIT_HERMES_HOME``, default
    ``~/.hermes/hermes-agent``) to the path. No-op when it is already
    importable.
    """
    import importlib.util

    if importlib.util.find_spec(_HERMES_PKG) is not None:
        return
    home = os.path.expanduser(
        os.environ.get(_HERMES_HOME_ENV, _DEFAULT_HERMES_HOME)
    )
    if home and os.path.isdir(home) and home not in sys.path:
        sys.path.insert(0, home)


def available() -> bool:
    """True when the built-in Hermes local runtime can be imported."""
    try:
        import importlib

        _ensure_hermes_importable()
        importlib.import_module(_HERMES_PKG + ".local_runtime.catalog")
        return True
    except Exception:
        return False


def _import_hermes():
    import importlib

    _ensure_hermes_importable()
    return (
        importlib.import_module(_HERMES_PKG + ".local_runtime.hardware"),
        importlib.import_module(_HERMES_PKG + ".local_runtime.catalog"),
        importlib.import_module(_HERMES_PKG + ".local_runtime.estimator"),
    )


def uma_compensate(budget: Any, headroom: float = UMA_HEADROOM) -> tuple[Any, bool]:
    """Credit the full unified pool on UMA hardware.

    The built-in ``_uma_budget`` returns ``ram_available_bytes=0`` and a low
    ``usable_vram_bytes`` on Apple Silicon, which physics-refuses models that
    in fact fit in the shared unified pool. We raise ``usable_vram_bytes`` to
    the unified pool minus headroom and hand the remainder to host RAM, so the
    built-in physics check credits the real 32 GB (or N GB) pool.
    """
    if not getattr(budget, "uma", False):
        return budget, False
    unified = int(getattr(budget, "total_device_bytes", 0))
    if unified <= 0:
        return budget, False
    usable = int(unified * (1.0 - headroom))
    current_usable = int(getattr(budget, "usable_vram_bytes", 0))
    current_host = int(getattr(budget, "ram_available_bytes", 0))
    extra_host = max(0, unified - current_usable - current_host)
    adjusted = replace(
        budget,
        usable_vram_bytes=max(usable, current_usable),
        ram_available_bytes=current_host + extra_host,
    )
    return adjusted, True


def _backend_from_budget(budget: Any) -> str:
    platform = getattr(budget, "platform", "")
    gpu_name = (getattr(budget, "gpu_name", "") or "").lower()
    gpu_pci_id = getattr(budget, "gpu_pci_id", None)
    if platform == "darwin":
        return "metal"
    if gpu_pci_id is not None or any(k in gpu_name for k in ("nvidia", "rtx", "gtx", "a100", "h100")):
        return "cuda"
    if "rad" in gpu_name or "rx" in gpu_name or platform == "linux":
        return "auto"
    return "auto"


# --- scan ------------------------------------------------------------------
def scan(backend: str = "auto") -> dict[str, Any]:
    """Run the built-in scan: probe, compensate UMA, physics-check the catalog.

    Returns a plain dict (see ``recommend`` for the overlaid form). Raises
    ``RuntimeError`` when the built-in runtime is unavailable.
    """
    hardware, catalog, estimator = _import_hermes()
    budget = hardware.probe_budget()
    adjusted, compensated = uma_compensate(budget)

    fits: list[ModelFit] = []
    refused: list[dict[str, Any]] = []
    entries = getattr(catalog, "CATALOG", None)
    if not entries:
        _by_id = getattr(catalog, "catalog_by_id", None)
        entries = list(_by_id().values()) if callable(_by_id) else []
    for entry in entries:
        selected = catalog.select_variant(entry, adjusted)
        alias = _alias_for(entry.id)
        banned = allowed_lineup.is_banned(alias)
        allowed = allowed_lineup.is_allowed_main(alias)
        if selected is None:
            # Minimum bytes across quantized variants = the floor it would need.
            floor = min((v.size_bytes for v in entry.variants), default=0)
            refused.append(
                {
                    "model": entry.id,
                    "quality": entry.quality,
                    "required_gb": round(floor / _GIB, 1),
                    "allowed": allowed,
                    "banned": banned,
                }
            )
            continue
        try:
            tps = float(
                catalog.predicted_decode_tok_s(
                    entry,
                    selected.variant,
                    adjusted,
                    spilled=not selected.zero_spill,
                    backend=backend,
                )
            )
        except Exception:
            tps = 0.0
        fits.append(
            ModelFit(
                model=entry.id,
                quality=entry.quality,
                variant=selected.variant.model_id,
                quant=selected.variant.model_id,
                context_window=getattr(entry, "n_ctx_train", None),
                zero_spill=selected.zero_spill,
                tok_per_sec=tps,
                required_gb=round(selected.variant.size_bytes / _GIB, 1),
                allowed=allowed,
                banned=banned,
                alias=alias,
            )
        )

    # Built-in preference order: highest quality, then fastest decode.
    fits.sort(key=lambda m: (m.quality, m.tok_per_sec), reverse=True)
    refused.sort(key=lambda r: (r["quality"], r["required_gb"]), reverse=True)

    built_in_pick: ModelFit | None = None
    # Use the built-in's own pick when it produced one (already UMA-compensated).
    pick_pair = catalog.recommended_entry(adjusted, backend=backend)
    if pick_pair is not None:
        pick_entry, _reason = pick_pair
        for fit in fits:
            if fit.model == pick_entry.id:
                built_in_pick = fit
                break
        if built_in_pick is None and fits:
            built_in_pick = fits[0]

    vram_gb = int(getattr(budget, "total_device_bytes", 0)) / _GIB
    host_ram_gb = int(getattr(budget, "ram_available_bytes", 0)) / _GIB
    if getattr(budget, "uma", False):
        # Unified memory: the "host RAM" is the same pool the GPU draws from.
        host_ram_gb = max(host_ram_gb, int(getattr(budget, "total_device_bytes", 0)) / _GIB)
        pool_kind = "unified"
        backend = _backend_from_budget(budget)
    else:
        pool_kind = "dedicated" if vram_gb > 0 else "cpu"
        backend = _backend_from_budget(budget)

    return {
        "budget": adjusted,
        "raw_budget": budget,
        "uma": bool(getattr(budget, "uma", False)),
        "compensated": compensated,
        "pool_kind": pool_kind,
        "backend": backend,
        "vram_gb": vram_gb,
        "host_ram_gb": host_ram_gb,
        "usable_vram_gb": int(getattr(adjusted, "usable_vram_bytes", 0)) / _GIB,
        "fits": fits,
        "refused": refused,
        "built_in_pick": built_in_pick,
    }


# --- overlay: TurboFit doctrine --------------------------------------------
def _turbofit_canonical(vram_gb: float, host_ram_gb: float, pool_kind: str, backend: str):
    vendor = "apple" if pool_kind == "unified" else ("nvidia" if backend == "cuda" else "other")
    try:
        options = allowed_lineup.check_local_options(
            vram_gb=vram_gb,
            host_ram_gb=host_ram_gb,
            memory_pool=pool_kind,
            backend=backend,
            vendor=vendor,
        )
    except Exception:
        return None
    if not options:
        return None
    top = options[0]

    def _s(value: object) -> str | None:
        return str(value) if value is not None else None

    return {
        "main": _s(top.get("alias")),
        "role": _s(top.get("role")),
        "engine": _s(top.get("engine")),
        "backend": _s(top.get("backend")),
        "residency": _s(top.get("residency")),
        "why": _s(top.get("why")),
        "options": [
            {"alias": _s(o.get("alias")), "role": _s(o.get("role")), "engine": _s(o.get("engine"))}
            for o in options[:6]
        ],
    }


def _stretch_target(fits, refused) -> dict[str, Any] | None:
    for fit in fits:
        if fit.model == FLASH_NEXT_ID:
            return {
                "model": FLASH_NEXT_ID,
                "quality": fit.quality,
                "fits": True,
                "required_gb": fit.required_gb,
                "zero_spill": fit.zero_spill,
                "tok_per_sec": round(fit.tok_per_sec, 1),
                "note": "Flash Next fits on this hardware.",
            }
    for ref in refused:
        if ref["model"] == FLASH_NEXT_ID:
            return {
                "model": FLASH_NEXT_ID,
                "quality": ref["quality"],
                "fits": False,
                "required_gb": ref["required_gb"],
                "note": (
                    f"Flash Next (q={ref['quality']}) is the next rung up: it needs "
                    f"~{ref['required_gb']} GB resident, which exceeds this box. "
                    f"Convergence target when a bigger box / quant lands."
                ),
            }
    return None


def recommend(backend: str = "auto") -> Recommendation:
    """Built-in Hermes recommendation + TurboFit doctrine overlay.

    Falls back to TurboFit's own ladders when the built-in runtime is
    unavailable, so the command is safe on a host without Hermes.
    """
    # --- Built-in path -----------------------------------------------------
    if available():
        try:
            data = scan(backend=backend)
        except Exception as exc:  # defensive: treat as unavailable
            data = None
        if data is not None:
            fits = data["fits"]
            refused = data["refused"]

            # Primary = the built-in's own pick (ban-filtered); if banned or
            # absent, fall back to the highest-quality fit.
            primary = data.get("built_in_pick")
            if primary is None or primary.banned:
                primary = next((m for m in fits if not m.banned), None)
            speed = max(
                (m for m in fits if not m.banned),
                key=lambda m: m.tok_per_sec,
                default=None,
            )
            canonical = _turbofit_canonical(
                data["vram_gb"], data["host_ram_gb"], data["pool_kind"], data["backend"]
            )
            if primary is None and canonical and canonical.get("main"):
                # Built-in found nothing allowed (all banned) -> use TurboFit's own pick.
                primary = ModelFit(
                    model=canonical["main"],
                    quality=0,
                    variant=None,
                    quant=None,
                    context_window=None,
                    zero_spill=False,
                    tok_per_sec=0.0,
                    required_gb=0.0,
                    allowed=True,
                    banned=False,
                    alias=canonical["main"],
                )
            ladder = "unified/moe" if data["uma"] else "dedicated/dense"
            why = (
                "Built-in Hermes local-runtime scan "
                f"({'UMA-compensated ' if data['compensated'] else ''}"
                f"pool={data['pool_kind']}, {data['usable_vram_gb']:.1f} GiB usable)"
                + (
                    f" -> primary {primary.model} (allowed, not banned)"
                    if primary
                    else " -> no built-in model fits; using TurboFit fallback"
                )
            )
            rec = Recommendation(
                source="hermes-builtin",
                uma=data["uma"],
                pool_kind=data["pool_kind"],
                backend=data["backend"],
                ladder=ladder,
                vram_gb=data["vram_gb"],
                host_ram_gb=data["host_ram_gb"],
                usable_vram_gb=data["usable_vram_gb"],
                compensated=data["compensated"],
                primary=primary,
                speed=speed,
                stretch=_stretch_target(fits, refused),
                fits=[m for m in fits if not m.banned],
                refused=refused,
                engine=(canonical or {}).get("engine"),
                turbofit_canonical=canonical,
                suggested_profile=None,
                why=why,
            )
            rec.suggested_profile = _suggested_profile(data)
            return rec

    # --- Fallback: TurboFit's own ladders + check_local_options ------------
    # Without the built-in runtime we can't physics-scan the catalog; degrade
    # to the allowlist + ladders so the command still yields a recommendation.
    # (Requires a hardware fingerprint; caller may supply one.)
    raise RuntimeError(
        "built-in Hermes local runtime unavailable and no TurboFit hardware "
        "fingerprint supplied; cannot recommend. Run on a Hermes host or pass "
        "a fingerprint."
    )


def _suggested_profile(data: dict[str, Any]) -> str | None:
    """Map the scan to TurboFit's canonical ``hardware-<N>gb`` profile id."""
    total_gb = max(data["usable_vram_gb"], data["host_ram_gb"])
    if total_gb <= 0:
        return None
    # Class buckets the profile catalog uses (16/24/32/48/64/80/144/256...).
    buckets = [8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 192, 256]
    chosen = 144
    for bucket in buckets:
        if total_gb >= bucket:
            chosen = bucket
    return f"hardware-{chosen}gb"


# --- self-contained entry for the CLI --------------------------------------
def build_recommendation(fingerprint: Any = None, backend: str = "auto") -> Recommendation:
    """Primary entry point: built-in-backed recommendation, fingerprint-optional."""
    if available():
        return recommend(backend=backend)
    # Fallback with a supplied TurboFit fingerprint (standalone hosts).
    if fingerprint is not None:
        vram_gb = getattr(fingerprint, "total_usable_memory_mb", 0) / 1024
        host_ram_gb = getattr(fingerprint, "system_ram_mb", 0) / 1024
        pool_kind = getattr(fingerprint, "memory_pool_kind", "cpu")
        backends = getattr(fingerprint, "backends", {"auto"})
        backend = sorted(backends)[0] if backends else "auto"
        vendor = (
            "apple"
            if pool_kind == "unified"
            else ("nvidia" if "cuda" in backends else "other")
        )
        options = allowed_lineup.check_local_options(
            vram_gb=vram_gb, host_ram_gb=host_ram_gb, memory_pool=pool_kind,
            backend=backend, vendor=vendor,
        )
        primary = None
        engine = None
        canonical = None
        if options:
            top = options[0]
            alias = str(top.get("alias"))
            engine = str(top.get("engine")) if top.get("engine") else None
            primary = ModelFit(
                model=alias, quality=0, variant=None, quant=None,
                context_window=None, zero_spill=False, tok_per_sec=0.0,
                required_gb=0.0, allowed=True, banned=False, alias=alias,
            )
            canonical = {
                "main": alias,
                "engine": engine,
                "why": str(top.get("why")) if top.get("why") else None,
            }
        return Recommendation(
            source="turbofit-fallback",
            uma=pool_kind == "unified",
            pool_kind=pool_kind,
            backend=backend,
            ladder="unified/moe" if pool_kind == "unified" else "dedicated/dense",
            vram_gb=vram_gb, host_ram_gb=host_ram_gb, usable_vram_gb=vram_gb,
            compensated=False, primary=primary, speed=None,
            stretch=_stretch_target([], []),
            fits=[] if primary is None else [primary], refused=[],
            engine=engine,
            turbofit_canonical=canonical,
            suggested_profile=None,
            why="TurboFit fallback (built-in Hermes runtime unavailable): "
                "allowlist + ladders only, no built-in physics scan.",
        )
    raise RuntimeError(
        "built-in Hermes local runtime unavailable and no hardware fingerprint "
        "supplied; cannot recommend."
    )
