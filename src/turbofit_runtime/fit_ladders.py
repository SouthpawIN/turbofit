"""Canonical two-list Fit List publisher.

Joins the live-refit fit ladders (fit_check) with the engine registry
(references/engine-serve-matrix.json) so each rung carries its *per-model
engine priority* — the "prioritize that by the models" ordering.

Two separate lists, kept clean by memory model:
  - dedicated_ladder(): GPU/VRAM — dense is fine on a real GPU.
  - unified_ladder():   CPU/RAM  — MoE stand-ins only (dense crawls on CPU).
Both converge on Qwen 3.8 Flash Next at 64+ GB.

This is the single source the list endpoint and the wiki should render.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

try:  # imported as part of the package (tests, services)
    from .fit_check import DEDICATED_TIERS, SHARED_TIERS
except ImportError:  # run directly: python fit_ladders.py
    from fit_check import DEDICATED_TIERS, SHARED_TIERS

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MATRIX = os.path.join(_REPO, "references", "engine-serve-matrix.json")

# First model/engine token -> (display name, class)
_META = {
    "bonsai-2":   ("Bonsai 2", "dense"),
    "underdog":   ("Underdog 27B", "dense"),
    "gsq-rco":    ("Qwen 3.8 GSQ-RCO 27B", "dense"),
    "maple":      ("Maple Preview", "moe"),
    "cyber-tiel": ("Cyber-Tiel Coder", "moe"),
    "strata":     ("Qwen 3.8 Flash Next", "moe"),
}

# MoE stand-in correspondence for the Unified list (first token -> stands in for)
_STANDS_IN_FOR = {
    "maple": "Bonsai 2 / Underdog (dense low rungs)",
    "cyber-tiel": "GSQ-RCO 27B (dense mid rung)",
    "strata": "both lists converge here",
}

_PAIRS: dict = {}
_ENGINES: dict = {}


def _load_pairs() -> dict:
    if not _PAIRS:
        with open(_MATRIX, encoding="utf-8") as fh:
            _PAIRS.update(json.load(fh).get("pairs", {}))
    return _PAIRS


def _load_engines() -> dict:
    if not _ENGINES:
        with open(_MATRIX, encoding="utf-8") as fh:
            _ENGINES.update(json.load(fh).get("engines", {}))
    return _ENGINES


def _pair_key_for(model_tokens) -> str | None:
    """Map a rung's model tokens to its engine-serve pair key (model-named)."""
    first = model_tokens[0]
    rest = list(model_tokens[1:])
    if first == "maple":
        return "maple-preview-tq2"
    if first == "underdog":
        return "underdog-27b-pq2"
    if first == "strata":
        return "qwen3-8-flash-next-strata"
    if first == "bonsai-2":
        return "ternary-bonsai-2-27b-" + ("ptq1" if "ptq1" in rest else "pq2")
    if first == "gsq-rco":
        return "qwen3-8-27b-gsq-rco-iq3" + ("xxs" if "xxs" in rest else "s")
    if first == "cyber-tiel":
        # all four quant rungs share the same model + serving path
        return "cyber-tiel-coder-35b-a3b-ud-q4-k-xl"
    return None


def _quant_label(model_tokens) -> str:
    first = model_tokens[0]
    if first == "strata":
        return "IQ2_XS"
    rest = [t for t in model_tokens[1:] if t not in ("flash", "next")]
    return "_".join(rest) if rest else "—"


def _released(engine_id: str) -> bool:
    return _load_engines().get(engine_id, {}).get("status") == "released"


@dataclass
class Rung:
    band: str
    model: str
    quant: str
    class_: str          # "dense" | "moe"
    engines: list = field(default_factory=list)   # ordered per-model priority
    low_vram_candidates: list = field(default_factory=list)  # candidate engines (qualification gate pending)
    apple_alternate: list = field(default_factory=list)
    status: str = "candidate"
    note: str = ""

    def engine_line(self) -> str:
        if not self.engines:
            return "—"
        return " > ".join(
            f"{e} ({'released' if _released(e) else 'candidate'})" for e in self.engines
        )

    def apple_line(self) -> str:
        return ", ".join(self.apple_alternate) if self.apple_alternate else "—"


def _rung(tier) -> Rung:
    first = tier.model_tokens[0]
    display, cls = _META.get(first, (tier.model_tokens[0], "moe"))
    key = _pair_key_for(tier.model_tokens)
    rec = _load_pairs().get(key, {}) if key else {}
    engines = list(rec.get("priority") or rec.get("eligible_engines") or [])
    low_vram_candidates = list(rec.get("low_vram_dense_candidates") or [])
    status = "released" if any(_released(e) for e in engines) else "candidate"
    _alt = rec.get("apple_alternate")
    apple_alternate = [_alt] if isinstance(_alt, str) else list(_alt or [])
    band = f"{tier.min_gb}–{tier.max_gb}" if tier.max_gb else f"{tier.min_gb}+"
    return Rung(
        band=band,
        model=display,
        quant=_quant_label(tier.model_tokens),
        class_=cls,
        engines=engines,
        low_vram_candidates=low_vram_candidates,
        apple_alternate=apple_alternate,
        status=status,
        note=_STANDS_IN_FOR.get(first, ""),
    )


def dedicated_ladder() -> list:
    return [_rung(t) for t in DEDICATED_TIERS]


def unified_ladder() -> list:
    return [_rung(t) for t in SHARED_TIERS]


def _table(rows, header) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def render_dedicated() -> str:
    header = ["Band (VRAM)", "Model", "Quant", "Engine priority (per-model)", "Status"]
    rows = [[r.band, r.model, r.quant, r.engine_line(), r.status] for r in dedicated_ladder()]
    return _table(rows, header)


def render_unified() -> str:
    header = ["Band (RAM)", "Model", "Quant", "Engine priority (per-model)", "Apple alt", "Stands in for", "Status"]
    rows = [[r.band, r.model, r.quant, r.engine_line(), r.apple_line(), r.note or "—", r.status]
            for r in unified_ladder()]
    return _table(rows, header)


def render() -> str:
    return (
        "# Fit Lists v1\n\n"
        "Two separate lists, kept clean by memory model. **Dense stays on the "
        "Dedicated list** (a real GPU can run dense); the **Unified/CPU list uses "
        "MoE stand-ins** because a dense 27B crawls on CPU. Both converge on Qwen "
        "3.8 Flash Next.\n\n"
        "## Dedicated (GPU / VRAM) — dense is fine on a real GPU\n\n"
        + render_dedicated()
        + "\n\n## Unified (CPU / RAM) — MoE stand-ins, never dense\n\n"
        + render_unified()
        + "\n\n> **Correspondence:** Maple is the MoE replacement for Bonsai 2 + "
        "Underdog (the low dense rungs); Cyber-Tiel is the MoE replacement for "
        "GSQ-RCO 27B (the mid dense rung). Dense never appears on the Unified "
        "list. On Apple silicon the 64 GB+ Flash Next rung is candidate-only "
        "(no released macOS strata build yet)."
    )


if __name__ == "__main__":
    print(render())
