"""Physics-predicted TurboFit List.

Distinct from the evidence-only List (:mod:`turbofit_runtime.turbofit_list`),
which only shows benchmark winners with physical evidence. This module
produces a *predicted* companion: for each hardware tier it runs the
built-in Hermes catalog's decode physics on a synthetic Apple-unified budget
of that capacity and reports the physics prediction.

Every row is explicitly ``status="predicted"`` — never measured, never
evidence. Measured winners (the evidence-only List) replace these values as
the physical/intelligence campaigns complete. This is the non-blocking bridge
the evidence-only List waits for, so the List is no longer blank while
benchmarks are still in flight.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Mapping

_GIB = 1024 ** 3
TIERS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)


def _tier_budget(base: Any, capacity_gb: int) -> Any:
    """A synthetic Apple-unified budget with the given total capacity.

    Starts from the host's real budget (so backend/vendor fields are valid)
    and overrides the memory pool to a unified pool of exactly ``capacity_gb``.
    ``uma_compensate`` then credits the full pool, mirroring real Apple Silicon.
    """
    return dataclasses.replace(
        base,
        uma=True,
        total_device_bytes=capacity_gb * _GIB,
        usable_vram_bytes=0,
        ram_available_bytes=0,
        platform="darwin",
        gpu_name="apple silicon",
        gpu_pci_id=None,
    )


def build_predicted_turbofit_list(backend: str = "auto") -> dict[str, Any]:
    """Compute a physics-predicted row for every tier.

    Uses the built-in Hermes catalog's decode physics (the same functions the
    built-in scan uses). Raises ``RuntimeError`` when the built-in runtime is
    unavailable.
    """
    from turbofit_runtime import hermes_fit

    if not hermes_fit.available():
        raise RuntimeError("built-in Hermes local runtime unavailable")
    hardware, catalog, _est = hermes_fit._import_hermes()
    base = hardware.probe_budget()
    levels: list[dict[str, Any]] = []
    for capacity in TIERS:
        budget = _tier_budget(base, capacity)
        adjusted, _comp = hermes_fit.uma_compensate(budget)
        backend_eff = hermes_fit._backend_from_budget(adjusted)
        row: dict[str, Any] = {
            "hardware_level_gb": capacity,
            "status": "predicted",
            "engine": backend_eff,
            "main": None,
            "quant": None,
            "context": None,
            "intelligence": None,
            "predicted_tps": None,
            "required_gb": None,
            "zero_spill": None,
        }
        pick = catalog.recommended_entry(adjusted, backend=backend_eff)
        if pick is not None:
            entry, _reason = pick
            selected = catalog.select_variant(entry, adjusted)
            if selected is not None:
                try:
                    tps = float(catalog.predicted_decode_tok_s(
                        entry, selected.variant, adjusted,
                        spilled=not selected.zero_spill, backend=backend_eff,
                    ))
                except Exception:
                    tps = 0.0
                row.update({
                    "main": entry.id,
                    "quant": selected.variant.model_id,
                    "context": getattr(entry, "n_ctx_train", None),
                    "intelligence": entry.quality,
                    "predicted_tps": round(tps, 1),
                    "required_gb": round(selected.variant.size_bytes / _GIB, 1),
                    "zero_spill": bool(selected.zero_spill),
                })
        levels.append(row)
    return {
        "schema": "turbofit.list.predicted/v1",
        "name": "TurboFit List (physics-predicted)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": "built-in Hermes catalog decode physics on a synthetic Apple-unified budget; predicted, not measured",
        "levels": levels,
    }


def render_predicted_turbofit_list(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TurboFit List (physics-predicted)",
        "",
        "Computed from the built-in Hermes catalog decode physics, **not** from measured benchmarks. Every row is a **prediction**, not evidence. Measured winners (the evidence-only List) replace these as the physical and intelligence campaigns complete.",
        "",
        "| Hardware level | Main model | Config | Context | Intelligence | Predicted TPS | Zero-spill |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for item in payload.get("levels", []):
        if not item.get("main"):
            lines.append(
                f"| {item['hardware_level_gb']} GB | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {item['hardware_level_gb']} GB | `{item['main']}` | {item['quant']} | "
            f"{item['context']} | {item['intelligence']} | {item['predicted_tps']} | "
            f"{'yes' if item['zero_spill'] else 'no'} |"
        )
    lines.extend([
        "",
        "All values above are physics predictions from the built-in catalog. They are not measurements and are not evidence.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_predicted_turbofit_list(build_predicted_turbofit_list()))
