"""Row-level memory gate for the catalog campaign.

The expert-offload planner refuses a model that fits neither VRAM nor VRAM+RAM,
and that refusal is correct: launching it would OOM. But it is a property of one
row, not an error condition for the campaign. Raising it out of recipe
resolution took down ``status``, ``dry-run`` and ``run`` for every row, so a
single oversized catalog entry made the provider unusable.

This module turns the planner's refusal into ``waiting-memory`` rows, quoting the
planner's own arithmetic. A row listed here is never selected for execution; the
launch paths still refuse it independently (``ResolvedRecipe.assert_launchable``
and ``CampaignBackend.start``), so a gate bug cannot turn into an OOM.

This is deliberately a different verdict from :mod:`turbofit_runtime.artifact_gate`:
``waiting-download`` means the weights are not here yet, ``waiting-memory`` means
they are here and this host cannot hold them. A user acts on those differently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

STATUS = "waiting-memory"


@dataclass(frozen=True)
class WaitingMemory:
    """A row whose measured weights cannot be held by this host."""

    row_id: str
    reason: str
    family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "status": STATUS,
            "reason": self.reason,
            "family": self.family,
        }


def memory_deferred_rows(
    configuration_index: Mapping[str, dict[str, Any]],
    recipes: Any,
) -> dict[str, WaitingMemory]:
    """Rows the planner measured as too large for this host, with its reason.

    Resolution failures are skipped: a broken recipe is not a memory verdict and
    is reported by the row's own resolution path instead.
    """
    result: dict[str, WaitingMemory] = {}
    for row_id, payload in configuration_index.items():
        try:
            recipe = recipes.resolve_catalog_configuration(dict(payload))
        except Exception:  # noqa: BLE001 - an unresolved recipe is not this gate's verdict
            continue
        infeasible = recipe.infeasible_components()
        if not infeasible:
            continue
        component = infeasible[0]
        result[row_id] = WaitingMemory(
            row_id=row_id,
            reason=component.offload_reason,
            family=component.family,
        )
    return result


__all__ = ["STATUS", "WaitingMemory", "memory_deferred_rows"]
