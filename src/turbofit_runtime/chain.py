"""The fallback chain and the three dials a user actually needs to see.

TurboFit's behaviour is easy to describe and hard to watch: it sheds context, then model
class, then releases residency entirely, and heals back when the machine is quiet again.
This module turns that into a small, stable payload — memory pressure, the context window
(effective vs native), and the ordered chain with the options available at each level —
so every surface (status tool, dashboard, desktop, TUI) shows the same three dials instead
of inventing its own view of the ladder.

Nothing here probes hardware or spawns processes: callers pass in a usage snapshot. That
keeps the dials unit-testable and keeps the module free of policy authority.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHAIN_SCHEMA = "turbofit.fallback-chain/v1"
DEFAULT_CHAIN_PATH = Path(__file__).resolve().parents[2] / "references" / "fallback-chain.json"

_LEVEL_KINDS = frozenset({"model", "context", "yield"})
_EVIDENCE = frozenset({"measured", "portable-fit", "candidate"})
_OPTIONAL_LEVEL_KEYS = frozenset({"options", "evidence_source", "note", "quant"})


@dataclass(frozen=True)
class Level:
    id: str
    kind: str
    model: str | None
    engine: str | None
    context: int
    needs_mb_per_card: int
    evidence: str
    options: tuple[Mapping[str, Any], ...] = ()
    evidence_source: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "model": self.model,
            "engine": self.engine,
            "context": self.context,
            "needs_mb_per_card": self.needs_mb_per_card,
            "evidence": self.evidence,
        }
        if self.evidence_source:
            payload["evidence_source"] = self.evidence_source
        if self.note:
            payload["note"] = self.note
        if self.options:
            payload["options"] = [dict(option) for option in self.options]
        return payload


@dataclass(frozen=True)
class Chain:
    schema: str
    context_floor: int
    lanes: Mapping[str, Mapping[str, Any]]

    def levels(self, lane: str) -> tuple[Level, ...]:
        lane_data = self.lanes.get(lane) or {}
        return tuple(lane_data.get("levels") or ())

    def lane_ids(self) -> tuple[str, ...]:
        return tuple(self.lanes)

    def level(self, lane: str, level_id: str) -> Level | None:
        return next((item for item in self.levels(lane) if item.id == level_id), None)

    def default_lane(self, *, unified: bool | None) -> str:
        if unified and "unified" in self.lanes:
            return "unified"
        return "dedicated" if "dedicated" in self.lanes else self.lane_ids()[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _parse_level(raw: Any, lane: str, index: int) -> Level:
    where = f"lanes[{lane}].levels[{index}]"
    _require(isinstance(raw, Mapping), f"{where} must be an object")
    known = {
        "id", "kind", "model", "engine", "context", "needs_mb_per_card", "evidence",
        *_OPTIONAL_LEVEL_KEYS,
    }
    unknown = set(raw) - known
    _require(not unknown, f"{where} has unknown fields: {sorted(unknown)}")
    for required in ("id", "kind", "context", "needs_mb_per_card", "evidence"):
        _require(required in raw, f"{where} is missing {required!r}")
    kind = str(raw["kind"])
    _require(kind in _LEVEL_KINDS, f"{where}.kind must be one of {sorted(_LEVEL_KINDS)}")
    evidence = str(raw["evidence"])
    _require(evidence in _EVIDENCE, f"{where}.evidence must be one of {sorted(_EVIDENCE)}")
    options = raw.get("options") or []
    _require(isinstance(options, list), f"{where}.options must be a list")
    for option_index, option in enumerate(options):
        _require(isinstance(option, Mapping), f"{where}.options[{option_index}] must be an object")
        _require(bool(option.get("model")), f"{where}.options[{option_index}] needs a model")
        option_evidence = str(option.get("evidence") or "candidate")
        _require(
            option_evidence in _EVIDENCE,
            f"{where}.options[{option_index}].evidence must be one of {sorted(_EVIDENCE)}",
        )
    model = raw.get("model")
    if kind != "yield":
        _require(bool(isinstance(model, str) and model), f"{where}.model is required for kind {kind!r}")
    return Level(
        id=str(raw["id"]),
        kind=kind,
        model=str(model) if isinstance(model, str) else None,
        engine=str(raw["engine"]) if isinstance(raw.get("engine"), str) else None,
        context=_non_negative_int(raw["context"], f"{where}.context"),
        needs_mb_per_card=_non_negative_int(
            raw["needs_mb_per_card"], f"{where}.needs_mb_per_card"
        ),
        evidence=evidence,
        options=tuple(dict(option) for option in options),
        evidence_source=raw.get("evidence_source"),
        note=raw.get("note"),
    )


def load_chain(path: str | Path | None = None) -> Chain:
    """Load and validate the chain document. Raises ValueError on a structural fault."""
    source = Path(path) if path is not None else DEFAULT_CHAIN_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), "chain root must be an object")
    _require(raw.get("schema") == CHAIN_SCHEMA, f"chain schema must equal {CHAIN_SCHEMA!r}")
    lanes_raw = raw.get("lanes")
    _require(bool(isinstance(lanes_raw, Mapping) and lanes_raw), "lanes must be a non-empty object")
    lanes: dict[str, Mapping[str, Any]] = {}
    for lane, lane_raw in lanes_raw.items():
        _require(isinstance(lane_raw, Mapping), f"lane {lane!r} must be an object")
        levels_raw = lane_raw.get("levels")
        _require(
            isinstance(levels_raw, list) and levels_raw,
            f"lane {lane!r} must contain a non-empty levels list",
        )
        parsed = tuple(_parse_level(item, str(lane), index) for index, item in enumerate(levels_raw))
        ids = [item.id for item in parsed]
        _require(len(ids) == len(set(ids)), f"lane {lane!r} has duplicate level ids")
        lanes[str(lane)] = {
            "label": str(lane_raw.get("label") or lane),
            "levels": parsed,
        }
    floor = _non_negative_int(raw.get("context_floor", 32768), "context_floor")
    return Chain(schema=CHAIN_SCHEMA, context_floor=floor, lanes=lanes)


def context_steps(native_window: int, floor: int = 32768) -> list[int]:
    """Halve a native window down to the floor, ending on something >= floor.

    The chain sheds context before it swaps models, so these are the windows the
    controller may re-launch at when a rung cannot hold its full window.
    """
    _require(isinstance(native_window, int) and native_window > 0, "native_window must be positive")
    _require(isinstance(floor, int) and floor > 0, "floor must be positive")
    steps: list[int] = []
    window = native_window
    while window > floor:
        steps.append(window)
        window //= 2
    steps.append(max(floor, window))
    return steps


def _gpu_reading(usage: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Normalise GPU rows from several probe shapes into (rows, unified)."""
    host = usage.get("host") if isinstance(usage.get("host"), Mapping) else {}
    topology = str(host.get("topology_key") or "")
    gpus = usage.get("gpus") or []
    rows: list[dict[str, Any]] = []
    for item in gpus if isinstance(gpus, Iterable) else []:
        if not isinstance(item, Mapping):
            continue
        total = item.get("total_mb") or item.get("memory_total_mb")
        used = item.get("used_mb") or item.get("memory_used_mb")
        free = item.get("free_mb")
        if not isinstance(total, int) or total <= 0:
            continue
        used_mb = used if isinstance(used, int) else max(0, total - (free or 0))
        rows.append({
            "gpu": item.get("gpu", len(rows)),
            "total_mb": total,
            "used_mb": max(0, used_mb),
            "free_mb": max(0, total - max(0, used_mb)),
            "utilization_pct": item.get("utilization_pct"),
        })
    unified = "uma" in topology or "unified" in topology or not rows
    return rows, unified


def _pressure_state(ratio: float) -> str:
    if ratio >= 0.90:
        return "critical"
    if ratio >= 0.75:
        return "elevated"
    return "nominal"


def build_dials(
    *,
    chain: Chain,
    usage: Mapping[str, Any] | None = None,
    lane: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the three dials from a usage snapshot. Pure; never probes."""
    usage = usage or {}
    rows, unified = _gpu_reading(usage)
    if lane is None:
        lane = chain.default_lane(unified=unified)
    if lane not in chain.lanes:
        lane = chain.default_lane(unified=unified)

    host = usage.get("host") if isinstance(usage.get("host"), Mapping) else {}
    if rows:
        total_mb = sum(row["total_mb"] for row in rows)
        used_mb = sum(row["used_mb"] for row in rows)
        headroom_mb = min(row["free_mb"] for row in rows)
    else:
        total_mb = int(host.get("host_usable_memory_mb") or 0)
        used_mb = 0
        headroom_mb = total_mb
    # No measurement is not the same as no headroom. An unprobed host must read as
    # unknown and claim nothing, otherwise every dial says "nothing fits" and the
    # active level collapses to yield on a machine that is perfectly idle.
    measured = total_mb > 0
    ratio = (used_mb / total_mb) if measured else 0.0

    levels = chain.levels(lane)
    reachable: list[dict[str, Any]] = []
    active: str | None = None
    for level in levels:
        if not measured:
            fits = None
        elif level.kind == "yield":
            fits = True
        else:
            fits = headroom_mb >= level.needs_mb_per_card
        entry = level.to_dict()
        entry["fits_now"] = fits
        entry["is_active"] = False
        reachable.append(entry)
        if active is None and fits:
            active = level.id
    for entry in reachable:
        entry["is_active"] = entry["id"] == active

    active_level = chain.level(lane, active) if active else None
    native_window = int((context or {}).get("native_max") or 0) or (
        active_level.context if active_level else (levels[0].context if levels else 0)
    )
    context_block = {
        "current": (context or {}).get("current"),
        "limit": (context or {}).get("limit"),
        "native_max": native_window or (context or {}).get("native_max"),
        "floor": chain.context_floor,
        "steps": context_steps(native_window, chain.context_floor) if native_window else [],
    }
    return {
        "memory_pressure": {
            "state": _pressure_state(ratio) if measured else "unknown",
            "ratio": round(ratio, 4) if measured else None,
            "headroom_mb": headroom_mb if measured else None,
            "measured": measured,
            "total_mb": total_mb,
            "cards": rows,
            "host_ram_mb": host.get("system_ram_mb"),
        },
        "context": context_block,
        "chain": {
            "lane": lane,
            "label": (chain.lanes.get(lane) or {}).get("label", lane),
            "active_level": active,
            "levels": reachable,
        },
        "options": [
            dict(option)
            for option in ((active_level.options if active_level else ()) or ())
        ],
    }


def dials_for_selection(
    *,
    usage: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
    chain_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: pick the lane from the profile's memory class, then build."""
    chain = load_chain(chain_path)
    unified: bool | None = None
    if isinstance(selection, Mapping):
        profile = str(selection.get("profile") or selection.get("requested") or "")
        if profile.startswith("apple-") or "unified" in profile:
            unified = True
        elif profile.startswith("hardware-"):
            unified = False
    return build_dials(chain=chain, usage=usage, context=context, lane=None if unified is None else
                       chain.default_lane(unified=unified))
