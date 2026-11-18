"""Machine-agnostic VRAM/RAM/SSD split optimiser for Strata.

Strata is not a "does it fit in VRAM" engine.  It serves a large MoE *pack* by
placing each byte of it in one of three tiers and paying that tier's bandwidth
on every read:

* **VRAM** -- the ``--native`` dense/attention file, the KV cache, and up to
  ``--expert-cache N`` routed-expert blobs.
* **RAM** -- ``--resident-cpu-experts`` / ``--resident-experts`` keeps the
  non-cached expert blobs in pinned system memory; ``--ple-io ram`` puts the
  per-layer embedding table there too.
* **SSD** -- ``--mmap-experts`` streams the rest straight off the pack file.

The engine negotiates all of this at runtime and refuses a request it cannot
honour (``only 5.34 GiB of VRAM is free ... Lower --expert-cache``), so the
useful question before a launch is not "will it run" but *which split is
fastest on this box*.  That is a small search: for every KV mode and every
expert-cache slot count that free VRAM admits, price the placement and rank it.

What this module is not
-----------------------
It is arithmetic over a model row and a machine budget.  It does not probe the
host (callers pass ``vram_bytes`` / ``ram_bytes`` / ``ssd_free_bytes`` from
whatever fingerprint they trust), it does not launch anything, and it does not
pretend the bandwidth constants below are measurements.  They are labelled
estimates in :data:`BANDWIDTH_GIB_S`; replace them when the repo has telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

GIB = 1024 ** 3
MIB = 1024 * 1024

#: Context floor and the context the operator prefers when the model allows it.
MIN_CONTEXT_TOKENS = 65_536
PREFERRED_CONTEXT_TOKENS = 262_144

#: Bytes of one routed-expert blob as the engine caches it.  Documented by
#: Strata: "expert blob is 1,510,400 bytes per slot when cached".  Used only to
#: bound ``--expert-cache`` and to derive a slot count for packs whose vendor
#: does not publish one; the engine applies its own bound at launch.
EXPERT_SLOT_BYTES = 1_510_400

#: Effective read bandwidth per tier, GiB/s.  **Estimates**, not measurements:
#: dual-channel DDR5 desktop RAM sits near 60-80 GiB/s, a Gen4 NVMe near
#: 5-7 GiB/s, and HBM on a modern accelerator is an order of magnitude above
#: RAM.  The absolute values do not matter for ranking (only their ratios do);
#: the ratios do, and they are what the cost model below is built on.  Replace
#: with measured numbers from ``references/`` when a run records them.
BANDWIDTH_GIB_S: dict[str, float] = {
    "vram": 1_500.0,   # estimate: HBM3-class accelerator, 1-2 TiB/s order
    "ram": 60.0,       # estimate: dual-channel DDR5 workstation
    "ssd": 6.0,        # estimate: PCIe 4.0 NVMe sequential/mmap random mix
}

#: KV cache bytes as a fraction of the fp16 baseline, per ``--kv`` mode.
#: int8 halves, q4_0 quarters, k8v4 keeps 8-bit keys and 4-bit values.
KV_MODE_SCALE: dict[str, float] = {
    "fp16": 1.0,
    "int8": 0.5,
    "q4_0": 0.25,
    "k8v4": 0.375,
}

#: Attention-throughput multiplier per KV mode.  Quantised KV saves memory but
#: costs a dequantise step on every attention read; fp16 stays the reference.
#: **Estimates** (single-digit percentages), not measurements.
KV_MODE_ATTENTION_COST: dict[str, float] = {
    "fp16": 1.0,
    "int8": 1.03,
    "q4_0": 1.12,
    "k8v4": 1.08,
}

#: fp16 KV bytes by context.  Mirrors ``tier_fit.KV_GIB`` (the repo's documented
#: "2-4 GB at 262K" upper bound, scaled linearly, 1 GiB at 64K).
KV_BASE_GIB: dict[int, float] = {
    65_536: 1.0,
    131_072: 2.0,
    262_144: 4.0,
}

#: Activation + compute-buffer headroom the engine allocates on the card, by
#: context.  **Estimates**; the operator's own engine refusals are what a real
#: number would come from ("only 5.34 GiB of VRAM is free").
ACTIVATION_HEADROOM_GIB: dict[int, float] = {
    65_536: 0.75,
    131_072: 1.5,
    262_144: 3.0,
}

#: Small fixed VRAM reserve so a placement never plans the card to zero.
VRAM_RESERVE_BYTES = 512 * MIB
#: Small fixed RAM reserve for the host process itself.
RAM_RESERVE_BYTES = 1024 * MIB

#: CPU capability names Strata cares about.  AVX2 is the floor the docs state;
#: AVX-512/AMX speed up the RAM- and SSD-resident expert paths (dequantise,
#: gather, memcpy) but never gate a placement.
STRATA_REQUIRED_FLAG = "avx2"
STRATA_ACCELERATOR_FLAGS = ("avx512f", "avx512_vnni", "amx_int8")

#: Multiplier on the non-VRAM tiers when the host has no AVX2: the engine still
#: runs (docs: "8 GB runs, slowly") but every off-card read is far more
#: expensive.  **Estimate**, used only to rank, never to refuse.
NO_AVX2_PENALTY = 3.0
#: Multiplier on the non-VRAM tiers when AVX-512-class dequantise is available.
ACCELERATED_GAIN = 0.85

#: Prefill chunk the planner asks for by default; the engine accepts ``auto``.
DEFAULT_PREFILL_CHUNK = "2048"
#: Speculative lookahead the planner asks for by default.
DEFAULT_SPEC_TOKENS = 4


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelRow:
    """One Strata pack, described in bytes rather than brand names.

    ``pack_bytes`` is the whole artifact on disk.  ``expert_bytes`` is the
    routed-expert share of it (the part ``--expert-cache`` / resident / mmap
    decide about).  ``dense_bytes`` is the part Strata loads from ``--native``
    and keeps on the card: attention, shared experts, norms.  ``embed_bytes`` is
    the per-layer embedding table, which ``--ple-io`` can put in RAM.
    """

    model_id: str
    family: str
    quant: str
    pack_bytes: int
    expert_bytes: int
    expert_layers: int
    dense_bytes: int
    embed_bytes: int = 0
    expert_slots: int = 0            # 0 -> derive from EXPERT_SLOT_BYTES
    native_context: int = PREFERRED_CONTEXT_TOKENS
    supported_contexts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.pack_bytes <= 0:
            raise ValueError("pack_bytes must be positive")
        if self.expert_bytes < 0 or self.dense_bytes < 0 or self.embed_bytes < 0:
            raise ValueError("expert/dense/embed bytes must not be negative")

    @property
    def non_expert_bytes(self) -> int:
        return max(0, self.pack_bytes - self.expert_bytes)

    @property
    def slot_bytes(self) -> int:
        """Bytes of one cached expert blob for *this* pack.

        Documented blob size when the pack's slot count is known; otherwise the
        documented constant, which is what the engine's cache is sized in.
        """
        if self.expert_slots > 0 and self.expert_bytes > 0:
            return max(1, self.expert_bytes // self.expert_slots)
        return EXPERT_SLOT_BYTES

    @property
    def total_slots(self) -> int:
        if self.expert_slots > 0:
            return self.expert_slots
        return max(1, self.expert_bytes // EXPERT_SLOT_BYTES)

    def contexts(self) -> tuple[int, ...]:
        """Contexts this model may be asked for, ascending."""
        allowed = tuple(self.supported_contexts) or (self.native_context,)
        return tuple(sorted({c for c in allowed if c >= MIN_CONTEXT_TOKENS}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "quant": self.quant,
            "pack_gib": round(self.pack_bytes / GIB, 2),
            "expert_gib": round(self.expert_bytes / GIB, 2),
            "dense_gib": round(self.dense_bytes / GIB, 2),
            "embed_gib": round(self.embed_bytes / GIB, 2),
            "expert_layers": self.expert_layers,
            "expert_slots": self.total_slots,
            "native_context": self.native_context,
            "supported_contexts": list(self.contexts()),
        }


@dataclass(frozen=True)
class MachineBudget:
    """What a host offers one Strata launch."""

    vram_bytes: int
    ram_bytes: int
    ssd_free_bytes: int | None = None     # None -> unbounded / not declared
    cpu_flags: tuple[str, ...] = ("avx2",)

    def __post_init__(self) -> None:
        if self.vram_bytes <= 0:
            raise ValueError("vram_bytes must be positive")
        if self.ram_bytes < 0:
            raise ValueError("ram_bytes must not be negative")

    @property
    def flags(self) -> frozenset[str]:
        return frozenset(f.strip().lower() for f in self.cpu_flags if f)

    @property
    def has_avx2(self) -> bool:
        return STRATA_REQUIRED_FLAG in self.flags

    @property
    def accelerated(self) -> bool:
        return any(f in self.flags for f in STRATA_ACCELERATOR_FLAGS)

    @property
    def tier_factor(self) -> float:
        """Cost multiplier on the RAM/SSD tiers from CPU capability alone."""
        if not self.has_avx2:
            return NO_AVX2_PENALTY
        return ACCELERATED_GAIN if self.accelerated else 1.0

    @property
    def usable_vram_bytes(self) -> int:
        return max(0, self.vram_bytes - VRAM_RESERVE_BYTES)

    @property
    def usable_ram_bytes(self) -> int:
        return max(0, self.ram_bytes - RAM_RESERVE_BYTES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vram_gib": round(self.vram_bytes / GIB, 2),
            "ram_gib": round(self.ram_bytes / GIB, 2),
            "ssd_free_gib": None if self.ssd_free_bytes is None
            else round(self.ssd_free_bytes / GIB, 2),
            "cpu_flags": sorted(self.flags),
            "tier_factor": self.tier_factor,
        }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Placement:
    """One feasible way to place a model on a machine, priced."""

    model_id: str
    family: str
    context: int
    kv_mode: str
    expert_cache_slots: int
    kv_resident_tokens: int
    ple_io: str
    resident_cpu_experts: bool
    mmap_experts: bool
    vram_bytes: int
    ram_bytes: int
    ssd_bytes: int
    vram_share_gib: dict[str, float]
    predicted_speed: float      # 1.0 == no faster than streaming every expert
    flags: tuple[str, ...]
    reason: str
    cost_seconds: float
    notes: tuple[str, ...] = ()

    @property
    def context_k(self) -> int:
        return self.context // 1024

    @property
    def fits_preferred_context(self) -> bool:
        return self.context >= PREFERRED_CONTEXT_TOKENS

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "context": self.context,
            "context_k": self.context_k,
            "kv_mode": self.kv_mode,
            "expert_cache_slots": self.expert_cache_slots,
            "kv_resident_tokens": self.kv_resident_tokens,
            "ple_io": self.ple_io,
            "vram_gib": round(self.vram_bytes / GIB, 2),
            "ram_gib": round(self.ram_bytes / GIB, 2),
            "ssd_gib": round(self.ssd_bytes / GIB, 2),
            "vram_share_gib": {k: round(v, 2) for k, v in self.vram_share_gib.items()},
            "predicted_speed": round(self.predicted_speed, 3),
            "flags": list(self.flags),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Refusal:
    """A model that no placement on this machine can serve at any context."""

    model_id: str
    family: str
    reason: str
    binding: tuple[str, ...] = ()
    best_context: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "reason": self.reason,
            "binding": list(self.binding),
            "best_context": self.best_context,
        }


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def _expert_tier_bytes(
    row: ModelRow, cache_bytes: int, ram_resident_bytes: int
) -> tuple[int, int, int]:
    """(vram, ram, ssd) expert bytes for one placement."""
    vram = min(cache_bytes, row.expert_bytes)
    rest = max(0, row.expert_bytes - vram)
    ram = min(ram_resident_bytes, rest)
    ssd = max(0, rest - ram)
    return vram, ram, ssd


def _expert_cost(row: ModelRow, tiers: tuple[int, int, int], tier_factor: float) -> float:
    """Relative time to service one token's expert reads.

    Every active expert read is served by the tier holding it.  A cache that
    holds a uniform fraction of blobs hits that fraction of the time, so the
    cost is the byte-share-weighted sum of per-tier transfer times -- the model
    a tiered cache actually has.  Result is in "seconds per GiB of expert
    traffic" and is only ever compared against another placement of the same
    row, so the constant cancels.
    """
    vram, ram, ssd = tiers
    if row.expert_bytes <= 0:
        return 0.0
    v = BANDWIDTH_GIB_S["vram"]
    r = BANDWIDTH_GIB_S["ram"] * tier_factor
    s = BANDWIDTH_GIB_S["ssd"] * tier_factor
    return (vram / v) + (ram / r) + (ssd / s)


def _streaming_only_cost(row: ModelRow, tier_factor: float) -> float:
    """Baseline: every expert byte streamed from SSD, fp16 attention.

    Predicted speeds are reported as ``baseline / cost``, so 1.0 means "no
    better than streaming the whole pack off disk" and higher is faster.
    """
    return (
        _expert_cost(row, (0, 0, row.expert_bytes), tier_factor)
        + _attention_cost(row, "fp16")
    )


def _attention_cost(row: ModelRow, kv_mode: str) -> float:
    """Per-token attention term: the dense weights, at the KV mode's cost.

    Additive, not multiplicative: a quantised KV cache makes attention reads
    dearer, it does not make expert reads dearer.  The dense bytes are what the
    attention pass touches every token, so they are the only bytes scaled here.
    """
    return (row.dense_bytes / BANDWIDTH_GIB_S["vram"]) * KV_MODE_ATTENTION_COST[kv_mode]


def _kv_bytes(context: int, kv_mode: str) -> int:
    base = KV_BASE_GIB.get(context)
    if base is None:
        # Linear in the documented curve rather than refusing an odd context.
        base = max(1.0, context / 65_536 * 1.0)
    return int(base * GIB * KV_MODE_SCALE[kv_mode])


def _headroom_bytes(context: int) -> int:
    if context in ACTIVATION_HEADROOM_GIB:
        return int(ACTIVATION_HEADROOM_GIB[context] * GIB)
    return int(max(0.75, context / 65_536 * 0.75) * GIB)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def _contexts_for(row: ModelRow, target_context: int) -> tuple[int, ...]:
    contexts = [c for c in row.contexts() if c <= target_context]
    if target_context in row.contexts():
        contexts.append(target_context)
    if not contexts:
        contexts = list(row.contexts())
    return tuple(sorted(set(contexts)))


def enumerate_placements(
    row: ModelRow,
    machine: MachineBudget,
    target_context: int = PREFERRED_CONTEXT_TOKENS,
    kv_modes: Sequence[str] = ("fp16", "int8", "q4_0", "k8v4"),
) -> tuple[tuple[Placement, ...], tuple[str, ...]]:
    """Every feasible placement of ``row`` on ``machine``, best first.

    Returns ``(placements, blocking_constraints)``; the second element is empty
    when at least one placement exists and lists what ran out otherwise.

    The search is deliberately exhaustive over the dimensions that matter and
    tiny (a few hundred candidates): context, KV mode, KV residency, expert
    cache size, PLE location and whether the uncached experts are pinned in RAM
    or memory-mapped from the pack.
    """
    usable_vram = machine.usable_vram_bytes
    usable_ram = machine.usable_ram_bytes
    tier_factor = machine.tier_factor
    baseline = _streaming_only_cost(row, tier_factor)
    found: list[Placement] = []
    blocking: list[str] = []

    for context in _contexts_for(row, target_context):
        headroom = _headroom_bytes(context)
        for kv_mode in kv_modes:
            if kv_mode not in KV_MODE_SCALE:
                raise ValueError(f"unknown kv mode {kv_mode!r}")
            kv_full = _kv_bytes(context, kv_mode)

            # KV stays on the card unless the dense part + headroom leave no
            # room; only then is part of it spilled to host memory with
            # --kv-resident, which keeps the context at the cost of attention
            # bandwidth.
            spill_options: tuple[int, ...] = (
                (0,)
                if row.dense_bytes + kv_full + _headroom_bytes(context)
                <= usable_vram
                else (0, kv_full)
            )
            for kv_resident_bytes in spill_options:
                kv_on_card = kv_full - kv_resident_bytes
                dense_on_card = row.dense_bytes
                # The per-layer embedding table either lives on the card
                # (--ple-io direct) or in host memory (--ple-io ram).  Both are
                # priced: RAM is cheaper per byte than VRAM, but keeping PLE
                # off-card frees the card for expert blobs.
                embed_on_card_options = (
                    [0, row.embed_bytes] if row.embed_bytes else [0]
                )

                for embed_on_card in embed_on_card_options:
                    embed_in_ram = row.embed_bytes - embed_on_card
                    fixed_vram = dense_on_card + embed_on_card + kv_on_card + headroom
                    if fixed_vram > usable_vram:
                        blocking.append(
                            f"VRAM: dense {row.dense_bytes / GIB:.1f} GiB + "
                            f"{kv_mode} KV {kv_on_card / GIB:.1f} GiB + "
                            f"headroom {headroom / GIB:.1f} GiB exceeds the "
                            f"{usable_vram / GIB:.1f} GiB usable"
                        )
                        continue
                    spare = usable_vram - fixed_vram
                    max_slots = min(
                        row.total_slots,
                        int(spare // max(1, row.slot_bytes)),
                    )

                    for slots in _slot_candidates(max_slots, row.total_slots):
                        cache_bytes = slots * row.slot_bytes
                        uncached = row.expert_bytes - cache_bytes
                        ram_for_experts = max(0, usable_ram - embed_in_ram
                                              - kv_resident_bytes)
                        resident = min(max(0, ram_for_experts), uncached)
                        streamed = uncached - resident
                        ram_used = embed_in_ram + kv_resident_bytes + resident
                        if ram_used > usable_ram:
                            blocking.append(
                                f"RAM: this placement needs "
                                f"{ram_used / GIB:.1f} GiB of host memory but only "
                                f"{usable_ram / GIB:.1f} GiB is usable"
                            )
                            continue
                        if (
                            machine.ssd_free_bytes is not None
                            and streamed > machine.ssd_free_bytes
                        ):
                            if streamed or resident:
                                blocking.append(
                                    f"SSD: {streamed / GIB:.1f} GiB would have to "
                                    f"stream but only "
                                    f"{machine.ssd_free_bytes / GIB:.1f} GiB of free "
                                    f"SSD is declared"
                                )
                            continue

                        tiers = (cache_bytes, resident, streamed)
                        cost = (
                            _expert_cost(row, tiers, tier_factor)
                            + _attention_cost(row, kv_mode)
                        )
                        speed = baseline / cost if cost > 0 else baseline

                        vram_used = fixed_vram + cache_bytes
                        ram_used = row.embed_bytes + kv_resident_bytes + resident
                        ple_io = "ram" if row.embed_bytes else "direct"
                        placement = Placement(
                            model_id=row.model_id,
                            family=row.family,
                            context=context,
                            kv_mode=kv_mode,
                            expert_cache_slots=slots,
                            kv_resident_tokens=_kv_resident_tokens(context, kv_resident_bytes, kv_full),
                            ple_io=ple_io,
                            resident_cpu_experts=resident > 0,
                            mmap_experts=streamed > 0,
                            vram_bytes=vram_used,
                            ram_bytes=ram_used,
                            ssd_bytes=streamed,
                            vram_share_gib={
                                "dense": row.dense_bytes / GIB,
                                "embed": embed_on_card / GIB,
                                "kv": kv_on_card / GIB,
                                "experts": cache_bytes / GIB,
                                "headroom": headroom / GIB,
                            },
                            predicted_speed=round(speed, 3),
                            flags=_engine_flags(
                                context=context,
                                slots=slots,
                                kv_mode=kv_mode,
                                kv_resident_tokens=_kv_resident_tokens(
                                    context, kv_resident_bytes, kv_full
                                ),
                                resident=resident > 0,
                                streamed=streamed > 0,
                                ple_io=ple_io,
                            ),
                            reason=_reason(row, context, kv_mode, slots, cache_bytes,
                                           resident, streamed, speed),
                            cost_seconds=round(cost, 3),
                        )
                        found.append(placement)

    found.sort(key=lambda p: (-p.context, -p.predicted_speed, -p.expert_cache_slots))
    # Deduplicate: identical flag sets reachable through different loop paths.
    deduped: list[Placement] = []
    seen: set[tuple[str, ...]] = set()
    for placement in found:
        if placement.flags in seen:
            continue
        seen.add(placement.flags)
        deduped.append(placement)
    if deduped:
        return tuple(deduped), ()
    return (), tuple(dict.fromkeys(blocking))


def _kv_resident_tokens(context: int, resident_bytes: int, full_bytes: int) -> int:
    if resident_bytes <= 0 or full_bytes <= 0:
        return 0
    return int(round(context * resident_bytes / full_bytes))


def _slot_candidates(max_slots: int, total_slots: int) -> tuple[int, ...]:
    """Slot counts worth pricing: 0, the max, and a log-ish ladder between.

    Pricing every integer to 30k would be pointless: cost is linear in slots, so
    the maximum cache is always the fastest placement for a given context and KV
    mode.  The ladder exists so the ranked list still offers smaller caches when
    a caller wants VRAM left over for another process.
    """
    if max_slots <= 0:
        return (0,)
    values = {0, max_slots, total_slots if total_slots <= max_slots else 0}
    for frac in (0.25, 0.5, 0.75):
        values.add(int(max_slots * frac))
    values.add(min(8, max_slots))
    values.add(min(64, max_slots))
    return tuple(sorted(v for v in values if 0 <= v <= max_slots))


def _engine_flags(
    *,
    context: int,
    slots: int,
    kv_mode: str,
    kv_resident_tokens: int,
    resident: bool,
    streamed: bool,
    ple_io: str,
) -> tuple[str, ...]:
    """The exact ``strata`` flags that realise a placement.

    Only flags the engine's own ``--help`` lists are emitted; nothing here
    invents an option.  Placement conflicts are avoided by construction: a
    resident tier and a streaming tier are not mutually exclusive in the engine
    (cached blobs in VRAM, the rest pinned in RAM, and only what RAM cannot hold
    memory-mapped), so both may appear together.
    """
    args = ["--max-context", str(int(context)),
            "--expert-cache", str(int(slots)), "--kv", kv_mode]
    if kv_resident_tokens > 0:
        args.extend(["--kv-resident", str(int(kv_resident_tokens))])
    if resident:
        args.append("--resident-cpu-experts")
    if streamed:
        args.append("--mmap-experts")
    args.extend(["--ple-io", ple_io])
    args.extend(["--prefill", DEFAULT_PREFILL_CHUNK,
                 "--spec", str(DEFAULT_SPEC_TOKENS)])
    return tuple(args)


def _reason(
    row: ModelRow,
    context: int,
    kv_mode: str,
    slots: int,
    cache_bytes: int,
    resident: int,
    streamed: int,
    speed: float,
) -> str:
    parts = [
        f"{cache_bytes / GIB:.1f} of {row.expert_bytes / GIB:.1f} GiB of experts "
        f"cached in VRAM ({slots} slots)"
    ]
    if resident:
        parts.append(f"{resident / GIB:.1f} GiB pinned in RAM")
    if streamed:
        parts.append(f"{streamed / GIB:.1f} GiB streamed from SSD")
    parts.append(f"{kv_mode} KV at {context // 1024}K")
    return "; ".join(parts) + f" -> {speed:.2f}x streaming baseline"


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
def best_placement(
    row: ModelRow,
    machine: MachineBudget,
    target_context: int = PREFERRED_CONTEXT_TOKENS,
) -> Placement | Refusal:
    """The fastest placement that meets the target context, else the largest held."""
    placements, blocking = enumerate_placements(row, machine, target_context)
    if placements:
        at_target = [p for p in placements if p.context >= target_context]
        return (at_target or list(placements))[0]
    return Refusal(
        model_id=row.model_id,
        family=row.family,
        reason=(
            f"no placement at any supported context on "
            f"{machine.vram_bytes / GIB:.0f} GB VRAM / "
            f"{machine.ram_bytes / GIB:.0f} GiB RAM -- "
            + ("; ".join(blocking[:3]) if blocking else "no candidate fits")
        ),
        binding=tuple(_binding_names(blocking)),
        best_context=0,
    )


def _binding_names(blocking: Iterable[str]) -> tuple[str, ...]:
    names = []
    for item in blocking:
        head = item.split(":", 1)[0].strip().upper()
        if head and head not in names:
            names.append(head)
    return tuple(names)


def recommend_for_machine(
    vram_bytes: int,
    ram_bytes: int,
    ssd_free_bytes: int | None = None,
    cpu_flags: Sequence[str] = ("avx2",),
    *,
    rows: Sequence[ModelRow] | None = None,
    target_context: int = PREFERRED_CONTEXT_TOKENS,
) -> dict[str, Placement | Refusal]:
    """Best placement per model family for one machine.

    Returns a mapping keyed by ``ModelRow.family`` (falls back to the model id
    when two rows share a family and only the id is unambiguous).
    """
    machine = MachineBudget(
        vram_bytes=vram_bytes,
        ram_bytes=ram_bytes,
        ssd_free_bytes=ssd_free_bytes,
        cpu_flags=tuple(cpu_flags),
    )
    chosen: dict[str, Placement | Refusal] = {}
    for row in rows if rows is not None else family_rows():
        key = row.family
        if key in chosen:
            key = row.model_id
        chosen[key] = best_placement(row, machine, target_context)
    return chosen


# ---------------------------------------------------------------------------
# The families Strata serves, in bytes
# ---------------------------------------------------------------------------
#: ``embed_bytes`` is the per-layer embedding table the engine can place with
#: ``--ple-io``.  Qwen 3.8's 27.5 GiB figure is documented (Strata's split-engine
#: note: "about 43 GiB of RAM is free, 27.5 GiB of it the embedding table"); the
#: other families carry the share of non-expert bytes that is not attention,
#: labelled estimates so they are never mistaken for measurements.
FAMILY_ROWS: tuple[ModelRow, ...] = (
    ModelRow(
        model_id="qwen3-8-flash-next-strata",
        family="Qwen 3.8 Flash Next",
        quant="IQ2_XS",
        pack_bytes=68_012_000_000,
        expert_bytes=35_456_000_000,
        expert_layers=48,
        dense_bytes=5_000_000_000,
        embed_bytes=29_500_000_000,
        native_context=262_144,
        supported_contexts=(65_536, 131_072, 262_144),
    ),
    ModelRow(
        model_id="qwen3-8-flash-next-iq3xxs",
        family="Qwen 3.8 Flash Next",
        quant="GSQ-RCO-IQ3_XXS",
        pack_bytes=75_840_000_000,
        expert_bytes=42_920_000_000,
        expert_layers=48,
        dense_bytes=5_600_000_000,
        embed_bytes=29_500_000_000,
        native_context=262_144,
        supported_contexts=(65_536, 131_072, 262_144),
    ),
    ModelRow(
        model_id="glm-5-3-flash-maya-xs",
        family="GLM 5.3 Flash",
        quant="Maya-XS (IQ1_S routed / Q4_K attention)",
        pack_bytes=74_530_000_000,
        expert_bytes=67_360_000_000,
        expert_layers=43,
        dense_bytes=4_600_000_000,
        embed_bytes=2_570_000_000,
        native_context=131_072,
        supported_contexts=(65_536, 131_072),
    ),
    ModelRow(
        model_id="glm-5-3-flash-maya-l",
        family="GLM 5.3 Flash",
        quant="Maya-L (IQ3_S routed / Q6_K attention)",
        pack_bytes=157_440_000_000,
        expert_bytes=148_480_000_000,
        expert_layers=43,
        dense_bytes=5_600_000_000,
        embed_bytes=3_360_000_000,
        native_context=131_072,
        supported_contexts=(65_536, 131_072),
    ),
    ModelRow(
        model_id="deepseek-v4-1-flash-1017-iq1m",
        family="DeepSeek V4.1 Flash",
        quant="GSQ-RCO-IQ1_M",
        pack_bytes=120_710_000_000,
        expert_bytes=114_560_000_000,
        expert_layers=40,
        dense_bytes=4_000_000_000,
        embed_bytes=2_150_000_000,
        expert_slots=15_360,   # documented: 384 per layer x 40 layers
        native_context=262_144,
        supported_contexts=(65_536, 131_072, 262_144),
    ),
    ModelRow(
        model_id="darwin-180b-rsi-strata",
        family="Darwin 180B RSI",
        quant="MQ-Q5-SSD-PLE-BF16",
        pack_bytes=134_470_000_000,
        expert_bytes=76_500_000_000,
        expert_layers=48,
        dense_bytes=6_000_000_000,
        embed_bytes=51_970_000_000,
        native_context=262_144,
        supported_contexts=(65_536, 131_072, 262_144),
    ),
)


def family_rows() -> tuple[ModelRow, ...]:
    """The shipped Strata family rows (one per quant variant)."""
    return FAMILY_ROWS


__all__ = [
    "ACCELERATED_GAIN",
    "ACTIVATION_HEADROOM_GIB",
    "BANDWIDTH_GIB_S",
    "EXPERT_SLOT_BYTES",
    "FAMILY_ROWS",
    "KV_BASE_GIB",
    "KV_MODE_ATTENTION_COST",
    "KV_MODE_SCALE",
    "MIN_CONTEXT_TOKENS",
    "NO_AVX2_PENALTY",
    "PREFERRED_CONTEXT_TOKENS",
    "MachineBudget",
    "ModelRow",
    "Placement",
    "Refusal",
    "best_placement",
    "enumerate_placements",
    "family_rows",
    "recommend_for_machine",
]
