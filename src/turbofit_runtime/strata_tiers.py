"""Strata tier placement: which large MoE fits which hardware tier, at what context.

Strata is a Colibri-style engine: it serves large MoE packs with expert tiering,
so a model whose *pack* is far larger than the card can still hold its required
context because the pack streams from system RAM and SSD. That makes the
placement question different from llama.cpp's:

* llama.cpp answers "do weights + KV fit this VRAM band?"
* Strata answers "is this pack documented to run on this VRAM class, with
  enough system RAM, and can it hold the context we would claim?"

The rules here are data-driven from the repo, not guessed:

* ``references/model-recipes.json`` -> ``strata.models[*].context_tokens`` is the
  documented context curve per VRAM class (``8gb`` -> 128K, ``12gb+`` -> 262K).
* ``references/engine-serve-matrix.json`` -> the ``strata`` engine note and the
  per-model ``pairs`` entry ("12-24 GB VRAM + 64 GB RAM", "8 GB cards start at
  128K").
* ``references/model-footprints.json`` -> the measured pack size used for the
  RAM floor of rows whose vendor documentation does not name one.
* ``context_floor.MIN_CONTEXT_TOKENS`` -> the hard 64K gate. A row that cannot
  hold 64K at a tier is not admitted at that tier, whatever its footprint.

Capability is capability, not brand: a host is Strata-capable when its CPU
advertises AVX2, which is what Strata's own docs specify. AVX-512 is reported as
an accelerator (faster prefill) and never used to exclude a host. Where AVX2 is
absent the rows are *excluded with the missing flag named*; they are never
silently ranked away, and they are never recommended on a host that cannot
decode them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .context_floor import MIN_CONTEXT_TOKENS, floor_refusal, meets_context_floor
from .cpu_features import (
    STRATA_ACCELERATOR,
    STRATA_REQUIRED,
    missing_flags,
    strata_accelerated,
    strata_supported,
)
from .experts import available_ram_bytes
from .quant_policy import default_admits
from .tier_fit import (
    CONTEXT_SLUGS,
    DEFAULT_RAM_GIB,
    RAM_RESERVE_BYTES,
    TIERS,
    default_ram_gib,
    kv_bytes,
    tier_budget_bytes,
)

GIB = 1024 ** 3
ENGINE = "strata"
#: Strata's own documented VRAM floor: 12 GB recommended, 8 GB "runs, slowly"
#: (``docs/DETAILS.md`` in Niko1221/Strata).
STRATA_RECOMMENDED_VRAM_GB = 12
STRATA_MINIMUM_VRAM_GB = 8
SLOW_8GB_CAVEAT = "8 GB runs slowly; 12 GB VRAM is recommended (docs/DETAILS.md)"


def default_root() -> Path:
    """The repository root that owns ``references/`` and ``runtime-profiles/``."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Row metadata
# ---------------------------------------------------------------------------
#: Per-family Strata placement metadata. ``min_vram_gb`` and ``min_ram_gib`` are
#: the *documented* floor the engine is known to run at; ``source`` names where
#: the repo says so. A family that appears in ``model-recipes.json`` with
#: ``runtime_flavor == "strata"`` but not here is a bug: ``load_rows`` raises, so
#: a new Strata family cannot reach users without a placement decision.
@dataclass(frozen=True)
class FamilyRule:
    family: str            # catalog ``family`` string
    recipe_alias: str      # recipe alias / strata pack identity
    min_vram_gb: int
    min_ram_gib: int | None  # None -> derive from the measured pack
    source: str
    #: Catalog ids the documented RAM floor was measured for. Variants that
    #: share the family but not the pack (Maya-M/L) get a pack-derived floor
    #: instead of borrowing the small variant's number.
    documented_models: frozenset[str] = frozenset()
    #: Per-catalog-id VRAM floors for a family that ships several packs sized
    #: for different cards (Inkling 2 Mini SQ-10/15/20). Ids not listed fall
    #: back to ``min_vram_gb``. The floor is the card the pack is *sized for*;
    #: below it the pack is refused in favour of the smaller sibling, not
    #: streamed onto a card the vendor never sized it for.
    model_min_vram_gb: tuple[tuple[str, int], ...] = ()


# Owner-set preference order for the Strata chain. Inkling 2 Mini leads until
# Qwen 3.8 Flash Next; everything else keeps its documented order behind them.
# Lower number = preferred. Unlisted families sort after all listed ones.
STRATA_FAMILY_PREFERENCE: tuple[str, ...] = (
    "inkling-2-mini",
    "qwen3-8-flash-next",
)


def _family(model_id: str) -> str:
    """Collapse a pack id to its family for chain ordering."""
    return next((p for p in STRATA_FAMILY_PREFERENCE
                 if model_id.startswith(p)), model_id)


def family_preference(model_id: str) -> int:
    """Rank a pack by its family's position in the owner's preference order."""
    for index, prefix in enumerate(STRATA_FAMILY_PREFERENCE):
        if model_id.startswith(prefix):
            return index
    return len(STRATA_FAMILY_PREFERENCE)


FAMILY_RULES: tuple[FamilyRule, ...] = (
    FamilyRule(
        family="Inkling 2 Mini",
        recipe_alias="inkling-2-mini-strata",
        min_vram_gb=8,
        min_ram_gib=None,
        model_min_vram_gb=(
            ("inkling-2-mini-sq10", 8),
            ("inkling-2-mini-sq15", 16),
            ("inkling-2-mini-sq20", 24),
        ),
        source=(
            "strata-engine/Inkling-2-Mini-Strata pack card: SQ-10 100% VRAM "
            "at 12 GiB, hybrid RAM+VRAM from 4 GiB (the ladder's smallest band "
            "is 8 GB); SQ-15 at 16 GiB; SQ-20 native lossless at 24 GiB+ and "
            "32 GiB+ unified memory; Strata v0.2.19 release notes "
            "(strata-inkling 0.1.65)"
        ),
    ),
    FamilyRule(
        family="Qwen 3.8 Flash Next",
        recipe_alias="qwen3-8-flash-next-strata",
        min_vram_gb=8,
        min_ram_gib=64,
        documented_models=frozenset(
            {"qwen3-8-flash-next-strata", "qwen3-8-flash-next-iq3xxs"}
        ),
        source=(
            "references/model-recipes.json strata.models[qwen3.8-flash-next] "
            "(8gb -> 131072, 12gb+ -> 262144); engine-serve-matrix strata note "
            "'12-24 GB VRAM + 64 GB RAM'; strata.SMALL_CARD_CONTEXT"
        ),
    ),
    FamilyRule(
        family="GLM 5.3 Flash",
        recipe_alias="glm-5-3-flash",
        min_vram_gb=24,
        min_ram_gib=64,
        documented_models=frozenset({"glm-5-3-flash-maya-xs"}),
        source=(
            "allowed_lineup Project Maya note: '64 GB RAM + one 24 GB card at "
            "128K'"
        ),
    ),
    FamilyRule(
        family="DeepSeek V4.1 Flash",
        recipe_alias="deepseek-v4-1-flash",
        min_vram_gb=96,
        min_ram_gib=None,
        source=(
            "README/tier doctrine: DeepSeek V4.1 Flash is a 96 GB+ rung; "
            "model-recipes DeepSeek V4.1 Flash (Strata) native_context 262144"
        ),
    ),
    FamilyRule(
        family="Darwin 180B RSI",
        recipe_alias="darwin-180b-rsi-strata",
        min_vram_gb=96,
        min_ram_gib=None,
        source=(
            "model-recipes Darwin 180B RSI (Strata): PLE/SSD streaming, "
            "native_context 262144"
        ),
    ),
)


@dataclass(frozen=True)
class StrataRow:
    """One Strata-served catalog main, with the numbers the gate needs."""

    model_id: str
    family: str
    label: str
    quant: str
    recipe_alias: str
    weight_bytes: int
    expert_bytes: int
    expert_layers: int
    native_context: int
    supported_contexts: tuple[int, ...]
    context_curve: tuple[tuple[int, int], ...]  # (min_vram_gb, context)
    min_vram_gb: int
    min_ram_gib: int
    source: str
    documented: bool = True
    measured_tps: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pack_gib(self) -> float:
        return self.weight_bytes / GIB

    def context_at(self, tier_gb: int) -> int:
        """Context this row claims at ``tier_gb``, from the documented curve."""
        context = 0
        for min_gb, tokens in self.context_curve:
            if tier_gb >= min_gb:
                context = max(context, tokens)
        if context == 0:
            context = self.native_context
        allowed = set(self.supported_contexts)
        if allowed and context not in allowed:
            # Never claim a context the catalog does not expose a row for.
            candidates = [item for item in allowed if item <= context]
            context = max(candidates) if candidates else 0
        return context


@dataclass(frozen=True)
class StrataPlacement:
    """An admitted Strata row at one tier, at its required context."""

    model_id: str
    family: str
    label: str
    quant: str
    engine: str
    tier_gb: int
    context: int
    total_bytes: int
    vram_bytes: int
    ram_bytes: int
    evidence: str
    reason: str
    caveats: tuple[str, ...] = ()

    @property
    def context_k(self) -> int:
        return self.context // 1024

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "label": self.label,
            "quant": self.quant,
            "engine": self.engine,
            "tier_gb": self.tier_gb,
            "context": self.context,
            "context_k": self.context_k,
            "total_gib": round(self.total_bytes / GIB, 2),
            "vram_gib": round(self.vram_bytes / GIB, 2),
            "ram_gib": round(self.ram_bytes / GIB, 2),
            "evidence": self.evidence,
            "reason": self.reason,
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class StrataRefusal:
    """A Strata row that was considered at a tier and refused, with the reason."""

    model_id: str
    family: str
    label: str
    tier_gb: int
    context: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "label": self.label,
            "tier_gb": self.tier_gb,
            "context": self.context,
            "context_k": self.context // 1024,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _strata_recipes(root: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads((root / "references/model-recipes.json").read_text(encoding="utf-8"))
    recipes = payload.get("models") or {}
    result = {}
    for _name, recipe in recipes.items():
        if isinstance(recipe, Mapping) and recipe.get("runtime_flavor") == ENGINE:
            result[str(recipe.get("alias") or "")] = recipe
    return result


def _context_curve(recipe: Mapping[str, Any], native_context: int) -> tuple[tuple[int, int], ...]:
    documented = (payload_strata_root(recipe) or {}).get("context_tokens") or {}
    curve: list[tuple[int, int]] = []
    for key, tokens in documented.items():
        if not isinstance(tokens, int):
            continue
        key = str(key).strip().lower()
        if key.endswith("gb+"):
            curve.append((int(key[:-3]), tokens))
        elif key.endswith("gb"):
            curve.append((int(key[:-2]), tokens))
    if not curve:
        return ((0, native_context),)
    return tuple(sorted(curve))


def payload_strata_root(recipe: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The recipe's ``strata`` block, when it carries one."""
    block = recipe.get("strata")
    return block if isinstance(block, Mapping) else None


def load_rows(root: str | Path | None = None) -> tuple[StrataRow, ...]:
    """Every Strata-served catalog main, with documented placement metadata.

    Raises ``ValueError`` when a Strata recipe has no ``FAMILY_RULES`` entry, so
    a new family fails loudly instead of silently missing from placement.
    """
    root = Path(root) if root is not None else default_root()
    catalog = json.loads((root / "references/model-catalog.json").read_text(encoding="utf-8"))
    footprints = json.loads(
        (root / "references/model-footprints.json").read_text(encoding="utf-8")
    ).get("models") or {}
    strata_payload = json.loads(
        (root / "references/model-recipes.json").read_text(encoding="utf-8")
    ).get("strata") or {}
    documented_models = strata_payload.get("models") or {}
    recipes = _strata_recipes(root)
    rules = {rule.family: rule for rule in FAMILY_RULES}

    covered = set()
    rows: list[StrataRow] = []
    for model in catalog.get("models") or []:
        if "main" not in (model.get("roles") or []):
            continue
        family = str(model.get("family") or "")
        rule = rules.get(family)
        if rule is None:
            continue
        record = footprints.get(str(model["id"]))
        if not record or not record.get("weight_bytes"):
            continue
        recipe = recipes.get(rule.recipe_alias) or {}
        native = int(recipe.get("native_context") or 0) or _largest_context(model)
        documented = documented_models.get(_family_key(rule.recipe_alias)) or {}
        curve = _context_curve({"strata": documented}, native)
        pack_gib = int(record["weight_bytes"]) / GIB
        derived_ram = max(32, int(pack_gib + 0.999))
        # Families Project Maya documents (GLM 5.3 Flash: "64 GB RAM + one 24 GB
        # card") keep the vendor's RAM figure across their quants; families with
        # no documented figure get a pack-derived floor instead of a guess.
        min_ram = rule.min_ram_gib if rule.min_ram_gib is not None else derived_ram
        # A family can ship packs sized for different cards (Inkling 2 Mini
        # SQ-10/15/20): each pack keeps its own documented VRAM floor.
        min_vram = dict(rule.model_min_vram_gb).get(str(model["id"]), rule.min_vram_gb)
        rows.append(
            StrataRow(
                model_id=str(model["id"]),
                family=family,
                label=str(model.get("name") or model["id"]),
                quant=str(model.get("quantization") or ""),
                recipe_alias=rule.recipe_alias,
                weight_bytes=int(record["weight_bytes"]),
                expert_bytes=int(record.get("expert_bytes") or 0),
                expert_layers=int(record.get("expert_layers") or 0),
                native_context=native,
                supported_contexts=tuple(
                    int(item) for item in (model.get("supported_contexts") or [])
                ),
                context_curve=curve,
                min_vram_gb=min_vram,
                min_ram_gib=int(min_ram),
                source=rule.source,
                documented=(
                    not rule.documented_models
                    or str(model["id"]) in rule.documented_models
                ),
                measured_tps=dict(documented.get("measured_tps") or {}),
            )
        )
        covered.add(rule.recipe_alias)

    missing = sorted(set(recipes) - covered)
    if missing:
        raise ValueError(
            "Strata recipe(s) have no tier placement rule in strata_tiers.FAMILY_RULES: "
            + ", ".join(missing)
        )
    return tuple(sorted(rows, key=lambda row: row.model_id))


def _family_key(recipe_alias: str) -> str:
    """Map a recipe alias to the ``strata.models`` family key.

    The recipe alias names the pack (``glm-5-3-flash``); ``strata.models`` keys
    the family the engine serves (``qwen3.8-flash-next``). Qwen's family key
    differs from its alias, and it carries a vendor-documented per-VRAM-class
    context curve; Inkling's alias carries the ``-strata`` suffix while its
    ``strata.models`` entry keys the bare family (its curve is derived, and
    says so).
    """
    if recipe_alias.startswith("qwen3-8-flash-next"):
        return "qwen3.8-flash-next"
    if recipe_alias.startswith("inkling-2-mini"):
        return "inkling-2-mini"
    return recipe_alias


def _largest_context(model: Mapping[str, Any]) -> int:
    supported = [int(item) for item in (model.get("supported_contexts") or [])]
    return max(supported) if supported else 262_144


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrataCapability:
    """Whether this CPU can decode Strata rows, and why not when it cannot."""

    supported: bool | None
    missing_flags: tuple[str, ...]
    reason: str
    accelerated: bool | None = None
    accelerator_note: str = ""

    @property
    def excluded(self) -> bool:
        return self.supported is False

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "missing_flags": list(self.missing_flags),
            "reason": self.reason,
            "accelerated": self.accelerated,
            "accelerator_note": self.accelerator_note,
        }


def capability(flags: Iterable[str] | None) -> StrataCapability:
    """Classify a CPU flag set against Strata's documented AVX2 requirement.

    AVX-512 is reported as an accelerator, never as a gate: a host with AVX2
    and no AVX-512 runs Strata, just with slower prefill. ``None`` means "this
    platform cannot be interrogated" (no /proc): the rows are not claimed and
    not excluded, because neither answer would be honest.
    """
    normalized = frozenset(flags) if flags is not None else None
    supported = strata_supported(normalized)
    missing = missing_flags(STRATA_REQUIRED, normalized)
    accelerated = strata_accelerated(normalized)
    if accelerated is True:
        accelerator_note = (
            "AVX-512 present: faster prefill (an accelerator, not a requirement)."
        )
    elif accelerated is False:
        accelerator_note = (
            "AVX-512 absent: Strata runs on AVX2; prefill is slower than an "
            "AVX-512 host."
        )
    else:
        accelerator_note = "AVX-512 presence unknown."
    if supported is True:
        reason = (
            "CPU advertises avx2; Strata rows are eligible. " + accelerator_note
        )
    elif supported is False:
        reason = (
            f"Strata models are deferred because this CPU lacks {', '.join(missing)} "
            "(Strata's documented CPU requirement is AVX2; without it setup installs "
            "an experimental, slow build)."
        )
    else:
        reason = (
            "CPU flags were not probed (no /proc/cpuinfo, or no probe was passed), "
            "so Strata rows are not claimed (the engine requires avx2)."
        )
    return StrataCapability(
        supported=supported,
        missing_flags=missing,
        reason=reason,
        accelerated=accelerated,
        accelerator_note=accelerator_note,
    )


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def place_row(
    row: StrataRow,
    *,
    tier_gb: int,
    ram_bytes: int,
) -> StrataPlacement | StrataRefusal:
    """Place one row at one tier, or refuse it with the reason the gate gave."""
    context = row.context_at(tier_gb)
    if not meets_context_floor(context):
        return StrataRefusal(
            model_id=row.model_id, family=row.family, label=row.label,
            tier_gb=tier_gb, context=context,
            reason=floor_refusal(context, subject=row.label),
        )
    if tier_gb < row.min_vram_gb:
        return StrataRefusal(
            model_id=row.model_id, family=row.family, label=row.label,
            tier_gb=tier_gb, context=context,
            reason=(
                f"{row.label} is documented from {row.min_vram_gb} GB of VRAM"
                + (
                    f" ({SLOW_8GB_CAVEAT})"
                    if row.min_vram_gb <= STRATA_MINIMUM_VRAM_GB
                    else ""
                )
                + f"; this tier is {tier_gb} GB"
            ),
        )
    ram_gib = ram_bytes / GIB
    if ram_gib < row.min_ram_gib:
        return StrataRefusal(
            model_id=row.model_id, family=row.family, label=row.label,
            tier_gb=tier_gb, context=context,
            reason=(
                f"{row.label} needs at least {row.min_ram_gib} GiB of system RAM "
                f"for its {row.pack_gib:.1f} GiB pack ({row.source}); this host "
                f"offers {ram_gib:.0f} GiB"
            ),
        )
    total = row.weight_bytes + kv_bytes(context)
    vram_bytes = min(total, tier_budget_bytes(tier_gb))
    ram_bytes_used = total - vram_bytes
    caveats: tuple[str, ...] = ()
    if tier_gb < STRATA_RECOMMENDED_VRAM_GB:
        caveats = (SLOW_8GB_CAVEAT,)
    reason = (
        f"strata: {row.label} ({row.quant}) holds {context // 1024}K at "
        f"{tier_gb} GB — {total / GIB:.1f} GiB pack, {vram_bytes / GIB:.1f} GiB "
        f"VRAM + {ram_bytes_used / GIB:.1f} GiB expert-tiered RAM/SSD"
    )
    if caveats:
        reason += f" (caveat: {caveats[0]})"
    return StrataPlacement(
        model_id=row.model_id,
        family=row.family,
        label=row.label,
        quant=row.quant,
        engine=ENGINE,
        tier_gb=tier_gb,
        context=context,
        total_bytes=total,
        vram_bytes=vram_bytes,
        ram_bytes=ram_bytes_used,
        evidence="documented-strata-placement",
        reason=reason,
        caveats=caveats,
    )


def place_tier(
    tier_gb: int,
    *,
    ram_bytes: int,
    rows: Iterable[StrataRow] | None = None,
    root: str | Path | None = None,
) -> tuple[list[StrataPlacement], list[StrataRefusal]]:
    """Admitted and refused rows for one tier, admitted ones ranked first-best."""
    rows = tuple(rows) if rows is not None else load_rows(root)
    admitted: list[StrataPlacement] = []
    refused: list[StrataRefusal] = []
    for row in rows:
        result = place_row(row, tier_gb=tier_gb, ram_bytes=ram_bytes)
        if isinstance(result, StrataPlacement):
            admitted.append(result)
        else:
            refused.append(result)
    # Owner preference first: Inkling 2 Mini is the preferred pick wherever it is
    # admitted, with Qwen 3.8 Flash Next next in the chain. This is a deliberate
    # product decision, not a size heuristic — Inkling is a smaller MoE that
    # matches or beats the 27B on agentic suites, and SQ-20 is bit-exact lossless.
    # It outranks the longer-context Flash Next at 12-16 GB on purpose; the 64K
    # floor still holds, so no sub-floor model is ever promoted by this.
    # Within a family: longest admitted context, then the documented pack, then
    # the larger pack.
    documented = {row.model_id: row.documented for row in rows}
    # One pack per family leads the chain, so the top reads Inkling -> Flash Next
    # rather than three Inkling packs. The remaining packs stay in the list, just
    # behind every family's best, so nothing is dropped from the tier view.
    best_ctx: dict[str, tuple] = {}
    for item in admitted:
        fam = _family(item.model_id)
        rank = (-item.context, not documented.get(item.model_id, True),
                -item.total_bytes, item.model_id)
        if fam not in best_ctx or rank < best_ctx[fam][0]:
            best_ctx[fam] = (rank, item.model_id)
    leaders = {v[1] for v in best_ctx.values()}

    admitted.sort(
        key=lambda item: (
            0 if item.model_id in leaders else 1,
            family_preference(item.model_id),
            -item.context,
            not documented.get(item.model_id, True),
            -item.total_bytes,
            item.model_id,
        )
    )
    refused.sort(key=lambda item: item.model_id)
    return admitted, refused


# ---------------------------------------------------------------------------
# The llama.cpp comparison
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LlamaCandidate:
    """The best llama.cpp catalog main a tier admits at the 64K floor."""

    model_id: str
    label: str
    context: int
    weight_bytes: int = 0

    @property
    def pack_gib(self) -> float:
        return self.weight_bytes / GIB

    @property
    def context_k(self) -> int:
        return self.context // 1024


def _llamacpp_aliases(root: Path) -> frozenset[str]:
    """Recipe aliases served by llama.cpp (mainline or the Prism fork)."""
    payload = json.loads((root / "references/model-recipes.json").read_text(encoding="utf-8"))
    llama_flavors = {None, "", "mainline", "prism"}
    return frozenset(
        str(recipe.get("alias") or "")
        for recipe in (payload.get("models") or {}).values()
        if isinstance(recipe, Mapping) and recipe.get("runtime_flavor") in llama_flavors
    )


def best_llamacpp(
    tier_gb: int,
    *,
    ram_bytes: int | None = None,
    root: str | Path | None = None,
) -> LlamaCandidate | None:
    """The strongest llama.cpp-served main this band fits at >= 64K.

    "Strongest" is the same ordering the ladder uses: the largest context the
    model holds entirely on the card, then the largest weights. Strata rows and
    other engines (vLLM, SGLang) are excluded here — this is the llama.cpp
    baseline Strata is compared against.
    """
    root = Path(root) if root is not None else default_root()
    if ram_bytes is None:
        ram_bytes = available_ram_bytes() or int(default_ram_gib() * GIB)
    catalog = json.loads((root / "references/model-catalog.json").read_text(encoding="utf-8"))
    footprints = json.loads(
        (root / "references/model-footprints.json").read_text(encoding="utf-8")
    ).get("models") or {}
    strata_families = {rule.family for rule in FAMILY_RULES}
    llamacpp = _llamacpp_aliases(root)
    best: tuple[int, int, str, str] | None = None
    for model in catalog.get("models") or []:
        if "main" not in (model.get("roles") or []):
            continue
        if str(model.get("family") or "") in strata_families:
            continue
        if str(model["id"]) not in llamacpp:
            continue
        record = footprints.get(str(model["id"]))
        if not record or not record.get("weight_bytes"):
            continue
        admitted, _reason = default_admits(model, fits_fully_in_vram=True)
        if not admitted:
            continue
        contexts = [
            int(context)
            for context in (model.get("supported_contexts") or [])
            if meets_context_floor(int(context))
            and tier_fit_placement(
                str(model["id"]), record, int(context), tier_gb, ram_bytes
            )
            is not None
        ]
        if not contexts:
            continue
        context = max(contexts)
        key = (context, int(record["weight_bytes"]), str(model["id"]), str(model.get("name")))
        if best is None or key > best:
            best = key
    if best is None:
        return None
    return LlamaCandidate(model_id=best[2], label=best[3], context=best[0], weight_bytes=best[1])


# ---------------------------------------------------------------------------
# The recommendation view
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrataTierView:
    """What Strata contributes to one host's recommendation, and why."""

    tier_gb: int
    ram_gib: float
    capability: StrataCapability
    admitted: tuple[StrataPlacement, ...]
    refused: tuple[StrataRefusal, ...]
    llamacpp_best: LlamaCandidate | None
    preferred_over_llamacpp: bool
    reason: str

    @property
    def preferred(self) -> StrataPlacement | None:
        return self.admitted[0] if self.admitted else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": ENGINE,
            "tier_gb": self.tier_gb,
            "ram_gib": round(self.ram_gib, 1),
            "capability": self.capability.as_dict(),
            "admitted": [item.as_dict() for item in self.admitted],
            "refused": [item.as_dict() for item in self.refused],
            "llamacpp_best": (
                {
                    "model_id": self.llamacpp_best.model_id,
                    "label": self.llamacpp_best.label,
                    "context": self.llamacpp_best.context,
                    "context_k": self.llamacpp_best.context_k,
                    "pack_gib": round(self.llamacpp_best.pack_gib, 1),
                }
                if self.llamacpp_best
                else None
            ),
            "preferred_over_llamacpp": self.preferred_over_llamacpp,
            "reason": self.reason,
        }


def tier_view(
    *,
    tier_gb: int,
    ram_bytes: int | None = None,
    cpu_flags: Iterable[str] | None = None,
    root: str | Path | None = None,
    rows: Iterable[StrataRow] | None = None,
) -> StrataTierView:
    """Strata's contribution at one tier for one host.

    ``cpu_flags`` is passed through so capability is decided by what the CPU
    advertises, never by the card or the brand.
    """
    if ram_bytes is None:
        ram_bytes = available_ram_bytes()
        if not ram_bytes:
            ram_bytes = int(default_ram_gib() * GIB)
    capability_state = capability(cpu_flags)
    if capability_state.supported is not True:
        return StrataTierView(
            tier_gb=tier_gb,
            ram_gib=ram_bytes / GIB,
            capability=capability_state,
            admitted=(),
            refused=(),
            llamacpp_best=best_llamacpp(tier_gb, ram_bytes=ram_bytes, root=root),
            preferred_over_llamacpp=False,
            reason=capability_state.reason,
        )
    admitted, refused = place_tier(tier_gb, ram_bytes=ram_bytes, rows=rows, root=root)
    best = best_llamacpp(tier_gb, ram_bytes=ram_bytes, root=root)
    preferred = False
    if admitted:
        top = admitted[0]
        reason = top.reason
        # The llama.cpp comparison asks what Strata CAN hold at this tier, not
        # what the owner's preference happens to put first. Preference picks the
        # recommended pack; capability is the longest context any admitted row
        # reaches. Conflating the two demotes the whole engine when a preferred
        # but shorter-context family leads the chain.
        deepest = max(admitted, key=lambda item: item.context)
        if best is not None and best.context > deepest.context:
            preferred = False
            reason += (
                f"; NOT preferred at this tier: the best llama.cpp option, "
                f"{best.label}, holds {best.context_k}K against Strata's "
                f"{deepest.context_k}K"
            )
        elif best is not None:
            preferred = True
            reason += (
                f"; the best llama.cpp option, {best.label}, holds "
                f"{best.context_k}K at {best.pack_gib:.0f} GiB against Strata's "
                f"{deepest.context_k}K; {top.label} leads by owner preference"
            )
        else:
            preferred = True
            reason += "; no llama.cpp option holds 64K at this tier"
    elif any("floor" in item.reason for item in refused):
        reason = (
            "no Strata row holds 64K at this tier: "
            + ", ".join(
                f"{item.label} tops out at {item.context // 1024}K"
                for item in refused if "floor" in item.reason
            )
            + " (the 64K context floor is a hard gate)"
        )
    elif refused and all("documented from" in item.reason for item in refused):
        reason = (
            f"no Strata row is documented for {tier_gb} GB of VRAM: the smallest "
            f"documented floor is {STRATA_MINIMUM_VRAM_GB} GB ({SLOW_8GB_CAVEAT})"
        )
    else:
        reason = (
            f"no Strata row fits {tier_gb} GB VRAM with {ram_bytes / GIB:.0f} GiB "
            "of system RAM at 64K or more"
        )
        if best is not None:
            reason += f"; {best.label} holds {best.context_k}K on llama.cpp"
    return StrataTierView(
        tier_gb=tier_gb,
        ram_gib=ram_bytes / GIB,
        capability=capability_state,
        admitted=tuple(admitted),
        refused=tuple(refused),
        llamacpp_best=best,
        preferred_over_llamacpp=preferred,
        reason=reason,
    )


def placement_table(
    *,
    ram_gib: float | None = None,
    tiers: Iterable[int] = TIERS,
    root: str | Path | None = None,
) -> list[StrataTierView]:
    """The whole tier table for a stated RAM budget (the audit artefact)."""
    budget = int((ram_gib if ram_gib is not None else DEFAULT_RAM_GIB) * GIB)
    return [
        tier_view(tier_gb=int(tier), ram_bytes=budget, cpu_flags=STRATA_REQUIRED, root=root)
        for tier in tiers
    ]


def context_slug(context: int) -> str:
    if context in CONTEXT_SLUGS:
        return CONTEXT_SLUGS[context]
    raise ValueError(f"unsupported context for a Strata row: {context}")


def tier_for_vram(vram_gb: float) -> int:
    """The hardware tier a VRAM figure belongs to (largest tier <= the figure)."""
    tier = 0
    for candidate in TIERS:
        if float(vram_gb) >= candidate:
            tier = candidate
    return tier


_QUANT_TOKEN = __import__("re").compile(r"(IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*)")


def _quant_token(quantization: str) -> str | None:
    matches = _QUANT_TOKEN.findall(str(quantization or ""))
    return matches[-1].strip("_") if matches else None


def variant_weight_bytes(
    model_id: str, *, root: str | Path | None = None
) -> int | None:
    """Bytes of the *file this variant launches*, plus its shared projector.

    ``references/model-footprints.json`` is per catalog **family**: it sums every
    pinned file in the family, so a family with several quants (or an MTP sidecar)
    overstates any single variant. Placing a variant against that total would
    refuse a model that fits. This resolves the variant's own GGUF by its quant
    token inside the family's repos - the same file the recipe names - and
    returns ``None`` when the match is ambiguous rather than guessing.
    """
    root = Path(root) if root is not None else default_root()
    catalog = json.loads((root / "references/model-catalog.json").read_text(encoding="utf-8"))
    footprints = json.loads(
        (root / "references/model-footprints.json").read_text(encoding="utf-8")
    ).get("models") or {}
    manifest = json.loads(
        (root / "references/artifact-manifest.json").read_text(encoding="utf-8")
    )
    record = footprints.get(model_id)
    model = next(
        (item for item in catalog.get("models") or [] if str(item.get("id")) == model_id),
        None,
    )
    if not record or model is None:
        return None
    token = _quant_token(str(model.get("quantization") or ""))
    if not token:
        return None
    repos = set(record.get("repos") or [])
    normalized = token.replace("-", "_").upper()
    matches: dict[str, int] = {}
    projectors: dict[str, int] = {}
    for artifact in manifest.get("artifacts") or []:
        if str(artifact.get("repo_id")) not in repos:
            continue
        path = str(artifact.get("path") or "")
        name = Path(path).name
        if "mmproj" in name.lower():
            if model_id in (artifact.get("families") or []):
                projectors[path] = int(artifact["size_bytes"])
            continue
        if normalized in name.replace("-", "_").upper():
            matches[path] = int(artifact["size_bytes"])
    if len(matches) > 1:
        # A family can pin an MTP sidecar next to the base file. The base file
        # is the one the engine loads; the sidecar is a drafter, not weights.
        without_mtp = {
            path: size for path, size in matches.items()
            if "mtp" not in Path(path).name.lower()
        }
        matches = without_mtp
    if len(matches) != 1:
        return None
    return sum(matches.values()) + sum(projectors.values())


def placed_context(
    model_id: str,
    tier_gb: int,
    ram_bytes: int,
    *,
    root: str | Path | None = None,
) -> int | None:
    """Largest context this variant holds at this tier; ``None`` when unknown.

    Uses the variant's own artifact size (see :func:`variant_weight_bytes`) and
    :func:`tier_fit.placement_for` - the same arithmetic the launcher uses for
    ``--n-cpu-moe`` - so a dense build that only fits by spilling fails while an
    MoE that legitimately offloads experts passes. ``None`` means the variant's
    size could not be resolved; callers must treat that as *unknown*, not as a
    refusal.
    """
    root = Path(root) if root is not None else default_root()
    catalog = json.loads((root / "references/model-catalog.json").read_text(encoding="utf-8"))
    model = next(
        (item for item in catalog.get("models") or [] if str(item.get("id")) == model_id),
        None,
    )
    if model is None:
        return None
    weight_bytes = variant_weight_bytes(model_id, root=root)
    if weight_bytes is None:
        return None
    record = {"weight_bytes": weight_bytes}
    # Expert share is a family measurement; scale it to this variant's size so
    # an offload plan is still possible when the variant's file is smaller.
    footprints = json.loads(
        (root / "references/model-footprints.json").read_text(encoding="utf-8")
    ).get("models") or {}
    family_record = footprints.get(model_id) or {}
    family_bytes = int(family_record.get("weight_bytes") or 0)
    if family_record.get("expert_bytes") and family_bytes > 0:
        share = min(1.0, weight_bytes / family_bytes)
        record["expert_bytes"] = int(int(family_record["expert_bytes"]) * share)
        record["expert_layers"] = int(family_record["expert_layers"])
    supported = model.get("supported_contexts") or [65_536, 131_072, 262_144, 1_048_576]
    contexts = [
        int(context)
        for context in supported
        if meets_context_floor(int(context))
        and tier_fit_placement(model_id, record, int(context), tier_gb, ram_bytes) is not None
    ]
    return max(contexts) if contexts else None


def floor_state(
    model_id: str,
    *,
    tier_gb: int,
    ram_bytes: int,
    root: str | Path | None = None,
) -> bool | None:
    """Whether a variant holds >= 64K here: True, False, or None when unknown."""
    root = Path(root) if root is not None else default_root()
    if variant_weight_bytes(model_id, root=root) is None:
        return None
    return placed_context(model_id, tier_gb, ram_bytes, root=root) is not None


def tier_fit_placement(
    model_id: str, record: Mapping[str, Any], context: int, tier_gb: int, ram_bytes: int
) -> Any:
    """``tier_fit.placement_for`` behind a seam (keeps imports one-directional)."""
    from .tier_fit import placement_for

    return placement_for(model_id, record, context, tier_gb, ram_bytes)
