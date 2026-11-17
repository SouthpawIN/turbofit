"""One answer to "what should I run on this machine, and what is missing?".

The dashboard already exposes the leaderboard (what was measured), the artifact
manifest (what the campaign considers installed) and the CPU probe (what the
engine can execute). Reconciling those three by hand is the last manual step
before a first run, and it is exactly where people go wrong: a leaderboard entry
that looks great is useless if its weights are absent, and a "missing model"
message that hides the reason sends the user hunting in the wrong place.

This module joins them for one hardware tier at a time and states, in one
sentence, either what to run or what blocks it. It owns no new definitions:

* the recommendation comes from :mod:`turbofit_runtime.intel_leaderboard`;
* what a configuration needs comes from :mod:`turbofit_runtime.recipes`
  (the same resolved components the launcher runs);
* "installed" is :func:`turbofit_runtime.artifacts.inspect_file` -- the same
  size- and stub-aware definition the campaign uses;
* engine support on this CPU is :mod:`turbofit_runtime.cpu_features`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifacts import expected_size_for_model_root, human_bytes, inspect_file
from .cpu_features import (
    STRATA_ACCELERATOR,
    STRATA_REQUIRED,
    cpu_flags,
    missing_flags,
    strata_accelerated,
    strata_supported,
)

SCHEMA = "turbofit.fit-view/v1"

#: The auxiliary every tier runs with. A measured pairing may name something
#: else; the fit view still reports Maple because that is what ships.
MAPLE_AUX_ID = "maple-preview-tq2"

ENGINE_LABELS = {
    "mainline": "llama.cpp",
    "ik": "llama.cpp (ik)",
    "prism": "prism-llama.cpp",
    "strata": "strata",
    "vllm": "vllm",
}

STATUS_READY = "ready"
STATUS_MISSING = "missing-artifacts"
STATUS_ENGINE = "engine-unsupported"
STATUS_MEMORY = "waiting-memory"
STATUS_UNMEASURED = "no-measurements"
STATUS_UNRESOLVABLE = "unresolvable"


def engine_label(flavor: str | None) -> str:
    """Human name for a recipe runtime flavor."""
    if not flavor:
        return "unknown engine"
    return ENGINE_LABELS.get(str(flavor), str(flavor))


def _cpu_report(flags: frozenset[str] | None) -> dict[str, Any]:
    missing = list(missing_flags(STRATA_REQUIRED, flags)) if flags is not None else []
    supported = strata_supported(flags)
    accelerated = strata_accelerated(flags)
    if supported is None:
        statement = "This host exposes no CPU feature flags, so engine support could not be checked."
    elif supported and accelerated:
        statement = "This CPU has AVX2 and AVX-512, so Strata's expert kernels run at full prefill speed."
    elif supported:
        statement = (
            "This CPU has AVX2, which is what Strata requires; AVX-512 is absent, "
            "so prefill is slower than an AVX-512 host."
        )
    else:
        statement = (
            "This CPU lacks " + ", ".join(missing) + ", so Strata cannot decode here "
            "(its documented requirement is AVX2)."
        )
    return {
        "flags_known": flags is not None,
        "strata_supported": supported,
        "missing_strata_flags": missing,
        "statement": statement,
    }


def _engine_state(
    component: Any,
    flags: frozenset[str] | None,
    binary_installed: Callable[[str], bool],
) -> dict[str, Any]:
    flavor = str(getattr(component, "engine", "") or "")
    command = tuple(getattr(component, "command", ()) or ())
    binary = command[0] if command else ""
    installed = bool(binary) and bool(binary_installed(binary))
    usable = installed
    reason: str | None = None
    if flavor == "strata":
        supported = strata_supported(flags)
        if supported is False:
            usable = False
            reason = (
                "Strata needs AVX2 and this CPU lacks "
                + ", ".join(missing_flags(STRATA_REQUIRED, flags))
                + ", so it cannot decode here."
            )
        elif supported is None and usable:
            reason = "CPU support for Strata could not be confirmed on this host."
        elif strata_accelerated(flags) is False:
            missing_accel = ", ".join(missing_flags(STRATA_ACCELERATOR, flags))
            reason = (
                "AVX2 present, so Strata runs; AVX-512 (" + missing_accel + ") is "
                "absent, which costs prefill speed only."
            )
    if not installed:
        usable = False
        reason = reason or f"the engine binary is not installed at {binary or '<unknown>'}."
    return {
        "name": engine_label(flavor),
        "flavor": flavor,
        "binary": binary,
        "installed": installed,
        "usable": usable,
        "reason": reason,
    }


def _pack_path(component: Any) -> str:
    """The expert pack directory a Strata launch points at, if it has one."""
    command = [str(item) for item in getattr(component, "command", ()) or ()]
    if "--pack" in command:
        index = command.index("--pack") + 1
        if index < len(command):
            return str(command[index])
    return ""


def _path_entry(
    *,
    role: str,
    path: str,
    model_root: Path,
    kind: str = "file",
) -> dict[str, Any] | None:
    """One artifact row, using the campaign's definition of "installed"."""
    if not path:
        return None
    target = Path(path)
    if kind == "directory" or target.is_dir():
        present = target.is_dir()
        return {
            "role": role,
            "path": str(target),
            "kind": "directory",
            "present": present,
            "size_bytes": None,
            "expected_bytes": None,
            "missing_bytes": None,
            "reason": "directory present" if present else "missing",
        }
    expected = expected_size_for_model_root(target, model_root=model_root)
    state = inspect_file(target, expected)
    return {
        "role": role,
        "path": str(target),
        "kind": "file",
        "present": state.present,
        "size_bytes": state.size_bytes,
        "expected_bytes": state.expected_bytes,
        "missing_bytes": state.shortfall if not state.present and state.expected_bytes else None,
        "reason": state.reason,
    }


def component_artifacts(component: Any, model_root: Path) -> list[dict[str, Any]]:
    """Artifact rows for one resolved component, in the campaign's vocabulary.

    Public so first-run onboarding asks the *same* question ("is this file
    installed?") that the fit view and the campaign ask.
    """
    role = str(getattr(component, "role", "") or "main")
    rows = [
        _path_entry(role=f"{role}:model", path=str(getattr(component, "model_path", "") or ""), model_root=model_root),
        _path_entry(role=f"{role}:projector", path=str(getattr(component, "projector_path", "") or ""), model_root=model_root),
        _path_entry(role=f"{role}:draft", path=str(getattr(component, "draft_path", "") or ""), model_root=model_root),
    ]
    if str(getattr(component, "engine", "")) == "strata":
        rows.append(
            _path_entry(
                role=f"{role}:expert-pack",
                path=_pack_path(component),
                model_root=model_root,
                kind="directory",
            )
        )
    return [row for row in rows if row is not None]


def _missing(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if not row.get("present")]


def _missing_bytes(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(row["missing_bytes"]) for row in rows if isinstance(row.get("missing_bytes"), int))


def _byte_phrase(total: int, unknown: bool) -> str:
    if total and unknown:
        return f"{human_bytes(total)} (plus files of unknown size)"
    if total:
        return human_bytes(total)
    return "an unknown amount"


def _context_label(context: int) -> str:
    return f"{int(context) // 1024}K" if context else "its own context"


def _sentence(
    *,
    status: str,
    tier_gb: int,
    main_name: str = "",
    context: int = 0,
    missing: Iterable[Mapping[str, Any]] = (),
    families: Iterable[str] = (),
    blocker: Mapping[str, Any] | None = None,
    measured: Mapping[str, Any] | None = None,
    detail: str = "",
) -> str:
    """One sentence a user can act on -- never a status code."""
    rows = list(missing)
    if status == STATUS_UNMEASURED:
        return (
            f"No measurements yet on {tier_gb} GB hardware, so there is nothing to "
            f"recommend for this tier; run the campaign on this machine first."
        )
    if status == STATUS_UNRESOLVABLE:
        return (
            f"The {tier_gb} GB measurement names a configuration this runtime cannot "
            f"resolve into a launch recipe, so it cannot be run as measured."
        )
    if status == STATUS_ENGINE:
        return str((blocker or {}).get("reason") or "the engine cannot run on this machine.")
    if status == STATUS_MEMORY:
        return (
            f"{main_name or 'This configuration'} cannot be loaded on this machine: "
            f"{detail or 'its weights do not fit in VRAM plus usable RAM'}."
        )
    if status == STATUS_MISSING:
        # Report in the order the user thinks about them, not alphabetically.
        wanted = ("model", "projector", "draft", "expert-pack")
        present = {str(row["role"]).split(":", 1)[-1] for row in rows}
        roles = ", ".join(role for role in wanted if role in present) or "weights"
        unknown = any(row.get("missing_bytes") is None for row in rows)
        request = " ".join(f"--family {name}" for name in families)
        fetch = f" run scripts/download-artifacts {request} to fetch it." if request else ""
        engine_note = ""
        if blocker and blocker.get("installed") is False:
            engine_note = f" The {blocker.get('name')} engine is not installed yet either."
        return (
            f"Missing {_byte_phrase(_missing_bytes(rows), unknown)} to run {main_name} "
            f"at {_context_label(context)} ({roles});{fetch}{engine_note}"
        )
    speed = ""
    if measured and measured.get("throughput_tps") is not None:
        speed = f" at {float(measured['throughput_tps']):.1f} tok/s"
    intelligence = ""
    if measured and measured.get("intelligence_score") is not None:
        intelligence = f", intelligence {float(measured['intelligence_score']):.1f}"
    return (
        f"Ready: run {main_name} at {_context_label(context)}{speed}{intelligence} with "
        f"{MAPLE_AUX_ID} as the auxiliary."
    )


def _unmeasured_entry(tier_gb: int) -> dict[str, Any]:
    return {
        "hardware_tier_gb": int(tier_gb),
        "status": STATUS_UNMEASURED,
        "measured": False,
        "recommendation": None,
        "options": [],
        "missing_bytes": 0,
        "message": _sentence(status=STATUS_UNMEASURED, tier_gb=int(tier_gb)),
    }


def _resolve_components(recipe_book: Any, tier: Mapping[str, Any], row: Mapping[str, Any] | None):
    """Resolve the launch pairing exactly as the launcher would.

    Returns ``(main_component, aux_component)``. ``aux_component`` is ``None``
    when the tier's main already *is* Maple, so the auxiliary is the same
    process (shared-main pairing) rather than a second load.
    """
    recommended = tier.get("recommended") or {}
    configuration_id = str(recommended.get("configuration_id") or "")
    main = str((row or {}).get("main") or recommended.get("main") or "")
    context = int((row or {}).get("context") or recommended.get("context") or 0)
    if not main or context <= 0:
        raise ValueError("the measurement records no main family or context")
    if main == MAPLE_AUX_ID:
        component = recipe_book.resolve_component(
            main, role="main", gpu="0", port=11605, context=context
        )
        return component, None
    recipe = recipe_book.resolve_named(configuration_id or main, main, MAPLE_AUX_ID, context)
    main_component = next((item for item in recipe.components if item.role == "main"), recipe.components[0])
    aux_component = next((item for item in recipe.components if item.role == "aux"), None)
    return main_component, aux_component


def _tier_entry(
    tier: Mapping[str, Any],
    *,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    recipe_book: Any,
    model_root: Path,
    flags: frozenset[str] | None,
    binary_installed: Callable[[str], bool],
    model_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    tier_gb = int(tier.get("hardware_tier_gb") or 0)
    recommended = tier.get("recommended")
    if not recommended:
        return _unmeasured_entry(tier_gb)

    configuration_id = str(recommended.get("configuration_id") or "")
    row = rows_by_id.get(configuration_id)
    base = {
        "hardware_tier_gb": tier_gb,
        "measured": True,
        "configuration_id": configuration_id,
        "options": list(tier.get("options") or []),
    }
    try:
        main_component, aux_component = _resolve_components(recipe_book, tier, row)
    except (ValueError, KeyError) as exc:
        return {
            **base,
            "status": STATUS_UNRESOLVABLE,
            "recommendation": None,
            "missing_bytes": 0,
            "message": _sentence(status=STATUS_UNRESOLVABLE, tier_gb=tier_gb),
            "detail": str(exc),
        }

    shared = aux_component is None
    family = str(recommended.get("main") or main_component.family)
    names = model_names or {}
    main_name = str(
        names.get(family)
        or names.get(main_component.family)
        or names.get(str(recommended.get("main") or ""))
        or family
    )
    context = int(recommended.get("context") or main_component.context)
    engine = _engine_state(main_component, flags, binary_installed)
    aux_engine = dict(engine) if shared else _engine_state(aux_component, flags, binary_installed)

    artifacts = component_artifacts(main_component, model_root)
    aux_artifacts: list[dict[str, Any]] = []
    if not shared:
        aux_artifacts = component_artifacts(aux_component, model_root)
        artifacts.extend(aux_artifacts)
    missing = _missing(artifacts)

    # A CPU that cannot execute the engine is not a download problem, so it is
    # reported before (and instead of) missing artifacts.
    memory_reason = next(
        (
            component.offload_reason
            for component in (main_component, aux_component)
            if component is not None and component.offload_infeasible
        ),
        "",
    )
    strata_blocked = main_component.engine == "strata" and engine["usable"] is False
    if memory_reason:
        status = STATUS_MEMORY
    elif strata_blocked:
        status = STATUS_ENGINE
    elif missing:
        status = STATUS_MISSING
    elif not engine["usable"] or not aux_engine["usable"]:
        status = STATUS_ENGINE
    else:
        status = STATUS_READY
    blocker = engine if not engine["usable"] else aux_engine

    # Offload is a placement, not a footnote: a row that only runs because a
    # chunk of its experts live in system RAM must say so, with the RAM it
    # needs, everywhere it is shown.
    offload_reason = str(getattr(main_component, "offload_reason", "") or "")
    n_cpu_moe = int(getattr(main_component, "n_cpu_moe", 0) or 0)
    placement = "offload" if n_cpu_moe > 0 else "gpu"
    placement_block = {
        "kind": placement,
        "n_cpu_moe": n_cpu_moe,
        "ram_requirement": offload_reason if placement == "offload" else "",
        "label": (
            f"offload · {offload_reason}"
            if placement == "offload"
            else f"fits entirely in the {tier_gb} GB memory band"
        ),
    }

    families = [
        str(component.alias or component.family)
        for component in (main_component, aux_component)
        if component is not None
    ]
    message = _sentence(
        status=status,
        tier_gb=tier_gb,
        main_name=main_name,
        context=context,
        missing=missing,
        families=families,
        blocker=blocker,
        measured=recommended,
        detail=memory_reason,
    )
    return {
        **base,
        "status": status,
        "recommendation": {
            "configuration_id": configuration_id,
            "main": main_name,
            "main_family": family,
            "main_alias": main_component.alias,
            "context": context,
            "intelligence_score": recommended.get("intelligence_score"),
            "throughput_tps": recommended.get("throughput_tps"),
            "level": recommended.get("level"),
            "recorded_at": recommended.get("recorded_at"),
            "engine": engine,
            "placement": placement_block,
            "auxiliary": {
                "id": MAPLE_AUX_ID,
                "source": "shared-main" if shared else "pinned",
                "measured_auxiliary": recommended.get("auxiliary"),
                "engine": aux_engine,
                "artifacts": [
                    {key: item[key] for key in ("role", "path", "present", "reason")}
                    for item in aux_artifacts
                ],
            },
            "artifacts": artifacts,
            "missing": missing,
            "runnable": status == STATUS_READY,
            "summary": message,
        },
        "missing_bytes": _missing_bytes(missing),
        "message": message,
    }


def _overall(entries: Iterable[Mapping[str, Any]]) -> str:
    rows = list(entries)
    ready = [row for row in rows if row.get("status") == STATUS_READY]
    if ready:
        best = ready[0]
        return f"On this machine, run {best['recommendation']['main']} ({best['hardware_tier_gb']} GB tier)."
    for blocker in (STATUS_MEMORY, STATUS_ENGINE, STATUS_MISSING):
        blocked = [row for row in rows if row.get("status") == blocker]
        if blocked:
            return str(blocked[0].get("message") or "")
    if rows:
        return "No measured configuration can run here yet: " + str(rows[0].get("message") or "")
    return "No hardware tier has a measured configuration yet."


def build_fit_view(
    *,
    leaderboard: Mapping[str, Any],
    recipe_book: Any,
    configurations: Iterable[Mapping[str, Any]] = (),
    flags: frozenset[str] | None = None,
    probe_flags: bool = True,
    binary_installed: Callable[[str], bool] | None = None,
    tier_gb: int | None = None,
    model_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """What to run per measured tier, why it cannot run, and what is missing.

    ``flags`` may be injected (tests) or probed from ``/proc/cpuinfo``; passing
    ``probe_flags=False`` reports CPU support as unconfirmed instead of reading
    the host. ``binary_installed`` defaults to a filesystem check of the engine
    binary the recipe would exec.
    """
    if flags is None and probe_flags:
        flags = cpu_flags()
    installed = binary_installed or (lambda path: Path(path).exists())
    model_root = Path(getattr(recipe_book, "model_root", Path.home() / "Models"))
    rows_by_id = {
        str(row.get("id")): row
        for row in configurations
        if isinstance(row, Mapping) and row.get("id")
    }

    entries = [
        _tier_entry(
            tier,
            rows_by_id=rows_by_id,
            recipe_book=recipe_book,
            model_root=model_root,
            flags=flags,
            binary_installed=installed,
            model_names=model_names,
        )
        for tier in leaderboard.get("tiers") or []
        if isinstance(tier, Mapping) and tier.get("hardware_tier_gb") is not None
    ]
    if tier_gb is not None:
        selected = [row for row in entries if int(row["hardware_tier_gb"]) == int(tier_gb)]
        entries = selected or [_unmeasured_entry(int(tier_gb))]
    entries.sort(key=lambda row: int(row["hardware_tier_gb"]))

    return {
        "ok": True,
        "schema": SCHEMA,
        "maple_aux_id": MAPLE_AUX_ID,
        "cpu": _cpu_report(flags),
        "tiers": entries,
        "missing_bytes": sum(int(row.get("missing_bytes") or 0) for row in entries),
        "summary": _overall(entries),
    }


def load_fit_view(
    root: str | Path,
    *,
    tier_gb: int | None = None,
    flags: frozenset[str] | None = None,
    probe_flags: bool = True,
    binary_installed: Callable[[str], bool] | None = None,
    recipe_book: Any = None,
) -> dict[str, Any]:
    """Build the view from the committed references under *root*.

    The dashboard is one caller, but the same four files answer the question for
    any host, so the wiring lives here rather than in the HTTP layer.

    ``flags``, ``probe_flags``, ``binary_installed`` and ``recipe_book`` are
    injection points for offline callers (first-run planning, tests): without
    them every caller that needed a pinned CPU flag set had to re-assemble the
    view by hand and could drift from the dashboard's answer.
    """
    from .model_catalog import ModelCatalog
    from .recipes import RecipeBook

    base = Path(root) / "references"
    leaderboard = json.loads((base / "intelligence-leaderboard.json").read_text(encoding="utf-8"))
    configurations = json.loads((base / "configuration-matrix.json").read_text(encoding="utf-8"))["rows"]
    catalog = ModelCatalog.load(base / "model-catalog.json")
    names: dict[str, str] = {}
    for item in catalog.models:
        names[item.id] = item.name
        if item.family:
            names.setdefault(item.family, item.name)
    return build_fit_view(
        leaderboard=leaderboard,
        recipe_book=recipe_book or RecipeBook.load(base / "model-recipes.json"),
        configurations=configurations,
        flags=flags,
        probe_flags=probe_flags,
        binary_installed=binary_installed,
        tier_gb=tier_gb,
        model_names=names,
    )


__all__ = [
    "MAPLE_AUX_ID",
    "SCHEMA",
    "STATUS_ENGINE",
    "STATUS_MEMORY",
    "STATUS_MISSING",
    "STATUS_READY",
    "STATUS_UNMEASURED",
    "STATUS_UNRESOLVABLE",
    "build_fit_view",
    "component_artifacts",
    "engine_label",
    "load_fit_view",
]
