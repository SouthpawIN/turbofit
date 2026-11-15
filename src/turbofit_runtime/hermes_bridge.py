"""TurboFit as an extension of Hermes' built-in local runtime.

Hermes ships a managed llama.cpp runtime (``hermes_cli.local_runtime``) that is
already better at several things TurboFit re-implemented: a PM-pinned engine
(never downloaded at boot), one supervised llama-server in router mode with a
touch-generation readiness signal, a context ladder with growth, a hardware
budget with physics checks, presets, endpoint resolution and idle unload.

The rule this module encodes: **adopt the primitive where Hermes has one, and
keep TurboFit's own code only where Hermes has no equivalent.** TurboFit's
remaining contributions are the parts Hermes lacks — external-pressure sensing,
the residency-yield action, the two evidence-gated ladders, and the federated
gateway. Everything else should call into Hermes.

Nothing here imports the runtime at module import time: the distribution ships
to machines with different Hermes versions, and an eager import would turn a
missing submodule into an import error at plugin load. Each primitive resolves
on first use and reports honestly when it is absent.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

PACKAGE = "hermes_cli.local_runtime"

#: primitive name -> (submodule, attribute). Attribute None means "the module".
PRIMITIVES: dict[str, tuple[str, str | None]] = {
    "engine_pins": ("binaries", "installed_engine"),
    "context_ladder": ("context_policy", "ladder"),
    "window_decision": ("context_policy", "initial_window"),
    "launch_plan": ("context_policy", "plan_launch"),
    "launch_args": ("context_policy", "launch_args"),
    "growth": ("growth", "maybe_grow_window"),
    "hardware_budget": ("hardware", "probe_budget"),
    "supervisor": ("supervisor", "LlamaServerSupervisor"),
    "endpoint": ("endpoint", "resolve_llamacpp_endpoint"),
    "recovery": ("recovery", None),
}

#: Primitives that replace a TurboFit implementation, with what it replaces.
REPLACES: dict[str, str] = {
    "engine_pins": "turbofit_runtime.binaries (engine selection and pinning)",
    "context_ladder": "the hand-written context shed steps",
    "window_decision": "the 'does this window fit' arithmetic",
    "launch_plan": "turbofit_runtime.recipes launch composition",
    "launch_args": "per-model llvm flags incl. MTP",
    "growth": "the heal-back-up behaviour at the context level",
    "hardware_budget": "turbofit_runtime.fit_check budget maths",
    "supervisor": "turbofit_runtime.native_backend process lifecycle",
    "endpoint": "the localhost port probe",
    "recovery": "restart-after-crash handling",
}

#: TurboFit's fallback shed steps, used only when Hermes' ladder is unavailable.
_FALLBACK_STEPS = (262_144, 131_072, 65_536, 32_768)


#: Why a primitive could not be resolved, keyed by primitive name.
_REASONS: dict[str, str] = {}

#: Where a Hermes install might keep its source, in preference order.
HERMES_ROOTS = (
    "{HERMES_SOURCE}",
    "{HERMES_HOME}/hermes-agent",
    "~/.hermes/hermes-agent",
    "/opt/hermes",
)

_path_checked = False


def _hermes_roots() -> list[Path]:
    roots: list[Path] = []
    for template in HERMES_ROOTS:
        expanded = template.format(
            HERMES_SOURCE=os.environ.get("HERMES_SOURCE", ""),
            HERMES_HOME=os.environ.get("HERMES_HOME", str(Path.home())),
        )
        if not expanded or expanded.startswith("{"):
            continue
        roots.append(Path(expanded).expanduser())
    return roots


def ensure_hermes_on_path() -> Path | None:
    """Put a Hermes install on this interpreter's path, if one is not already there.

    TurboFit's gateway and controller run as standalone services with their own
    Python; Hermes' runtime is usually installed beside them and simply not on
    that path. Adoption is only real if the import actually resolves, so rather
    than declaring the primitives missing we look for the install.

    Returns the root that provided ``hermes_cli``, or None.
    """
    global _path_checked
    # An editable Hermes install maps hermes_cli only; pm and hermes_platform --
    # which its own submodules import -- stay unmapped, so importing the package
    # succeeding is not the same as the runtime being usable. Put the install root
    # (the parent of the hermes_cli package) on the path either way.
    try:
        package = import_module("hermes_cli")
        root = Path(package.__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.append(str(root))
        _path_checked = True
        return root
    except Exception:  # noqa: BLE001 - any import failure means "not here yet"
        pass
    for root in _hermes_roots():
        if (root / "hermes_cli").is_dir():
            if str(root) not in sys.path:
                sys.path.append(str(root))
            _path_checked = True
            return root
    _path_checked = True
    return None


def _resolve(name: str) -> Any | None:
    """Import one Hermes primitive, or None when this Hermes has no such thing."""
    module_name, attribute = PRIMITIVES[name]
    try:
        module = import_module(f"{PACKAGE}.{module_name}")
    except Exception as exc:  # noqa: BLE001 - a dependency error must not be fatal
        if not _path_checked and ensure_hermes_on_path() is not None:
            try:
                module = import_module(f"{PACKAGE}.{module_name}")
            except Exception as retry_exc:  # noqa: BLE001
                _REASONS[name] = f"{type(retry_exc).__name__}: {retry_exc}"
                return None
        else:
            _REASONS[name] = f"{type(exc).__name__}: {exc}"
            return None
    if attribute is None:
        _REASONS.pop(name, None)
        return module
    value = getattr(module, attribute, None)
    if value is None:
        _REASONS[name] = f"{PACKAGE}.{module_name} has no {attribute}"
    else:
        _REASONS.pop(name, None)
    return value


def available() -> dict[str, bool]:
    """Which Hermes primitives this installation offers."""
    return {name: _resolve(name) is not None for name in PRIMITIVES}


def report() -> dict:
    """What TurboFit would adopt here, and what it keeps owning."""
    found = available()
    adopted = sorted(name for name, present in found.items() if present)
    missing = sorted(name for name, present in found.items() if not present)
    return {
        "hermes_local_runtime": bool(adopted),
        "package": PACKAGE,
        "adopted": adopted,
        "missing": missing,
        "missing_reasons": {name: _REASONS.get(name, "not present") for name in missing},
        "replaces": {name: REPLACES[name] for name in adopted},
        "kept_by_turbofit": [
            "external pressure sensing (games, renders, other processes on the card)",
            "residency yield and restore",
            "the two evidence-gated model ladders",
            "the federated gateway and tailnet model sharing",
            "benchmark campaigns and evidence digests",
        ],
        "summary": f"{len(adopted)} of {len(PRIMITIVES)} primitives adopted from Hermes",
    }


def context_ladder(native_tokens: int) -> list[int]:
    """Context windows to shed through, largest first.

    Hermes' ladder is the authority when it exists: it already encodes the 64K
    floor and the growth ratio. TurboFit's static steps are the fallback.
    """
    ladder = _resolve("context_ladder")
    if ladder is not None:
        try:
            steps = [int(step) for step in ladder(int(native_tokens))]
            if steps:
                return sorted(set(steps), reverse=True)
        except (TypeError, ValueError):
            pass
    native = int(native_tokens)
    return [step for step in _FALLBACK_STEPS if step <= native] or [native]


def probe_budget() -> Any | None:
    """The live hardware budget, when Hermes can measure it."""
    probe = _resolve("hardware_budget")
    if probe is None:
        return None
    try:
        return probe()
    except Exception:  # a broken probe must not take the gateway down
        return None


def supervisor(binary: str, models_dir: str, **kwargs: Any) -> Any | None:
    """Hermes' llama-server supervisor, when this install has it.

    The caller keeps ownership of *what* to run (TurboFit decides the rung);
    the supervisor owns *how* one llama-server lives (spawn, readiness, idle
    unload, teardown). None means fall back to TurboFit's native backend.
    """
    cls = _resolve("supervisor")
    if cls is None:
        return None
    try:
        return cls(binary, models_dir, **kwargs)
    except TypeError:
        return None


def unload(supervisor_instance: Any, model_id: str) -> bool:
    """Ask Hermes' supervisor to unload one model; False when it cannot."""
    unload_model: Callable[[str], Any] | None = getattr(
        supervisor_instance, "unload_model", None
    )
    if unload_model is None:
        return False
    try:
        unload_model(model_id)
        return True
    except Exception:
        return False


def idle_release(supervisor_instance: Any) -> list[str]:
    """Reclaim VRAM from models that went quiet (Hermes sweeps, TurboFit yields)."""
    sweep = getattr(supervisor_instance, "sweep_idle", None)
    if sweep is None:
        return []
    try:
        return list(sweep())
    except Exception:
        return []


__all__ = [
    "HERMES_ROOTS",
    "PACKAGE",
    "PRIMITIVES",
    "REPLACES",
    "available",
    "ensure_hermes_on_path",
    "context_ladder",
    "idle_release",
    "probe_budget",
    "report",
    "supervisor",
    "unload",
]
