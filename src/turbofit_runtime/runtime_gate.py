"""Row-level engine-binary gate for the catalog campaign.

A recipe names an engine binary (``prism-llama.cpp``'s ``llama-server``, a vLLM
entry point, the Strata CLI). When that build has not been installed yet —
``scripts/install.sh`` has not run, or a rebuild is in progress and the output
directory is momentarily empty — :class:`turbofit_runtime.backend.CampaignBackend`
correctly refuses to start it, but a campaign that ran the row first recorded a
prerequisite the user can fix in one command as a model failure.

This module moves that judgement to row selection, the same way
:mod:`turbofit_runtime.artifact_gate` does for weights and
:mod:`turbofit_runtime.offload_gate` does for memory. The three verdicts answer
three different user questions:

* ``waiting-download`` — the weights are not here yet.
* ``waiting-memory`` — the weights are here and this host cannot hold them.
* ``waiting-runtime`` — the weights and the host are fine; the engine binary
  for this family is not installed.

``waiting-runtime`` is the catalog-campaign spelling of the intelligence
campaign's existing vocabulary (``ready`` → ``waiting-runtime`` →
``blocked-runtime``), not a fourth name: the row is waiting for a runtime that
is not present, and it becomes runnable as soon as the install finishes.

The launch paths keep their own refusal. This gate only decides selection; a
row it lists is still refused by ``CampaignBackend.start`` if it is launched
directly (``run-id`` on a hand-built row), with the original message intact.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

STATUS = "waiting-runtime"

#: Same instruction the backend's launch-time error gives, so the row reason
#: and a hand-launched failure tell the user to do the same one thing.
INSTALL_HINT = (
    "Install the runtime (scripts/install.sh) or set the engine path in "
    "references/model-recipes.json for this family."
)


@dataclass(frozen=True)
class WaitingRuntime:
    """A row whose recipe names an engine binary this host does not have."""

    row_id: str
    reason: str
    binary: str
    family: str = ""
    install_hint: str = INSTALL_HINT

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "status": STATUS,
            "reason": self.reason,
            "binary": self.binary,
            "family": self.family,
            "install_hint": self.install_hint,
        }


def engine_binary(component: Any) -> str:
    """The binary this component's command would exec, or ``""``."""
    command = getattr(component, "command", ()) or ()
    return str(command[0]) if command else ""


def _looks_like_path(binary: str) -> bool:
    return (
        os.sep in binary
        or bool(os.altsep and os.altsep in binary)
        or binary.startswith("~")
    )


def engine_binary_installed(
    binary: str, *, which: Callable[[str], str | None] = shutil.which
) -> bool:
    """True when *binary* is an existing file or resolvable on ``PATH``.

    Absolute recipe paths (``~/.local/share/turbofit/runtimes/.../bin/llama-server``)
    are the native builders' output; a bare name (``vllm``) is resolved the way
    ``exec`` would resolve it. A directory is not a binary.
    """
    if not binary:
        return False
    if _looks_like_path(binary):
        return Path(binary).expanduser().is_file()
    return which(binary) is not None


def describe_missing_runtime(component: Any, binary: str) -> str:
    """The row reason: which binary, and the one command that installs it."""
    family = str(getattr(component, "family", "") or getattr(component, "alias", "") or "")
    return (
        f"{STATUS}: engine binary not found for {family}: {binary!r}. {INSTALL_HINT}"
    )


def waiting_runtime_rows(
    configuration_index: Mapping[str, dict[str, Any]],
    recipes: Any,
    *,
    skip: Iterable[str] = (),
) -> dict[str, WaitingRuntime]:
    """Rows whose resolved recipe names an engine binary absent from this host.

    Resolution failures are skipped: an unresolvable recipe is not proof of a
    missing engine and is reported by the row's own resolution path.

    *skip* is the set of rows another gate already claimed (weights missing,
    or measured as too large). Each row gets exactly one verdict so the status
    counts partition the matrix instead of double-counting a row.
    """
    excluded = set(skip)
    result: dict[str, WaitingRuntime] = {}
    for row_id, payload in configuration_index.items():
        if row_id in excluded:
            continue
        try:
            recipe = recipes.resolve_catalog_configuration(dict(payload))
        except Exception:  # noqa: BLE001 - an unresolved recipe is not this gate's verdict
            continue
        missing = next(
            (
                component
                for component in recipe.components
                if not engine_binary_installed(engine_binary(component))
            ),
            None,
        )
        if missing is None:
            continue
        binary = engine_binary(missing)
        result[row_id] = WaitingRuntime(
            row_id=row_id,
            reason=describe_missing_runtime(missing, binary),
            binary=binary,
            family=str(getattr(missing, "family", "") or ""),
        )
    return result


__all__ = [
    "INSTALL_HINT",
    "STATUS",
    "WaitingRuntime",
    "describe_missing_runtime",
    "engine_binary",
    "engine_binary_installed",
    "waiting_runtime_rows",
]
