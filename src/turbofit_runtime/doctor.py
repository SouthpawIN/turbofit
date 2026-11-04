"""Resilience checks that hold on any install, not just the one they were written on.

Every defect fixed on omarchy this week had the same shape: TurboFit depended on a
fact about *that* machine (Hermes on the interpreter's path, a unit that happened to
point at the right script, a health endpoint that reported "ready" while a hosted
API answered). The checks here are written so a fresh install on Windows, a Mac, a
VPS or someone else's desktop can detect the same class of problem itself.

This module is pure logic: it takes observations and returns findings plus the
repairs that are safe to apply. ``scripts/turbofit-doctor`` does the I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OK = "ok"
WARN = "warn"
FAIL = "fail"

#: A health payload is lying when it promises local service while an API answers.
_LOCAL_CLAIM_KEYS = ("local", "cloud_fallback")


@dataclass
class Finding:
    name: str
    status: str
    detail: str
    repair: str | None = None  # what --fix would do, when it is safe to do it

    def to_mapping(self) -> dict:
        return {
            "check": self.name,
            "status": self.status,
            "detail": self.detail,
            "repair": self.repair,
        }


@dataclass
class Diagnosis:
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> str:
        for level in (FAIL, WARN, OK):
            if any(finding.status == level for finding in self.findings):
                return level
        return OK

    def repairs(self) -> list[Finding]:
        return [f for f in self.findings if f.repair]

    def to_mapping(self) -> dict:
        return {
            "status": self.worst,
            "findings": [finding.to_mapping() for finding in self.findings],
            "repairs_available": [finding.name for finding in self.repairs()],
        }


def classify_health(payload: dict | None) -> Finding:
    """Is the gateway telling the truth about who is answering?"""
    if not isinstance(payload, dict):
        return Finding(
            "health",
            FAIL,
            "the gateway did not answer /health — it is not running or not binding",
            repair="restart",
        )
    if not payload.get("ok"):
        return Finding("health", FAIL, f"gateway reports not-ok: {payload.get('main')!r}")
    entry = payload.get("main")
    if any(key in payload for key in _LOCAL_CLAIM_KEYS):
        if payload.get("local") is False and entry in {"ready", "loaded"}:
            # The old lie: "ready" with a hosted backend behind it.
            return Finding(
                "health",
                FAIL,
                "health says the local model is ready while an API is answering",
            )
        if payload.get("cloud_fallback") and entry == "ready":
            return Finding(
                "health",
                WARN,
                "a hosted API is answering while the main route claims ready",
            )
        if payload.get("main") == "yielded":
            return Finding(
                "health",
                OK,
                "local residency is yielded to external GPU pressure; a hosted API is "
                "answering by explicit fallback",
            )
    return Finding("health", OK, f"main={entry} local={payload.get('local')} "
                                 f"backend={payload.get('backend')}")


_UNIT_EXEC = re.compile(r"^ExecStart=(.+)$", re.MULTILINE)


def unit_exec_line(unit_text: str) -> str | None:
    match = _UNIT_EXEC.search(unit_text or "")
    return match.group(1).strip() if match else None


def unit_needs_runner(unit_text: str) -> bool:
    """True when a unit launches a script directly instead of through turbofit-run.

    Launching directly is what made adoption depend on the unit's Python: the
    gateway could not import Hermes' runtime even with Hermes installed beside it.
    """
    line = unit_exec_line(unit_text)
    if not line:
        return False
    return "turbofit-run" not in line


def rewrite_unit(unit_text: str, runner_path: str) -> str:
    """Point an existing ExecStart at turbofit-run, keeping its arguments."""

    def _replace(match: re.Match[str]) -> str:
        command = match.group(1).strip()
        if "turbofit-run" in command:
            return match.group(0)
        return f"ExecStart={runner_path} {command}"

    return _UNIT_EXEC.sub(_replace, unit_text or "", count=1)


def diagnose_interpreter(*, python: str | None, has_psutil: bool, has_hermes: bool,
                         adopted: int, total: int) -> Finding:
    """Adoption broken by the interpreter is the most common install failure."""
    if not python:
        return Finding("interpreter", FAIL, "no usable python interpreter found")
    if has_hermes and not has_psutil:
        return Finding(
            "interpreter",
            WARN,
            f"{python} has Hermes but not psutil, so supervisor/recovery cannot be "
            f"adopted ({adopted}/{total} available)",
            repair="python",
        )
    if not has_hermes:
        return Finding(
            "interpreter",
            WARN,
            f"{python} cannot import hermes_cli; TurboFit runs standalone "
            f"({adopted}/{total} primitives)",
            repair="python",
        )
    ratio = f"{adopted}/{total}"
    status = OK if adopted == total else WARN
    return Finding("interpreter", status, f"{python} adopts {ratio} Hermes primitives")


def diagnose_runtime_state(*, declared: dict | None, listening: bool,
                           released: bool) -> Finding:
    """A declared local route that is not listening is either yielded or dead."""
    if not declared:
        return Finding("runtime_state", WARN,
                       "no local route published yet; the controller has not selected a rung")
    rung = declared.get("rung_id") or "unknown"
    port = declared.get("port")
    if listening:
        return Finding("runtime_state", OK, f"rung {rung} is serving on port {port}")
    if released:
        return Finding(
            "runtime_state",
            OK,
            f"rung {rung} is released on purpose (external GPU pressure)",
        )
    return Finding(
        "runtime_state",
        FAIL,
        f"rung {rung} is declared but nothing is listening on port {port}",
        repair="restart",
    )


def diagnose_strata(*, installed: str | None, problems: list[str]) -> Finding:
    if installed is None:
        return Finding("strata", OK, "Strata is not installed here; nothing to check")
    if problems:
        return Finding("strata", WARN, "; ".join(problems))
    return Finding("strata", OK, installed)


def diagnose_engine_pins(*, expected: str | None, present: list[str]) -> Finding:
    """An engine pinned by one install and absent on another is a silent downgrade."""
    if expected and expected not in present:
        return Finding(
            "engine_pins",
            WARN,
            f"pinned engine {expected} is not in the local store "
            f"({len(present)} engines present)",
        )
    return Finding("engine_pins", OK, f"{len(present)} engine(s) in the local store")


def summarise(diagnosis: Diagnosis) -> str:
    counts = {level: sum(1 for f in diagnosis.findings if f.status == level)
              for level in (FAIL, WARN, OK)}
    return (
        f"{counts[FAIL]} failing, {counts[WARN]} warning, {counts[OK]} ok"
    )


__all__ = [
    "Diagnosis",
    "FAIL",
    "Finding",
    "OK",
    "WARN",
    "classify_health",
    "diagnose_engine_pins",
    "diagnose_interpreter",
    "diagnose_runtime_state",
    "diagnose_strata",
    "rewrite_unit",
    "summarise",
    "unit_exec_line",
    "unit_needs_runner",
]
