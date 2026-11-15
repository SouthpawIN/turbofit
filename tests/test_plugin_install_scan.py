"""Regression guard for patterns that block ``hermes plugins install``.

Hermes Agent scans a plugin tree before install. A *critical* finding makes the
verdict ``dangerous``, which ``--force`` cannot override (``tools/plugin_guard.py``
and ``tools/skills_guard.py`` in hermes-agent, scanner ``plugin-guard-v11``).

Test A — ``test_no_critical_pipe_to_interpreter_patterns``: guards the two shell
helpers that first tripped the install (``curl`` piped into Python / ``echo``
piped to a shell or interpreter). These are the exact supply-chain patterns
from ``tools/skills_guard.THREAT_PATTERNS`` that made this plugin
non-installable.

Test B — ``test_no_critical_security_findings``: scans the same file set with the
full critical pattern table from ``tools/skills_guard.THREAT_PATTERNS``. If a
critical pattern fires in plugin code or docs, the plugin installs as
``dangerous`` and ``--force`` cannot override it.

In both cases, Hermes only reads files whose extension is in its
``SCANNABLE_EXTENSIONS`` set, so the check mirrors that gate: the extensionless
``scripts/serve`` is intentionally out of scope, exactly as the scanner treats
it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]

# Mirror hermes-agent tools/skills_guard.py::SCANNABLE_EXTENSIONS.
SCANNABLE_EXTENSIONS = {
    ".md", ".txt", ".py", ".sh", ".bash", ".js", ".ts", ".rb", ".yaml", ".yml",
    ".json", ".toml", ".cfg", ".ini", ".conf", ".html", ".css", ".xml", ".tex",
    ".r", ".jl", ".pl", ".php",
}

EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}

# Mirror the two critical patterns from hermes-agent tools/skills_guard.py.
# Both require a pipe (|) between the command and the interpreter, so plain
# prose like "Run python3 ..." or "echo 'some python snippet'" is excluded.
_SHELL_NAMES = r"(?:bash|sh|zsh|ksh|dash)\b"
PIPE_CRITICAL_PATTERNS = {
    "curl_pipe_python": re.compile(
        rf"curl\s+[^|\s][^\n]*\|\s*python", re.IGNORECASE
    ),
    "echo_pipe_exec": re.compile(
        rf"echo\s+[^\n]*\|\s*(?:{_SHELL_NAMES}|python|perl|ruby|node)",
        re.IGNORECASE,
    ),
}


def _scannable_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in SCANNABLE_EXTENSIONS:
            files.append(path)
    return sorted(files)


def test_no_critical_pipe_to_interpreter_patterns() -> None:
    offenders = []
    for path in _scannable_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PIPE_CRITICAL_PATTERNS.items():
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} [{name}] {line.strip()}")
    assert not offenders, (
        "Hermes' plugin installer blocks these pipe-to-interpreter patterns as "
        "critical supply-chain findings, "
        "which makes the plugin non-installable:\n"
        + "\n".join(offenders)
    )


# ── Full critical-pattern table from the installed hermes-agent ─────────────────
try:
    import sys as _sys
    _sys.path.insert(
        0,
        "/Users/sovthpaw/.hermes/installs/ea68b0fc10a2df6f/environments/"
        "64a2ce56a23742c39be09c7165b5f3a2/workspace",
    )
    from tools.skills_guard import THREAT_PATTERNS, SCANNABLE_EXTENSIONS as _H_SCANNABLE

    _ALL_CRITICAL = [
        (re.compile(pattern, re.IGNORECASE), pid)
        for pattern, pid, sev, cat, desc in THREAT_PATTERNS
        if sev == "critical"
    ]
    _H_EXCLUDED = {".git", "__pycache__", "node_modules", ".venv", "venv"}
except Exception:  # pragma: no cover - fallback if hermes-agent path changes
    _ALL_CRITICAL = []
    _H_SCANNABLE = set()
    _H_EXCLUDED = set()


def test_no_critical_security_findings() -> None:
    if not _ALL_CRITICAL:
        return  # hermes-agent not installed in this environment; skip
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in _H_EXCLUDED for part in rel.parts):
            continue
        if path.suffix.lower() not in _H_SCANNABLE:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        lines = text.split("\n")
        for pattern, pid in _ALL_CRITICAL:
            for i, line in enumerate(lines, start=1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i} [{pid}] {line.strip()[:100]}")
    assert not offenders, (
        "The hermes-agent plugin scanner flags these as critical findings, "
        "which makes the plugin non-installable as a community source:\n"
        + "\n".join(offenders)
    )
