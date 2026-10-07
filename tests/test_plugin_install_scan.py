"""Regression guard for patterns that block ``hermes plugins install``.

Hermes Agent scans a plugin tree before install. A *critical* finding makes the
verdict ``dangerous``, which ``--force`` cannot override (``tools/plugin_guard.py``
and ``tools/skills_guard.py`` in hermes-agent, scanner ``plugin-guard-v9``). Two
shell helpers piped a response body into an inline interpreter and tripped:

* ``curl_pipe_python`` -- ``curl`` piped into Python
* ``echo_pipe_exec``   -- ``echo`` piped into a shell or interpreter

Either one alone made this plugin non-installable, so this test fails if they come
back. Hermes only reads files whose extension is in its ``SCANNABLE_EXTENSIONS``
set, so the check mirrors that gate: the extensionless ``scripts/serve`` is
intentionally out of scope, exactly as the scanner treats it.
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

# Mirror the two critical patterns that gated this plugin (plugin-guard-v9).
CRITICAL_PATTERNS = {
    "curl_pipe_python": re.compile(r"curl\s+[^\n]*\|\s*python", re.IGNORECASE),
    "echo_pipe_exec": re.compile(
        r"echo\s+[^\n]*\|\s*(?:(?:bash|sh|zsh|ksh|dash)\b|python|perl|ruby|node)",
        re.IGNORECASE,
    ),
}

EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


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
            for name, pattern in CRITICAL_PATTERNS.items():
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno} [{name}] {line.strip()}")
    assert not offenders, (
        "Hermes' plugin installer blocks these pipe-to-interpreter patterns as "
        "critical supply-chain findings,\nwhich makes the plugin non-installable:\n"
        + "\n".join(offenders)
    )
