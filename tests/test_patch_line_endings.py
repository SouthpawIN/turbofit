"""Keep ``git apply`` working on the runtime patches.

The runtime installer applies ``runtime-patches/**/*.patch`` with ``git apply``,
which rejects a CRLF patch as corrupt before it even looks at a hunk
("corrupt patch at line ..."). Git for Windows ships with ``core.autocrlf=true``
by default, so a plain checkout used to rewrite these patches to CRLF and the
documented one-shot Windows install failed.

Two layers keep this fixed:

* ``.gitattributes`` pins ``*.patch`` to LF so checkouts cannot corrupt them, and
* ``install-dspark-runtime`` normalizes the bytes it feeds to ``git apply``, so
  even an already-corrupted checkout applies cleanly.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "scripts" / "install-dspark-runtime"
PATCHES = sorted((ROOT / "runtime-patches").rglob("*.patch"))


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "install_dspark_runtime_eol",
        INSTALLER,
        loader=SourceFileLoader("install_dspark_runtime_eol", str(INSTALLER)),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_patches_exist() -> None:
    assert PATCHES, "no runtime patches found"


def test_runtime_patches_are_lf_only() -> None:
    crlf = [str(path.relative_to(ROOT)) for path in PATCHES if b"\r" in path.read_bytes()]
    assert not crlf, f"runtime patches must stay LF-only, found CR in: {crlf}"


def test_gitattributes_keeps_patch_checkouts_lf() -> None:
    rules = [
        line.split("#", 1)[0].strip()
        for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    ]
    assert any(rule.startswith("*.patch") and "eol=lf" in rule for rule in rules), (
        "*.patch must be pinned to eol=lf so core.autocrlf cannot rewrite patches"
    )


def test_installer_normalizes_crlf_patch_bytes(tmp_path) -> None:
    module = _load_installer()
    patch = tmp_path / "crlf.patch"
    patch.write_bytes(b"diff --git a/f b/f\r\n--- a/f\r\n+++ b/f\r\n@@ -1 +1 @@\r\n-a\r\n+b\r\n")
    normalized = module.normalized_patch_bytes(patch)
    assert b"\r" not in normalized
    assert normalized == b"diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n"


def test_installer_feeds_lf_bytes_to_git_apply(monkeypatch, tmp_path) -> None:
    module = _load_installer()
    patch = tmp_path / "crlf.patch"
    patch.write_bytes(b"--- a/f\r\n+++ b/f\r\n")
    monkeypatch.setattr(module, "PATCHES", (patch,))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs.get("input")))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.apply_patches(tmp_path)

    assert [command for command, _ in calls] == [
        ["git", "-C", str(tmp_path), "apply", "--check", "-"],
        ["git", "-C", str(tmp_path), "apply", "-"],
    ]
    assert all(data is not None and b"\r" not in data for _, data in calls)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_apply_parses_every_runtime_patch() -> None:
    for patch in PATCHES:
        result = subprocess.run(
            ["git", "apply", "--stat", str(patch)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{patch.relative_to(ROOT)} failed to parse: {result.stderr.strip()}"
