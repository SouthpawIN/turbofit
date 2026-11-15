"""The Maple 3:1 SWA fallback: derivation logic plus patch integrity.

Maple's GGUF conversions shipped without ``maple.attention.sliding_window_pattern``
and newer engines read that array with ``required=true``, so those checkpoints
die with ``array key not found in model``. The engine patch under
``runtime-patches/maple-llama.cpp/`` derives ``set_swa_pattern(4, false)`` and
lets an explicit array override it. These tests cover the rule (pure logic) and
that the shipped patch still applies to the hunk it was written against.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from turbofit_runtime import maple_swa

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runtime-patches/maple-llama.cpp/derive-sliding-window-pattern.patch"
INSTALLER = ROOT / "scripts/install-native-runtimes"


def _installer():
    loader = SourceFileLoader("install_native_runtimes_maple", str(INSTALLER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# derivation
# --------------------------------------------------------------------------- #
def test_derived_pattern_is_sssf_repeated_for_24_layers() -> None:
    pattern = maple_swa.sliding_window_pattern()

    assert len(pattern) == 24
    assert pattern[:8] == [1, 1, 1, 0, 1, 1, 1, 0]
    assert sum(1 for value in pattern if value == 0) == 6  # full-attention layers
    assert sum(pattern) == 18                              # sliding-window layers
    assert maple_swa.describe(pattern) == "SSSF" * 6


def test_derivation_matches_published_config_layer_types() -> None:
    """config.json: layer_types is sliding_attention x3 then full_attention, x6."""
    layer_types = ["sliding_attention"] * 3 + ["full_attention"]  # SSSF
    expected = [1 if kind == "sliding_attention" else 0 for kind in layer_types * 6]

    assert maple_swa.sliding_window_pattern(n_layers=24) == expected


def test_full_attention_lands_on_every_fourth_layer() -> None:
    pattern = maple_swa.sliding_window_pattern(n_layers=24)

    assert [i for i, value in enumerate(pattern) if value == 0] == [3, 7, 11, 15, 19, 23]


def test_derivation_scales_to_other_layer_counts() -> None:
    assert maple_swa.sliding_window_pattern(n_layers=4) == [1, 1, 1, 0]
    assert maple_swa.sliding_window_pattern(n_layers=6) == [1, 1, 1, 0, 1, 1]
    assert maple_swa.sliding_window_pattern(n_layers=8, period=2) == [1, 0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("bad", [0, -1])
def test_derivation_rejects_nonsense(bad: int) -> None:
    with pytest.raises(ValueError):
        maple_swa.sliding_window_pattern(n_layers=bad)
    with pytest.raises(ValueError):
        maple_swa.sliding_window_pattern(period=bad)


# --------------------------------------------------------------------------- #
# fallback decision
# --------------------------------------------------------------------------- #
def test_missing_key_falls_back_to_derived_pattern() -> None:
    keys = {"maple.block_count", "maple.attention.sliding_window"}

    pattern = maple_swa.resolve_pattern("maple", keys, n_layers=24)

    assert pattern == maple_swa.sliding_window_pattern(24)


def test_present_key_is_left_to_the_loader() -> None:
    keys = {"maple.attention.sliding_window_pattern"}

    assert maple_swa.resolve_pattern("maple", keys, n_layers=24) is None


def test_explicit_pattern_wins_over_derivation() -> None:
    explicit = [1] * 24

    pattern = maple_swa.resolve_pattern(
        "maple", set(), n_layers=24, explicit_pattern=explicit
    )

    assert pattern == explicit


def test_other_architectures_are_not_our_decision() -> None:
    assert maple_swa.resolve_pattern("gemma3", set()) is None
    assert maple_swa.resolve_pattern("llama", set()) is None


# --------------------------------------------------------------------------- #
# the shipped patch
# --------------------------------------------------------------------------- #
def _patch_lines() -> list[str]:
    return PATCH.read_text(encoding="utf-8").splitlines()


def _hunk_old_side() -> tuple[int, list[str]]:
    """Rebuild the pre-patch file body for the first hunk, from the patch itself."""
    lines = _patch_lines()
    header = next(line for line in lines if line.startswith("@@"))
    match = re.match(r"@@ -(\d+)", header)
    assert match, f"no hunk header in {PATCH}"
    start = int(match.group(1))
    body = []
    for line in lines:
        if line.startswith(("@@", "---", "+++", "diff ", "index ", "+")):
            continue
        body.append(line[1:] if line.startswith((" ", "-")) else line)
    return start, body


def _seed_checkout(tmp_path: Path) -> Path:
    """Lay out a git repo with ``src/models/maple.cpp`` at the patch's target lines."""
    start, body = _hunk_old_side()
    target = tmp_path / "src" / "models" / "maple.cpp"
    target.parent.mkdir(parents=True)
    filler = [f"// filler {i}" for i in range(start - 1)]
    target.write_text("\n".join(filler + body) + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return target


def test_patch_targets_the_maple_hparams_loader() -> None:
    text = PATCH.read_text(encoding="utf-8")

    assert "a/src/models/maple.cpp" in text
    # the patch must keep the derived default, not add a second set_swa_pattern call
    assert text.count("+    hparams.set_swa_pattern(4, false);") == 0
    assert text.count("     hparams.set_swa_pattern(4, false);") == 1
    assert "LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN" in text
    assert re.search(r"hparams\.n_layer\(\), false\);", text), "the pattern read must be optional"


def test_patch_is_lf_only() -> None:
    assert b"\r" not in PATCH.read_bytes()


def test_runtime_patch_is_discovered_for_the_maple_runtime() -> None:
    module = _installer()

    assert module.runtime_patches("maple-llama.cpp") == [PATCH]
    assert module.runtime_patches("mainline-llama.cpp") == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_patch_still_applies_to_the_hunk_it_was_written_against(tmp_path) -> None:
    target = _seed_checkout(tmp_path)

    applied = _installer().apply_patches(tmp_path, [PATCH])

    assert applied == [PATCH.name]
    patched = target.read_text(encoding="utf-8")
    assert "hparams.set_swa_pattern(4, false);" in patched
    assert "LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN" in patched
    assert "array key not found" not in patched


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_apply_patches_is_idempotent(tmp_path) -> None:
    """A repair run over an already-patched checkout must not fail."""
    _seed_checkout(tmp_path)

    module = _installer()
    assert module.apply_patches(tmp_path, [PATCH]) == [PATCH.name]
    assert module.apply_patches(tmp_path, [PATCH]) == []
