"""Rows whose weights are not installed wait instead of burning attempts."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from turbofit_runtime.artifact_gate import (
    artifact_problems,
    waiting_download_rows,
)

ROOT = Path(__file__).parents[1]

#: Real manifest rows, used so the test exercises the shipped size pinning.
BONSAI_F16 = ("Bonsai-27B/Bonsai-27B-F16.gguf", 53808280640)
INKLING_SNAPSHOT = ("Inkling-2-Small", 393489000000)


def sparse(path: Path, size: int) -> Path:
    """A file whose stat size is *size* without writing the bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


def component(role: str, model: Path) -> SimpleNamespace:
    return SimpleNamespace(
        role=role, model_path=str(model), projector_path="", draft_path="",
        command=("engine", "-m", str(model)),
    )


def fake_recipes(model: Path, *, model_root: Path) -> SimpleNamespace:
    def resolve(row: dict) -> SimpleNamespace:
        return SimpleNamespace(components=(component("main", model),))

    return SimpleNamespace(model_root=model_root, resolve_catalog_configuration=resolve)


def test_truncated_model_is_waiting_download_with_missing_bytes(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    relative, expected = BONSAI_F16
    model = sparse(model_root / relative, 30 * 1024**3)
    rows = {"row-a": {"id": "row-a", "main": "m", "auxiliary": "auto", "context": 65536}}

    waiting = waiting_download_rows(rows, fake_recipes(model, model_root=model_root))

    assert set(waiting) == {"row-a"}
    entry = waiting["row-a"]
    assert entry.reason.startswith("waiting-download:")
    assert entry.missing_bytes == expected - 30 * 1024**3
    assert "missing" in entry.reason
    assert entry.to_dict()["status"] == "waiting-download"


def test_complete_model_is_runnable(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    relative, expected = BONSAI_F16
    model = sparse(model_root / relative, expected)
    rows = {"row-a": {"id": "row-a", "main": "m", "auxiliary": "auto", "context": 65536}}

    waiting = waiting_download_rows(rows, fake_recipes(model, model_root=model_root))

    assert waiting == {}


def test_sub_mebibyte_stub_is_waiting_download(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    relative, expected = INKLING_SNAPSHOT
    stub = model_root / relative
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_bytes(b"<html>404 not found</html>")
    rows = {"row-b": {"id": "row-b", "main": "m", "auxiliary": "auto", "context": 65536}}

    waiting = waiting_download_rows(rows, fake_recipes(stub, model_root=model_root))

    entry = waiting["row-b"]
    assert "stub" in entry.reason or "truncated" in entry.reason
    assert entry.missing_bytes == expected - stub.stat().st_size


def test_missing_file_is_waiting_download(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    missing = model_root / "nowhere" / "weights.gguf"
    rows = {"row-c": {"id": "row-c", "main": "m", "auxiliary": "auto", "context": 65536}}

    waiting = waiting_download_rows(rows, fake_recipes(missing, model_root=model_root))

    assert waiting["row-c"].missing_bytes == 0
    assert "missing" in waiting["row-c"].reason


def test_empty_checkpoint_directory_is_waiting_download(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    problems = artifact_problems([component("main", snapshot)], model_root=tmp_path)

    assert problems and "empty" in problems[0].reason


def test_sharded_model_waits_for_the_missing_shard(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    first = model_root / "MiniMax-M3" / "MiniMax-M3-UD-Q4_K_M-00001-of-00007.gguf"
    sparse(first, 4096)
    problems = artifact_problems([component("main", first)], model_root=model_root)

    missing = {Path(item.path).name for item in problems}
    assert "MiniMax-M3-UD-Q4_K_M-00007-of-00007.gguf" in missing


def test_unresolvable_recipe_is_not_gated(tmp_path: Path) -> None:
    class Broken:
        model_root = tmp_path

        def resolve_catalog_configuration(self, row: dict):
            raise ValueError("unknown model family")

    assert waiting_download_rows({"row-d": {"id": "row-d"}}, Broken()) == {}
