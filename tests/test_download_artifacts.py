from __future__ import annotations

import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/download-artifacts"


def load_script():
    loader = SourceFileLoader("download_artifacts", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_artifacts_filters_complete_family() -> None:
    module = load_script()
    payload = __import__("json").loads((ROOT / "references/artifact-manifest.json").read_text())

    selected = module.selected_artifacts(payload, {"bonsai-27b"})

    assert {Path(row["destination"]).name for row in selected} == {
        "Bonsai-27B-Q1_0.gguf",
        "Bonsai-27B-mmproj-Q8_0.gguf",
    }


def test_apple_mlx_8bit_family_is_a_complete_pinned_snapshot() -> None:
    module = load_script()
    payload = __import__("json").loads((ROOT / "references/artifact-manifest.json").read_text())

    selected = module.selected_artifacts(
        payload, {"qwen3-8-27b-uncensored-mlx-8bit"}
    )

    assert {row["path"] for row in selected} == {
        "8-bit/README.md",
        "8-bit/chat_template.jinja",
        "8-bit/config.json",
        "8-bit/generation_config.json",
        "8-bit/model-00001-of-00006.safetensors",
        "8-bit/model-00002-of-00006.safetensors",
        "8-bit/model-00003-of-00006.safetensors",
        "8-bit/model-00004-of-00006.safetensors",
        "8-bit/model-00005-of-00006.safetensors",
        "8-bit/model-00006-of-00006.safetensors",
        "8-bit/model.safetensors.index.json",
        "8-bit/preprocessor_config.json",
        "8-bit/processor_config.json",
        "8-bit/tokenizer.json",
        "8-bit/tokenizer_config.json",
        "8-bit/video_preprocessor_config.json",
        "8-bit/vocab.json",
    }
    assert {
        row["revision"] for row in selected
    } == {"b4603df5fd2a51e7fed2560ee7090caa4e13e4b7"}
    assert all(
        row["destination"].startswith("Qwen3.8-27B-Uncensored-MLX/8-bit/")
        for row in selected
    )


def test_install_artifact_verifies_before_materializing(tmp_path: Path) -> None:
    module = load_script()
    cached = tmp_path / "cache" / "model.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"pinned-model")
    item = {
        "destination": "family/model.gguf",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "path": "remote/model.gguf",
        "size_bytes": cached.stat().st_size,
        "sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
    }

    result = module.install_artifact(
        item,
        root=tmp_path / "models",
        download_fn=lambda **_: str(cached),
    )

    assert result["downloaded"] is True
    assert (tmp_path / "models/family/model.gguf").read_bytes() == b"pinned-model"


def test_install_artifact_materializes_cached_symlink_target_not_the_symlink(tmp_path: Path) -> None:
    module = load_script()
    blob = tmp_path / "cache/blobs/model"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"pinned-model")
    cached = tmp_path / "cache/snapshots/model.gguf"
    cached.parent.mkdir()
    cached.symlink_to(blob)
    item = {
        "destination": "family/model.gguf",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "path": "remote/model.gguf",
        "size_bytes": blob.stat().st_size,
        "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
    }

    module.install_artifact(item, root=tmp_path / "models", download_fn=lambda **_: str(cached))

    destination = tmp_path / "models/family/model.gguf"
    assert destination.is_file()
    assert not destination.is_symlink()
    assert destination.read_bytes() == b"pinned-model"


def test_install_artifact_accepts_a_valid_destination_created_during_link(tmp_path: Path, monkeypatch) -> None:
    module = load_script()
    cached = tmp_path / "cache" / "model.gguf"
    cached.parent.mkdir()
    cached.write_bytes(b"pinned-model")
    item = {
        "destination": "family/model.gguf",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "path": "remote/model.gguf",
        "size_bytes": cached.stat().st_size,
        "sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
    }
    destination = tmp_path / "models" / item["destination"]
    real_link = os.link

    def link_after_race(source, target):
        destination.parent.mkdir(parents=True, exist_ok=True)
        real_link(source, target)
        raise FileExistsError("another installer already materialized this artifact")

    monkeypatch.setattr(module.os, "link", link_after_race)

    result = module.install_artifact(item, root=tmp_path / "models", download_fn=lambda **_: str(cached))

    assert result == {"destination": str(destination), "downloaded": False, "verified": True}


def test_destination_cannot_escape_model_root(tmp_path: Path) -> None:
    module = load_script()

    with pytest.raises(ValueError, match="escapes model root"):
        module.safe_destination(tmp_path / "models", "../outside")


def test_destination_allows_content_addressed_blob_symlink(tmp_path: Path) -> None:
    module = load_script()
    root = tmp_path / "storage/models"
    blobs = tmp_path / "storage/blobs"
    root.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "abc123"
    blob.write_bytes(b"model")
    link = root / "model.gguf"
    link.symlink_to(blob)

    assert module.safe_destination(root, "model.gguf") == link


def test_destination_rejects_symlink_outside_blob_store(tmp_path: Path) -> None:
    module = load_script()
    root = tmp_path / "storage/models"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"not a model")
    (root / "model.gguf").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink escapes blob store"):
        module.safe_destination(root, "model.gguf")
