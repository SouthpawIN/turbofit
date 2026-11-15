from __future__ import annotations

import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/download-artifacts"
# Real weights are always megabytes; a tiny fixture is now correctly treated as
# a placeholder stub by the artifact verifier.
PINNED = b"pinned-model" + b"\0" * (1024 * 1024)


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
    cached.write_bytes(PINNED)
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
    assert (tmp_path / "models/family/model.gguf").read_bytes() == PINNED


def test_install_artifact_materializes_cached_symlink_target_not_the_symlink(tmp_path: Path) -> None:
    module = load_script()
    blob = tmp_path / "cache/blobs/model"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(PINNED)
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
    assert destination.read_bytes() == PINNED


def test_install_artifact_accepts_a_valid_destination_created_during_link(tmp_path: Path, monkeypatch) -> None:
    module = load_script()
    cached = tmp_path / "cache" / "model.gguf"
    cached.parent.mkdir()
    cached.write_bytes(PINNED)
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


# ---------------------------------------------------------------------------
# 2. Missing shards are work items; families fail independently
# ---------------------------------------------------------------------------

REVISION = "a" * 40
MIN = 1024 * 1024


def shard_payload(index: int) -> bytes:
    body = f"shard-{index}".encode()
    return (body * ((MIN // len(body)) + 2))[: MIN + index * 512]


class FakeTree:
    """Offline stand-in for ArtifactTree; raises when configured to."""

    def __init__(self, listing):
        self.listing = listing
        self.calls: list[tuple[str, str | None]] = []

    def siblings(self, repo_id, revision=None):
        self.calls.append((repo_id, revision))
        if isinstance(self.listing, Exception):
            raise self.listing
        return self.listing


def sharded_row(prefix: str, destination_dir: str = "Sharded", families=("sharded",)) -> dict:
    return {
        "destination": f"{destination_dir}/{prefix}.gguf",
        "families": list(families),
        "repo_id": "owner/sharded",
        "revision": REVISION,
        "path": f"{prefix}.gguf",
        "sha256": "0" * 64,
        "size_bytes": 4,
    }


def sharded_listing(prefix: str, count: int, omit: tuple[int, ...] = (), cache: Path | None = None):
    from turbofit_runtime.artifacts import SiblingInfo

    listing = {}
    for index in range(1, count + 1):
        if index in omit:
            continue
        name = f"{prefix}-{index:05d}-of-{count:05d}.gguf"
        payload = shard_payload(index)
        if cache is not None:
            (cache / name).write_bytes(payload)
        listing[name] = SiblingInfo(
            name, size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    return listing


def test_missing_sibling_becomes_a_download_item(tmp_path: Path) -> None:
    """A shard the listing omits is still fetched, not turned into an exception.

    Upstream listings can be stale; only the fetch proves a shard is gone.
    """
    module = load_script()
    prefix = "Tiny-Set"
    cache = tmp_path / "cache"
    cache.mkdir()
    # The listing omits shard 3, but the file is really available upstream.
    listing = sharded_listing(prefix, 4, omit=(3,), cache=cache)
    (cache / f"{prefix}-00003-of-00004.gguf").write_bytes(shard_payload(3))

    rows = module.expand_rows([sharded_row(prefix)], tree=FakeTree(listing))

    assert len(rows) == 4
    assert rows[0]["path"] == f"{prefix}-00003-of-00004.gguf"
    assert rows[0]["unlisted_shard"] is True
    assert rows[0]["size_bytes"] is None
    assert "sha256" not in rows[0]

    summary = module.download_families(
        [sharded_row(prefix)],
        root=tmp_path / "models",
        tree=FakeTree(listing),
        download_fn=lambda *, filename, **_: str(cache / filename),
        err=__import__("io").StringIO(),
    )

    assert summary["families_failed"] == 0
    landed = tmp_path / "models" / "Sharded"
    assert sorted(p.name for p in landed.iterdir()) == [
        f"{prefix}-{index:05d}-of-00004.gguf" for index in range(1, 5)
    ]
    assert (landed / f"{prefix}-00003-of-00004.gguf").read_bytes() == shard_payload(3)


def test_unlisted_shard_that_cannot_be_fetched_fails_its_family_with_reason(tmp_path: Path) -> None:
    module = load_script()
    prefix = "Gone-Set"
    cache = tmp_path / "cache"
    cache.mkdir()
    listing = sharded_listing(prefix, 3, omit=(3,), cache=cache)

    def download(*, repo_id, revision, filename):
        if filename.endswith("00003-of-00003.gguf"):
            raise RuntimeError("404 Client Error: Entry Not Found for url: ...")
        return str(cache / filename)

    summary = module.download_families(
        [sharded_row(prefix)],
        root=tmp_path / "models",
        tree=FakeTree(listing),
        download_fn=download,
        err=__import__("io").StringIO(),
    )

    record = summary["failed"]["sharded"]
    assert "404" in record["reason"] and "Entry Not Found" in record["reason"]
    assert record["failed_destination"].endswith(f"{prefix}-00003-of-00003.gguf")
    # Shards 1 and 2 were never fetched because the miss was probed first.
    assert not (tmp_path / "models" / "Sharded" / f"{prefix}-00001-of-00003.gguf").exists()


def test_one_failing_family_does_not_stop_the_others(tmp_path: Path) -> None:
    module = load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    good = cache / "good.gguf"
    good.write_bytes(PINNED)
    gated = cache / "gated.gguf"
    gated.write_bytes(PINNED)
    rows = [
        {
            "destination": "Good/good.gguf",
            "families": ["good"],
            "repo_id": "owner/good",
            "revision": REVISION,
            "path": "good.gguf",
            "sha256": hashlib.sha256(PINNED).hexdigest(),
            "size_bytes": len(PINNED),
        },
        {
            "destination": "Gated/gated.gguf",
            "families": ["gated"],
            "repo_id": "owner/gated",
            "revision": REVISION,
            "path": "gated.gguf",
            "sha256": hashlib.sha256(PINNED).hexdigest(),
            "size_bytes": len(PINNED),
        },
    ]

    def download(*, repo_id, revision, filename):
        if repo_id == "owner/gated":
            raise RuntimeError("401 Client Error: Repository requires authentication")
        return str(cache / filename)

    log = __import__("io").StringIO()
    summary = module.download_families(
        rows, root=tmp_path / "models", expand_shards=False, download_fn=download, err=log
    )

    assert summary["families"] == 2
    assert summary["families_ok"] == 1
    assert summary["families_failed"] == 1
    # "gated" sorts first, so this also proves the later family still ran.
    assert (tmp_path / "models/Good/good.gguf").read_bytes() == PINNED
    assert not (tmp_path / "models/Gated/gated.gguf").exists()
    assert "401" in summary["failed"]["gated"]["reason"]
    assert summary["failed"]["gated"]["missing_bytes"] == len(PINNED)
    assert "1 families failed: gated" in log.getvalue()
    assert "summary: 2 families, 1 ok, 1 failed" in log.getvalue()


def test_complete_sets_are_not_downloaded_again(tmp_path: Path) -> None:
    module = load_script()
    root = tmp_path / "models"
    target = root / "Done/model.gguf"
    target.parent.mkdir(parents=True)
    target.write_bytes(PINNED)
    row = {
        "destination": "Done/model.gguf",
        "families": ["done"],
        "repo_id": "owner/done",
        "revision": REVISION,
        "path": "model.gguf",
        "sha256": hashlib.sha256(PINNED).hexdigest(),
        "size_bytes": len(PINNED),
    }

    def download(**_):
        raise AssertionError("a complete artifact must not be fetched again")

    summary = module.download_families(
        [row], root=root, expand_shards=False, download_fn=download,
        err=__import__("io").StringIO(),
    )

    assert summary["families_failed"] == 0
    assert summary["artifacts_fetched"] == 0


def test_cli_attempts_every_family_and_exits_nonzero_with_summary(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = load_script()
    cache = tmp_path / "cache"
    cache.mkdir()
    listing = sharded_listing("Broken", 2, omit=(2,), cache=cache)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(__import__("json").dumps({
        "schema": "turbofit.artifact-manifest/v1",
        "artifacts": [
            sharded_row("Broken", "Broken", ("broken",)),
            {
                "destination": "Fine/fine.gguf",
                "families": ["fine"],
                "repo_id": "owner/fine",
                "revision": REVISION,
                "path": "fine.gguf",
                "sha256": hashlib.sha256(PINNED).hexdigest(),
                "size_bytes": len(PINNED),
            },
        ],
    }), encoding="utf-8")
    (cache / "fine.gguf").write_bytes(PINNED)

    monkeypatch.setattr(module, "ArtifactTree", lambda: FakeTree(listing))
    monkeypatch.setattr(module, "hf_download", lambda **kwargs: str(cache / kwargs["filename"]))

    code = module.main(["--all", "--manifest", str(manifest), "--model-root", str(tmp_path / "models")])

    captured = capsys.readouterr()
    assert code == 1
    assert "1 families failed: broken" in captured.err
    assert "broken: FAILED" in captured.err
    assert "fine: ok" in captured.err
    assert (tmp_path / "models/Fine/fine.gguf").read_bytes() == PINNED
    payload = __import__("json").loads(captured.out)
    assert set(payload["failed"]) == {"broken"}


def test_list_still_works_without_a_download(capsys) -> None:
    module = load_script()
    assert module.main(["--list"]) == 0
    rows = __import__("json").loads(capsys.readouterr().out)
    manifest = __import__("json").loads((ROOT / "references/artifact-manifest.json").read_text())
    assert len(rows) == len(manifest["artifacts"])


def test_complete_artifact_symlinked_outside_the_blob_store_is_left_alone(tmp_path: Path) -> None:
    """Archived models live behind symlinks to another volume; verifying is fine.

    Rewriting through such a link would be wrong (the archive is untouched by
    unlinking the link), but a hash-matching file needs no write at all.
    """
    module = load_script()
    root = tmp_path / "storage/models"
    archive = tmp_path / "archive"
    archive.mkdir(parents=True)
    archived = archive / "model.gguf"
    archived.write_bytes(PINNED)
    link = root / "family/model.gguf"
    link.parent.mkdir(parents=True)
    link.symlink_to(archived)
    row = {
        "destination": "family/model.gguf",
        "families": ["archived"],
        "repo_id": "owner/archived",
        "revision": REVISION,
        "path": "model.gguf",
        "sha256": hashlib.sha256(PINNED).hexdigest(),
        "size_bytes": len(PINNED),
    }

    def download(**_):
        raise AssertionError("an archived complete artifact must not be re-fetched")

    result = module.install_artifact(row, root=root, download_fn=download)

    assert result["downloaded"] is False
    assert link.is_symlink()
    assert archived.read_bytes() == PINNED
