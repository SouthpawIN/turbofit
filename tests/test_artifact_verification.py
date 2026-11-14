"""Regression tests for artifact completeness, size metadata and shard sets.

These cover the failure that reached a user: a 15-byte placeholder was counted
as an installed model because the manifest row had no ``size_bytes``, so
llama.cpp reported ``failed to load model`` on a file the installer called
complete. Every check here is offline.
"""
from __future__ import annotations

import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from turbofit_runtime import artifacts
from turbofit_runtime.artifacts import (
    ArtifactMetadataError,
    ArtifactTree,
    FileState,
    ShardSetError,
    SiblingInfo,
    describe_load_failure,
    discover_shard_set,
    expand_artifact,
    inspect_file,
    resolve_expected_size,
)

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/download-artifacts"


def load_script():
    if "download_artifacts_under_test" not in __import__("sys").modules:
        loader = SourceFileLoader("download_artifacts_under_test", str(SCRIPT))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        __import__("sys").modules[loader.name] = module
        spec.loader.exec_module(module)
    return __import__("sys").modules["download_artifacts_under_test"]


class FakeTree:
    """Stand-in for ArtifactTree/siblings that never touches the network."""

    def __init__(self, listing):
        self.listing = listing
        self.calls: list[tuple[str, str | None]] = []

    def siblings(self, repo_id, revision=None):
        self.calls.append((repo_id, revision))
        if isinstance(self.listing, Exception):
            raise self.listing
        return self.listing


def weights(size: int, seed: bytes = b"w") -> bytes:
    return (seed * ((size // len(seed)) + 1))[:size]


# ---------------------------------------------------------------------------
# 1. Stub / short-file detection
# ---------------------------------------------------------------------------


def test_stub_weights_below_floor_count_as_absent(tmp_path: Path) -> None:
    stub = tmp_path / "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
    stub.write_bytes(b"<html>404</html>")  # 15 bytes, exactly the observed case

    state = inspect_file(stub, expected_bytes=11_771_546_784)

    assert state.present is False
    assert state.reason.startswith("stub")
    assert state.shortfall == 11_771_546_784 - stub.stat().st_size
    assert "is expected" in state.describe()


def test_complete_weights_pass_and_wrong_size_is_absent(tmp_path: Path) -> None:
    good = tmp_path / "model.gguf"
    good.write_bytes(weights(artifacts.MIN_WEIGHT_BYTES))
    assert inspect_file(good, expected_bytes=good.stat().st_size).present is True

    partial = tmp_path / "partial.gguf"
    partial.write_bytes(weights(artifacts.MIN_WEIGHT_BYTES))
    state = inspect_file(partial, expected_bytes=2 * artifacts.MIN_WEIGHT_BYTES)
    assert state.present is False
    assert state.reason == "truncated download"
    assert state.percent is not None and state.percent < 100


def test_metadata_files_are_never_floor_checked(tmp_path: Path) -> None:
    files = {
        "config.json": b'{"architectures": ["Qwen3ForCausalLM"]}',
        "README.md": b"# model card",
        "chat_template.jinja": b"{{ messages }}",
        "tokenizer.json": b"{}",
        "generation_config.json": b"{}",
        ".gitattributes": b"*.gguf filter=lfs",
    }
    for name, payload in files.items():
        target = tmp_path / name
        target.write_bytes(payload)
        assert artifacts.is_weight_path(target) is False
        assert inspect_file(target, expected_bytes=len(payload)).present is True, name


def test_floor_only_applies_to_weight_suffixes(tmp_path: Path) -> None:
    assert artifacts.minimum_size("x.GGUF") == artifacts.MIN_WEIGHT_BYTES
    assert artifacts.minimum_size("x.safetensors") == artifacts.MIN_WEIGHT_BYTES
    assert artifacts.minimum_size("config.json") == 0
    assert artifacts.minimum_size("README.md") == 0


def test_missing_and_directory_are_absent(tmp_path: Path) -> None:
    assert inspect_file(tmp_path / "nope.gguf").reason == "missing"
    directory = tmp_path / "shards.gguf"
    directory.mkdir()
    assert inspect_file(directory).present is False


# ---------------------------------------------------------------------------
# 2. Size metadata: manifest or Hugging Face tree API, never "anything goes"
# ---------------------------------------------------------------------------


def test_resolve_expected_size_uses_manifest_value_without_network() -> None:
    tree = FakeTree(RuntimeError("network must not be touched"))
    assert resolve_expected_size(
        {"destination": "a/b.gguf", "repo_id": "o/r", "revision": "a" * 40, "path": "b.gguf",
         "size_bytes": 12_000},
        tree=tree,
    ) == 12_000
    assert tree.calls == []


def test_resolve_expected_size_falls_back_to_hf_tree_api() -> None:
    tree = FakeTree({"sub/model.gguf": SiblingInfo("sub/model.gguf", size_bytes=8_589_934_592)})

    size = resolve_expected_size(
        {"destination": "x/model.gguf", "repo_id": "o/r", "revision": "b" * 40,
         "path": "sub/model.gguf"},
        tree=tree,
    )

    assert size == 8_589_934_592
    assert tree.calls == [("o/r", "b" * 40)]


def test_resolve_expected_size_refuses_artifact_without_any_metadata() -> None:
    tree = FakeTree({})
    with pytest.raises(ArtifactMetadataError, match="not present in o/r"):
        resolve_expected_size(
            {"destination": "x/model.gguf", "repo_id": "o/r", "revision": "b" * 40,
             "path": "missing.gguf"},
            tree=tree,
        )


def test_resolve_expected_size_refuses_when_tree_has_no_size() -> None:
    tree = FakeTree({"model.gguf": SiblingInfo("model.gguf", size_bytes=None)})
    with pytest.raises(ArtifactMetadataError, match="no size"):
        resolve_expected_size(
            {"destination": "x/model.gguf", "repo_id": "o/r", "path": "model.gguf"},
            tree=tree,
        )


def test_resolve_expected_size_rejects_artifact_without_repo_to_ask() -> None:
    with pytest.raises(ArtifactMetadataError, match="no repo_id/path"):
        resolve_expected_size({"destination": "x/model.gguf"}, tree=FakeTree({}))


def test_existing_stub_is_replaced_not_reported_complete(tmp_path: Path) -> None:
    module = load_script()
    payload = weights(artifacts.MIN_WEIGHT_BYTES + 1024)
    cached = tmp_path / "cache/model.gguf"
    cached.parent.mkdir()
    cached.write_bytes(payload)
    item = {
        "destination": "family/model.gguf",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "path": "remote/model.gguf",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    destination = tmp_path / "models" / item["destination"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"partial")  # stale placeholder from a failed run

    result = module.install_artifact(
        item, root=tmp_path / "models", download_fn=lambda **_: str(cached)
    )

    assert result["downloaded"] is True
    assert destination.stat().st_size == len(payload)
    assert destination.read_bytes() == payload


def test_install_artifact_accepts_tiny_metadata_with_pinned_size(tmp_path: Path) -> None:
    module = load_script()
    payload = b'{"architectures": ["Qwen3ForCausalLM"]}'
    cached = tmp_path / "cache/config.json"
    cached.parent.mkdir()
    cached.write_bytes(payload)
    item = {
        "destination": "snapshot/config.json",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "path": "8-bit/config.json",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    result = module.install_artifact(
        item, root=tmp_path / "models", download_fn=lambda **_: str(cached)
    )

    assert result["verified"] is True
    assert (tmp_path / "models/snapshot/config.json").read_bytes() == payload


# ---------------------------------------------------------------------------
# 3. Shard sets are enumerated from repo siblings
# ---------------------------------------------------------------------------


def sharded_listing(prefix: str = "Qwen3.8-27B-GSQ-RCO-IQ3_S", count: int = 4):
    return {
        f"{prefix}-{index:05d}-of-{count:05d}.gguf": SiblingInfo(
            f"{prefix}-{index:05d}-of-{count:05d}.gguf",
            size_bytes=2_000_000_000 + index,
            sha256=f"{index:064x}",
        )
        for index in range(1, count + 1)
    } | {"README.md": SiblingInfo("README.md", size_bytes=900)}


def test_shard_set_is_found_when_catalog_names_the_basename() -> None:
    siblings = sharded_listing()

    members = discover_shard_set("Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", siblings)

    assert members == [f"Qwen3.8-27B-GSQ-RCO-IQ3_S-{i:05d}-of-00004.gguf" for i in range(1, 5)]


def test_discover_shard_set_reports_missing_siblings() -> None:
    siblings = sharded_listing()
    del siblings["Qwen3.8-27B-GSQ-RCO-IQ3_S-00003-of-00004.gguf"]

    with pytest.raises(ShardSetError, match="missing Qwen3.8-27B-GSQ-RCO-IQ3_S-00003-of-00004.gguf"):
        discover_shard_set("Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", siblings)


def test_single_file_repo_still_returns_the_named_path() -> None:
    siblings = {"solo.gguf": SiblingInfo("solo.gguf", size_bytes=3_000_000_000)}
    assert discover_shard_set("solo.gguf", siblings) == ["solo.gguf"]


def test_expand_artifact_pins_size_and_hash_per_shard() -> None:
    row = {
        "destination": "Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
        "families": ["qwen3-8-27b-gsq-rco-iq3s"],
        "repo_id": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
        "revision": "d" * 40,
        "path": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
        "sha256": "0" * 64,
        "size_bytes": 4_000_000_000,
    }

    rows = expand_artifact(row, sharded_listing())

    assert len(rows) == 4
    assert {r["destination"] for r in rows} == {
        f"Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S-{i:05d}-of-00004.gguf"
        for i in range(1, 5)
    }
    assert all(r["size_bytes"] > 0 for r in rows)
    assert all(r["sha256"] in {s.sha256 for s in sharded_listing().values()} for r in rows)


def test_expand_rows_materializes_every_shard_offline(tmp_path: Path) -> None:
    module = load_script()
    prefix = "MiniMax-M3-UD-Q4_K_M"
    payloads = {}
    cache = tmp_path / "cache"
    cache.mkdir()
    siblings = {}
    for index in range(1, 4):
        name = f"{prefix}-{index:05d}-of-00003.gguf"
        payload = weights(artifacts.MIN_WEIGHT_BYTES + index * 512, seed=bytes([index]))
        payloads[name] = payload
        (cache / name).write_bytes(payload)
        siblings[name] = SiblingInfo(
            name, size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
    row = {
        "destination": f"UD-Q4_K_M/{prefix}.gguf",
        "families": ["minimax-m3"],
        "repo_id": "unsloth/MiniMax-M3-GGUF",
        "revision": "e" * 40,
        "path": f"{prefix}.gguf",
        "sha256": "0" * 64,
        "size_bytes": 3,
    }
    tree = FakeTree(siblings)
    rows = module.expand_rows([row], tree=tree)

    assert len(rows) == 3
    root = tmp_path / "models"
    for member in rows:
        module.install_artifact(
            member, root=root, tree=tree, download_fn=lambda *, filename, **_: str(cache / filename)
        )

    for name in payloads:
        landed = root / "UD-Q4_K_M" / name
        assert landed.is_file()
        assert landed.read_bytes() == payloads[name]


# ---------------------------------------------------------------------------
# 4. The error itself: model, file, actual vs expected size, likely cause
# ---------------------------------------------------------------------------


def test_load_failure_message_names_model_file_sizes_and_cause(tmp_path: Path) -> None:
    stub = tmp_path / "Qwen3.8-27B-GSQ-RCO" / "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(b"placeholder")
    log = tmp_path / "campaign-main.log"
    log.write_text(
        "llama_model_load: error loading model: failed to load model from "
        f"{stub}\n",
        encoding="utf-8",
    )

    message = describe_load_failure(
        role="main",
        model="qwen3-8-27b-gsq-rco-iq3s",
        file=stub,
        expected_bytes=11_771_546_784,
        log_path=log,
    )

    assert "failed to load model qwen3-8-27b-gsq-rco-iq3s" in message
    assert "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf" in message
    assert "11.0 GiB is expected" in message
    assert "15 B on disk" in message or "11 B on disk" in message
    assert "placeholder" in message
    assert "error loading model" in message


def test_load_failure_message_calls_out_missing_shards(tmp_path: Path) -> None:
    first = tmp_path / "MiniMax-M3-UD-Q4_K_M-00001-of-00003.gguf"
    first.write_bytes(weights(artifacts.MIN_WEIGHT_BYTES))

    message = describe_load_failure(
        role="main",
        model="minimax-m3",
        file=first,
        expected_bytes=artifacts.MIN_WEIGHT_BYTES,
        missing_shards=["MiniMax-M3-UD-Q4_K_M-00002-of-00003.gguf"],
    )

    assert "shard set is incomplete" in message
    assert "MiniMax-M3-UD-Q4_K_M-00002-of-00003.gguf" in message


def test_backend_load_failure_reports_stub_without_spawning(tmp_path: Path, monkeypatch) -> None:
    from turbofit_runtime.backend import CampaignBackend

    stub = tmp_path / "Qwen3.8-27B-GSQ-RCO" / "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(b"placeholder")
    component = type("Component", (), {
        "role": "main",
        "family": "qwen3-8-27b-gsq-rco-iq3s",
        "alias": "qwen3-8-27b-gsq-rco-iq3s",
        "model_path": str(stub),
    })()

    message = CampaignBackend._load_failure(
        object.__new__(CampaignBackend), component, {"log": str(tmp_path / "nope.log")}
    )

    assert "qwen3-8-27b-gsq-rco-iq3s" in message
    assert str(stub) in message
    assert "placeholder" in message
    assert "exited during load" not in message


def test_product_ops_presence_rejects_stub_model(tmp_path: Path, monkeypatch) -> None:
    import product_ops

    monkeypatch.setenv("TURBOFIT_MODEL_ROOT", str(tmp_path))
    row = artifacts.manifest_rows()[0]
    target = tmp_path / row["destination"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"placeholder")

    rows = product_ops._family_files(str((row.get("families") or ["x"])[0]))

    assert rows, "manifest row must be found by family"
    assert all(item["present"] is False for item in rows)
    assert all(item["reason"] != "complete" for item in rows)


def test_artifact_tree_reads_hf_siblings_offline() -> None:
    class Sibling:
        def __init__(self, name, size, sha):
            self.rfilename = name
            self.size = size
            self.lfs = type("Lfs", (), {"sha256": sha})()

    class Api:
        def model_info(self, repo_id, revision=None, files_metadata=False):
            assert files_metadata is True
            return type("Info", (), {"siblings": [Sibling("8-bit/config.json", 42, "f" * 64)]})()

    tree = ArtifactTree(api=Api())

    listing = tree.siblings("owner/repo", "a" * 40)

    assert listing["8-bit/config.json"] == SiblingInfo("8-bit/config.json", 42, "f" * 64)


def test_check_artifact_existence_local_mode_flags_stubs(tmp_path: Path) -> None:
    if "check_artifact_existence_under_test" not in __import__("sys").modules:
        loader = SourceFileLoader(
            "check_artifact_existence_under_test", str(ROOT / "tools/check-artifact-existence")
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        __import__("sys").modules[loader.name] = module
        spec.loader.exec_module(module)
    module = __import__("sys").modules["check_artifact_existence_under_test"]
    metadata = b'{"hidden_size": 5120}'
    rows = [
        {"destination": "Real/model.gguf", "size_bytes": 2 * artifacts.MIN_WEIGHT_BYTES,
         "path": "model.gguf"},
        {"destination": "Real/config.json", "size_bytes": len(metadata), "path": "config.json"},
        # Directory snapshots are checked as trees, not file floors.
        {"destination": "Snapshot", "sha256": "snapshot:" + "a" * 40, "path": ""},
    ]
    (tmp_path / "Real").mkdir()
    (tmp_path / "Snapshot").mkdir()
    (tmp_path / "Real/model.gguf").write_bytes(b"placeholder")  # 15-byte stub
    (tmp_path / "Real/config.json").write_bytes(metadata)

    findings = module.local_check(rows, tmp_path)

    assert [row["destination"] for row in findings] == ["Real/model.gguf"]
    assert findings[0]["status"].startswith("stub")
    assert findings[0]["expected_bytes"] == 2 * artifacts.MIN_WEIGHT_BYTES
