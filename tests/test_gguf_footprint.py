"""GGUF header measurement: expert bytes and layer count from the file itself."""
from __future__ import annotations

import struct

import pytest

from turbofit_runtime.gguf_footprint import (
    ExpertFootprint,
    GGUFFormatError,
    is_expert_tensor,
    measure_expert_footprint,
    read_gguf_index,
    shard_paths,
)

GIB = 2**30
MIB = 2**20


def _gguf_string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf_value(value: object) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", 5) + struct.pack("<i", value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _gguf_string(value)
    raise TypeError(value)


def write_gguf(path, tensors, metadata=None, *, alignment: int = 32):
    """Write a minimal but well-formed GGUF v3 with the requested tensor sizes.

    `tensors` is [(name, byte_size)]; sizes are encoded the way llama.cpp does
    it, as consecutive offsets into the data section, so the reader has to
    derive them from the descriptor table.
    """
    metadata = {"general.architecture": "test", **(metadata or {})}
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata) + 1)
    for key, value in metadata.items():
        out += _gguf_string(key) + _gguf_value(value)
    out += _gguf_string("general.alignment") + _gguf_value(alignment)
    offsets = []
    running = 0
    for name, size in tensors:
        offsets.append(running)
        running += size
    for (name, _size), offset in zip(tensors, offsets):
        out += _gguf_string(name)
        out += struct.pack("<I", 2)
        out += struct.pack("<QQ", 1, 1)
        out += struct.pack("<I", 0)  # F32; sizing comes from offsets, not type
        out += struct.pack("<Q", offset)
    pad = (-len(out)) % alignment
    out += b"\0" * pad
    # Sparse tail: tensor sizes only need to exist in the descriptor offsets,
    # so a multi-gigabyte test model costs no real disk.
    with path.open("wb") as handle:
        handle.write(bytes(out))
        handle.truncate(len(out) + running)
    return path


def test_expert_tensor_names_match_llama_cpp_spellings():
    assert is_expert_tensor("blk.0.ffn_gate_exps.weight")
    assert is_expert_tensor("blk.12.ffn_up_exps.weight")
    assert is_expert_tensor("blk.3.ffn_down_exps.weight")
    assert is_expert_tensor("blk.3.ffn_gate_expert.weight")
    assert is_expert_tensor("blk.3.ffn_up_experts.weight")
    # Router and shared-expert weights are resident on the GPU, not offloaded.
    assert not is_expert_tensor("blk.0.ffn_gate_inp.weight")
    assert not is_expert_tensor("blk.0.ffn_gate_shexp.weight")
    assert not is_expert_tensor("blk.0.attn_q.weight")
    assert not is_expert_tensor("token_embd.weight")


def test_measures_expert_bytes_and_layer_count(tmp_path):
    tensors = [
        ("token_embd.weight", 8 * MIB),
        ("output.weight", 4 * MIB),
    ]
    for layer in range(4):
        tensors += [
            (f"blk.{layer}.attn_q.weight", 2 * MIB),
            (f"blk.{layer}.ffn_gate_inp.weight", 512 * 1024),
            (f"blk.{layer}.ffn_gate_exps.weight", 10 * MIB),
            (f"blk.{layer}.ffn_up_exps.weight", 10 * MIB),
            (f"blk.{layer}.ffn_down_exps.weight", 10 * MIB),
        ]
    path = write_gguf(tmp_path / "model.gguf", tensors,
                      {"general.block_count": 4})

    footprint = measure_expert_footprint(path, cache_dir=False)

    assert footprint.expert_bytes == 4 * 30 * MIB
    assert footprint.expert_layers == 4
    assert footprint.expert_tensor_count == 12
    assert footprint.total_bytes == path.stat().st_size
    assert footprint.dense_bytes == footprint.total_bytes - footprint.expert_bytes
    assert len(footprint.shards) == 1


def test_split_shards_are_summed(tmp_path):
    half = "shard"  # name only helps readability below
    first = write_gguf(
        tmp_path / "Model-00001-of-00002.gguf",
        [("blk.0.ffn_gate_exps.weight", 6 * MIB), ("blk.0.attn_q.weight", MIB)],
        {"general.block_count": 2},
    )
    write_gguf(
        tmp_path / "Model-00002-of-00002.gguf",
        [("blk.1.ffn_gate_exps.weight", 6 * MIB), ("blk.1.attn_q.weight", MIB)],
        {"general.block_count": 2},
    )

    footprint = measure_expert_footprint(first, cache_dir=False)

    assert footprint.expert_bytes == 12 * MIB
    assert footprint.expert_layers == 2
    assert len(footprint.shards) == 2
    assert footprint.total_bytes == sum(len(p.read_bytes()) for p in (tmp_path).glob("*.gguf"))


def test_missing_shard_is_reported_not_silently_undercounted(tmp_path):
    path = tmp_path / "Model-00001-of-00003.gguf"
    write_gguf(path, [("blk.0.ffn_gate_exps.weight", MIB)])

    with pytest.raises(GGUFFormatError, match="missing GGUF shard"):
        shard_paths(path)


def test_rejects_a_file_that_is_not_gguf(tmp_path):
    path = tmp_path / "nope.gguf"
    path.write_bytes(b"NOTGGUF" + b"\0" * 64)

    with pytest.raises(GGUFFormatError, match="bad magic"):
        read_gguf_index(path)


def test_falls_back_to_declared_block_count_without_layer_indices(tmp_path):
    path = write_gguf(
        tmp_path / "fused.gguf",
        [("expert_weights.0", 4 * MIB), ("attn_q.weight", MIB)],
        {"general.block_count": 12},
    )

    footprint = measure_expert_footprint(path, cache_dir=False)
    assert footprint.expert_layers == 12


def test_measurement_is_cached_and_invalidated_by_size_change(tmp_path):
    path = write_gguf(tmp_path / "cached.gguf",
                      [("blk.0.ffn_gate_exps.weight", 4 * MIB)], {"general.block_count": 1})
    cache = tmp_path / "cache"

    first = measure_expert_footprint(path, cache_dir=cache)
    files = list(cache.glob("*.json"))
    assert len(files) == 1
    second = measure_expert_footprint(path, cache_dir=cache)
    assert first == second

    # Rewriting the file changes size+mtime, so the stale entry must not win.
    write_gguf_tensors = [("blk.0.ffn_gate_exps.weight", 8 * MIB)]
    write_gguf(path, write_gguf_tensors, {"general.block_count": 1})
    third = measure_expert_footprint(path, cache_dir=cache)
    assert third.expert_bytes == 8 * MIB


def test_footprint_is_serialisable_for_evidence():
    footprint = ExpertFootprint(total_bytes=1, expert_bytes=1, expert_layers=1,
                                dense_bytes=0, tensor_count=1, expert_tensor_count=1,
                                expert_tensor_names=("blk.0.ffn_gate_exps.weight",),
                                shards=("a.gguf",))
    assert footprint.as_dict()["expert_tensor_names"] == ["blk.0.ffn_gate_exps.weight"]


def test_remote_http_range_measurement_matches_the_local_reader(tmp_path, monkeypatch):
    """A header read over HTTP Range must produce the local reader's numbers."""
    from turbofit_runtime import gguf_remote
    from turbofit_runtime.gguf_remote import RemoteShard, measure_remote_expert_footprint

    path = write_gguf(
        tmp_path / "remote.gguf",
        [("blk.0.ffn_gate_exps.weight", 4 * MIB), ("blk.0.attn_q.weight", MIB)],
        {"general.block_count": 1},
    )
    payload = path.read_bytes()
    # Force the growth path: the first read is deliberately too short.
    monkeypatch.setattr(gguf_remote, "INITIAL_HEADER_BYTES", 64)
    reads = []

    def fetch(url: str, start: int, length: int) -> bytes:
        reads.append(length)
        return payload[start:start + length]

    shard = RemoteShard(path="remote.gguf", url="https://example.invalid/x", size_bytes=len(payload))
    remote = measure_remote_expert_footprint([shard], fetch)
    local = measure_expert_footprint(path, cache_dir=False)

    assert remote.expert_bytes == local.expert_bytes == 4 * MIB
    assert remote.total_bytes == local.total_bytes
    assert remote.expert_layers == local.expert_layers == 1
    assert reads[0] == 64 and max(reads) > 64


def test_remote_reader_propagates_a_non_gguf_file(monkeypatch):
    from turbofit_runtime.gguf_remote import RemoteShard, read_remote_index

    shard = RemoteShard(path="ple.bin", url="https://example.invalid/x", size_bytes=64)

    with pytest.raises(GGUFFormatError, match="bad magic"):
        read_remote_index(shard, lambda url, start, length: b"NOTGGUF" + b"\0" * 64)
