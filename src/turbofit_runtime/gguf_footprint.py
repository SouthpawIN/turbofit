"""Measure a GGUF's expert footprint from its own tensor table.

`experts.plan_expert_offload` needs three numbers that only the file itself can
answer honestly: the whole model size, the bytes held by MoE expert tensors, and
how many expert layers there are. Guessing them from a parameter count or a
model-family table breaks the moment a user has a quant the table never saw.

So this reads the GGUF header directly: the metadata block and the tensor
descriptor list. That is a few kilobytes at the front of each shard, no matter
how large the weights are, and it needs no third-party package (llama.cpp hosts
often have a bare system Python).

Tensor byte sizes come from the descriptor offsets rather than a quantization
block-size table, so unusual quant types (IQ*, TQ*, ternaries) are measured
exactly instead of being guessed at.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")
LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")

# llama.cpp spells MoE tensors several ways across architectures:
#   blk.N.ffn_gate_exps.weight      (llama/qwen/deepseek modern)
#   blk.N.ffn_up_expert.weight      (older naming)
#   blk.N.ffn_gate_experts.weight   (gpt-oss style)
#   blk.N.moe_gate.weight / blk.N.ffn_gate_inp.weight  (router, NOT expert data)
EXPERT_NAME_RE = re.compile(
    r"(?:^|[._])(?:"
    r"ffn_(?:gate|up|down)_?exps?(?:\.|$)"
    r"|ffn_(?:gate|up|down)_experts?(?:\.|$)"
    r"|ffn_experts?(?:\.|$)"
    r"|expert_(?:gate|up|down)(?:\.|$)"
    r")"
)
# Router/gate weights look expert-ish but are small and always GPU-resident;
# counting them would inflate `expert_bytes` by a rounding error only, but they
# are excluded anyway so `expert_layers` matches the layer count llama.cpp uses.
NON_EXPERT_RE = re.compile(r"(?:^|[._])(?:ffn_gate_inp|moe_gate|ffn_gate_inp_shexp)(?:\.|$)")

# GGUF value types -> fixed width, used only to skip metadata.
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                   7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUFFormatError(ValueError):
    """The file does not look like a readable GGUF."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    offset: int
    shape: tuple[int, ...]
    ggml_type: int
    size_bytes: int = 0


@dataclass(frozen=True)
class GGUFFileIndex:
    path: str
    file_size: int
    data_start: int
    tensors: tuple[TensorInfo, ...]
    metadata: dict


@dataclass(frozen=True)
class LayerFootprint:
    """Bytes a single transformer block contributes."""

    index: int
    expert_bytes: int
    other_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.expert_bytes + self.other_bytes


@dataclass(frozen=True)
class ExpertFootprint:
    """What the file says about itself, in bytes and expert layers."""

    total_bytes: int
    expert_bytes: int
    expert_layers: int
    dense_bytes: int
    tensor_count: int
    expert_tensor_count: int
    expert_tensor_names: tuple[str, ...]
    shards: tuple[str, ...]
    layers: tuple[LayerFootprint, ...] = ()

    def as_dict(self) -> dict:
        data = asdict(self)
        data["expert_tensor_names"] = list(self.expert_tensor_names)
        data["shards"] = list(self.shards)
        data["layers"] = [asdict(layer) for layer in self.layers]
        return data


def is_expert_tensor(name: str) -> bool:
    if NON_EXPERT_RE.search(name):
        return False
    return bool(EXPERT_NAME_RE.search(name))


def _read_string(buf: bytes, pos: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    raw = buf[pos:pos + length]
    if len(raw) != length:
        raise GGUFFormatError("truncated GGUF string")
    return raw.decode("utf-8", errors="replace"), pos + length


def _skip_value(buf: bytes, pos: int, value_type: int) -> tuple[object, int]:
    if value_type in _SCALAR_SIZES:
        fmt = _SCALAR_FORMATS[value_type]
        value = struct.unpack_from(fmt, buf, pos)[0]
        return value, pos + _SCALAR_SIZES[value_type]
    if value_type == 8:  # string
        return _read_string(buf, pos)
    if value_type == 9:  # array
        (element_type,) = struct.unpack_from("<I", buf, pos)
        (count,) = struct.unpack_from("<Q", buf, pos + 4)
        pos += 12
        if element_type in _SCALAR_SIZES and element_type != 8:
            width = _SCALAR_SIZES[element_type]
            return None, pos + width * count
        for _ in range(count):
            _, pos = _skip_value(buf, pos, element_type)
        return None, pos
    raise GGUFFormatError(f"unknown GGUF metadata value type {value_type}")


def read_gguf_index(path: str | Path) -> GGUFFileIndex:
    """Parse one GGUF shard's header without touching the weight data."""
    try:
        return _read_gguf_index(path)
    except GGUFFormatError:
        raise
    except struct.error as exc:
        raise GGUFFormatError(f"{path} has a truncated or corrupt GGUF header") from exc


def _read_gguf_index(path: str | Path) -> GGUFFileIndex:
    target = Path(path)
    file_size = target.stat().st_size
    # Metadata for large models is still small; 64 MiB is a generous ceiling
    # that keeps a corrupt file from eating memory.
    with target.open("rb") as handle:
        head = handle.read(64 * 1024 * 1024)
    if head[:4] != GGUF_MAGIC:
        raise GGUFFormatError(f"{target} is not a GGUF file (bad magic)")
    version, = struct.unpack_from("<I", head, 4)
    if version not in {2, 3}:
        raise GGUFFormatError(f"{target} uses unsupported GGUF version {version}")
    n_tensors, = struct.unpack_from("<Q", head, 8)
    n_kv, = struct.unpack_from("<Q", head, 16)
    pos = 24
    metadata: dict = {}
    alignment = DEFAULT_ALIGNMENT
    for _ in range(n_kv):
        key, pos = _read_string(head, pos)
        value_type, = struct.unpack_from("<I", head, pos)
        pos += 4
        value, pos = _skip_value(head, pos, value_type)
        if key == "general.alignment" and isinstance(value, int) and value > 0:
            alignment = value
        if value is not None:
            metadata[key] = value
    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name, pos = _read_string(head, pos)
        n_dims, = struct.unpack_from("<I", head, pos)
        pos += 4
        if n_dims > 8:
            raise GGUFFormatError(f"{target} tensor {name} has {n_dims} dims")
        dims = struct.unpack_from(f"<{n_dims}Q", head, pos)
        pos += 8 * n_dims
        ggml_type, offset = struct.unpack_from("<IQ", head, pos)
        pos += 12
        tensors.append(TensorInfo(name=name, offset=offset, shape=tuple(dims),
                                  ggml_type=ggml_type))
    data_start = (pos + alignment - 1) // alignment * alignment

    # Descriptor offsets are relative to the data section and packed in order,
    # so the gap to the next tensor is that tensor's real byte size.
    ordered = sorted(tensors, key=lambda item: item.offset)
    sized: list[TensorInfo] = []
    for index, tensor in enumerate(ordered):
        end = ordered[index + 1].offset if index + 1 < len(ordered) else file_size - data_start
        size = max(0, end - tensor.offset)
        sized.append(TensorInfo(name=tensor.name, offset=tensor.offset, shape=tensor.shape,
                                ggml_type=tensor.ggml_type, size_bytes=size))
    return GGUFFileIndex(path=str(target), file_size=file_size, data_start=data_start,
                         tensors=tuple(sized), metadata=metadata)


def shard_paths(path: str | Path) -> tuple[Path, ...]:
    """All shards of a split GGUF, in order; a single file returns itself."""
    target = Path(path)
    match = SHARD_RE.match(target.name)
    if not match:
        return (target,)
    count = int(match.group("count"))
    directory = target.parent
    stems = [
        directory / f"{match.group('stem')}-{index:05d}-of-{count:05d}.gguf"
        for index in range(1, count + 1)
    ]
    missing = [str(item) for item in stems if not item.is_file()]
    if missing:
        raise GGUFFormatError(f"missing GGUF shard: {missing[0]}")
    return tuple(stems)


def measure_expert_footprint(
    path: str | Path, *, cache_dir: str | Path | bool | None = None
) -> ExpertFootprint:
    """Sum expert tensor bytes and count expert layers across all shards.

    The result is cached by (resolved path, size, mtime) so resolving a recipe
    twice does not re-read headers, while a re-quantized or replaced file
    invalidates automatically.
    """
    shards = shard_paths(path)
    key = _cache_key(shards)
    cache_file = _cache_path(cache_dir, key)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    total_bytes = 0
    expert_bytes = 0
    layer_indices: set[int] = set()
    expert_names: list[str] = []
    tensor_count = 0
    block_counts: list[int] = []
    per_layer: dict[int, list[int]] = {}
    for shard in shards:
        index = read_gguf_index(shard)
        total_bytes += index.file_size
        tensor_count += len(index.tensors)
        block_count = index.metadata.get("general.block_count") or index.metadata.get(
            f"{index.metadata.get('general.architecture', '')}.block_count"
        )
        if isinstance(block_count, int):
            block_counts.append(block_count)
        for tensor in index.tensors:
            layer_match = LAYER_RE.search(tensor.name)
            if layer_match:
                bucket = per_layer.setdefault(int(layer_match.group(1)), [0, 0])
                if is_expert_tensor(tensor.name):
                    bucket[0] += tensor.size_bytes
                else:
                    bucket[1] += tensor.size_bytes
            if not is_expert_tensor(tensor.name):
                continue
            expert_bytes += tensor.size_bytes
            expert_names.append(tensor.name)
            if layer_match:
                layer_indices.add(int(layer_match.group(1)))

    expert_layers = len(layer_indices)
    if expert_layers == 0 and block_counts:
        # Some architectures emit one fused expert tensor per layer under a
        # name without a block index; fall back to the declared block count.
        expert_layers = max(block_counts)
    footprint = ExpertFootprint(
        total_bytes=total_bytes,
        expert_bytes=expert_bytes,
        expert_layers=expert_layers,
        dense_bytes=max(0, total_bytes - expert_bytes),
        tensor_count=tensor_count,
        expert_tensor_count=len(expert_names),
        expert_tensor_names=tuple(expert_names),
        shards=tuple(str(item) for item in shards),
        layers=tuple(
            LayerFootprint(index=index, expert_bytes=bytes_[0], other_bytes=bytes_[1])
            for index, bytes_ in sorted(per_layer.items())
        ),
    )
    _write_cache(cache_file, key, footprint)
    return footprint


def _cache_key(shards: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for shard in shards:
        stat = shard.stat()
        digest.update(f"{shard.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _cache_path(cache_dir: str | Path | bool | None, key: str) -> Path | None:
    if cache_dir is False:  # explicit opt-out used by tests
        return None
    root = Path(
        cache_dir
        or os.environ.get("TURBOFIT_FOOTPRINT_CACHE")
        or Path.home() / ".cache" / "turbofit" / "gguf-footprints"
    ).expanduser()
    return root / f"{key}.json"


def _read_cache(path: Path | None) -> ExpertFootprint | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExpertFootprint(
            total_bytes=int(data["total_bytes"]),
            expert_bytes=int(data["expert_bytes"]),
            expert_layers=int(data["expert_layers"]),
            dense_bytes=int(data["dense_bytes"]),
            tensor_count=int(data["tensor_count"]),
            expert_tensor_count=int(data["expert_tensor_count"]),
            expert_tensor_names=tuple(data["expert_tensor_names"]),
            shards=tuple(data["shards"]),
            layers=tuple(
                LayerFootprint(index=int(layer["index"]),
                               expert_bytes=int(layer["expert_bytes"]),
                               other_bytes=int(layer["other_bytes"]))
                for layer in data["layers"]
            ),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path | None, key: str, footprint: ExpertFootprint) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = footprint.as_dict()
        payload["key"] = key
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        # A read-only home directory must not stop a launch.
        return
