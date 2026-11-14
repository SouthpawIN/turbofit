#!/usr/bin/env python3
"""Byte-level GGUF metadata editor.

Adds (or replaces) one metadata key in a GGUF file *without* re-encoding tensors
and without needing the tensor type to be known to any GGUF library. This is the
escape hatch for GGUF files whose tensors use a quantization type that the
Python ``gguf`` package (or any other tool) refuses to parse -- for example
``tq2`` (GGML type 42), which raises
``ValueError: np.uint32(42) is not a valid GGMLQuantizationType``.

How it works
------------
A GGUF file is::

    magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
    n_kv x { key: str, value_type: u32, value }
    n_tensors x { name: str, n_dims: u32, dims: u64[n_dims], type: u32, offset: u64 }
    pad -> alignment
    tensor data blob

Tensor ``offset`` fields are relative to the start of the tensor-data section,
so rewriting the header (which changes its length) keeps every offset valid as
long as the data blob is copied verbatim and the new header is padded to the
same alignment. Nothing inside the data section is read or interpreted.

Usage
-----
    # inspect
    patch_gguf_metadata.py --file model.gguf --list

    # add an array key
    patch_gguf_metadata.py --file model.gguf --key maple.attention.sliding_window_pattern \
        --array-of bool --values true,true,true,false

    # add a scalar key
    patch_gguf_metadata.py --file model.gguf --key maple.context_length --uint32 131072

The source file is never modified: output defaults to ``<stem>-patched.gguf``
next to it. Use ``--output`` to choose explicitly.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types (gguf_metadata_value_type)
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

TYPE_NAMES = {
    UINT8: "uint8", INT8: "int8", UINT16: "uint16", INT16: "int16",
    UINT32: "uint32", INT32: "int32", FLOAT32: "float32", BOOL: "bool",
    STRING: "string", ARRAY: "array", UINT64: "uint64", INT64: "int64",
    FLOAT64: "float64",
}
NAME_TO_TYPE = {v: k for k, v in TYPE_NAMES.items()}

_SCALAR_FMT = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h",
    UINT32: "<I", INT32: "<i", FLOAT32: "<f",
    UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}


class GGUFFormatError(Exception):
    """Raised when the input is not a GGUF file we can parse."""


# --------------------------------------------------------------------------- #
# low level read/write helpers
# --------------------------------------------------------------------------- #
class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GGUFFormatError(f"unexpected EOF: wanted {n} bytes, got {len(b)}")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="surrogateescape")


class _Writer:
    def __init__(self):
        self.parts: list[bytes] = []

    def add(self, b: bytes) -> None:
        self.parts.append(b)

    def u32(self, v: int) -> None:
        self.add(struct.pack("<I", v))

    def u64(self, v: int) -> None:
        self.add(struct.pack("<Q", v))

    def string(self, s: str) -> None:
        b = s.encode("utf-8", errors="surrogateescape")
        self.u64(len(b))
        self.add(b)

    def value(self, vtype: int, value) -> None:
        """Emit a single (type, value) pair -- type word is NOT written."""
        if vtype == STRING:
            self.string(value)
        elif vtype == BOOL:
            self.add(struct.pack("<B", 1 if value else 0))
        else:
            fmt = _SCALAR_FMT.get(vtype)
            if fmt is None:
                raise GGUFFormatError(f"unsupported value type {vtype} ({TYPE_NAMES.get(vtype)})")
            self.add(struct.pack(fmt, value))

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


# --------------------------------------------------------------------------- #
# header model
# --------------------------------------------------------------------------- #
class TensorInfo:
    __slots__ = ("name", "n_dims", "dims", "type", "offset")

    def __init__(self, name, n_dims, dims, type_, offset):
        self.name = name
        self.n_dims = n_dims
        self.dims = dims
        self.type = type_
        self.offset = offset

    def __repr__(self) -> str:
        return f"TensorInfo({self.name!r}, dims={self.dims}, type={self.type}, offset={self.offset})"


def _read_value(rd: _Reader, vtype: int):
    if vtype == STRING:
        return rd.string()
    if vtype == BOOL:
        return rd.read(1) != b"\x00"
    if vtype == ARRAY:
        etype = rd.u32()
        count = rd.u64()
        return (etype, [_read_value(rd, etype) for _ in range(count)])
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is None:
        raise GGUFFormatError(f"unsupported metadata value type {vtype}")
    return struct.unpack(fmt, rd.read(struct.calcsize(fmt)))[0]


def _write_value(wr: _Writer, vtype: int, value) -> None:
    wr.u32(vtype)
    if vtype == ARRAY:
        etype, items = value
        wr.u32(etype)
        wr.u64(len(items))
        for it in items:
            wr.value(etype, it)
    else:
        wr.value(vtype, value)


class GGUFHeader:
    """Parsed GGUF header: every KV pair plus tensor infos, in file order."""

    def __init__(self, version, kv, tensors, data_start, alignment, file_size):
        self.version = version
        self.kv = kv                    # list[(key, vtype, value)]
        self.tensors = tensors          # list[TensorInfo]
        self.data_start = data_start    # byte offset of the tensor-data section
        self.alignment = alignment
        self.file_size = file_size

    # -- convenience ------------------------------------------------------- #
    def get(self, key, default=None):
        for k, _t, v in self.kv:
            if k == key:
                return v
        return default

    def keys(self):
        return [k for k, _t, _v in self.kv]

    def describe(self) -> str:
        out = [
            f"version      : {self.version}",
            f"n_kv         : {len(self.kv)}",
            f"n_tensors    : {len(self.tensors)}",
            f"alignment    : {self.alignment}",
            f"data_start   : {self.data_start}",
            f"file_size    : {self.file_size}",
            "",
        ]
        for k, t, v in self.kv:
            tname = TYPE_NAMES.get(t, str(t))
            if t == ARRAY:
                etype, items = v
                ename = TYPE_NAMES.get(etype, str(etype))
                head = ",".join(str(x) for x in items[:8])
                if len(items) > 8:
                    head += f",... ({len(items)} total)"
                out.append(f"  {k:<52} {tname}[{ename}] = [{head}]")
            elif t == STRING:
                s = v if len(v) <= 60 else v[:57] + "..."
                out.append(f"  {k:<52} {tname} = {s!r}")
            else:
                out.append(f"  {k:<52} {tname} = {v}")
        return "\n".join(out)


def read_header(path: str) -> GGUFHeader:
    """Parse a GGUF header. Only the header is read; data is never touched."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFFormatError(f"{path}: bad magic {magic!r} (not a GGUF file)")
        rd = _Reader(fh)
        version = rd.u32()
        if version not in (2, 3):
            raise GGUFFormatError(f"{path}: unsupported GGUF version {version}")
        n_tensors = rd.u64()
        n_kv = rd.u64()

        kv = []
        for _ in range(n_kv):
            key = rd.string()
            vtype = rd.u32()
            kv.append((key, vtype, _read_value(rd, vtype)))

        tensors = []
        for _ in range(n_tensors):
            name = rd.string()
            n_dims = rd.u32()
            dims = [rd.u64() for _ in range(n_dims)]
            ttype = rd.u32()
            offset = rd.u64()
            tensors.append(TensorInfo(name, n_dims, dims, ttype, offset))

        header_end = fh.tell()

    alignment = DEFAULT_ALIGNMENT
    for k, t, v in kv:
        if k == "general.alignment" and t == UINT32:
            alignment = int(v) or DEFAULT_ALIGNMENT
            break
    data_start = (header_end + alignment - 1) // alignment * alignment
    return GGUFHeader(version, kv, tensors, data_start, alignment, size)


def build_header(hdr: GGUFHeader, kv: list) -> bytes:
    """Serialize a header from an (already edited) KV list + unchanged tensors."""
    wr = _Writer()
    wr.add(GGUF_MAGIC)
    wr.u32(hdr.version)
    wr.u64(len(hdr.tensors))
    wr.u64(len(kv))
    for key, vtype, value in kv:
        wr.string(key)
        _write_value(wr, vtype, value)
    for t in hdr.tensors:
        wr.string(t.name)
        wr.u32(t.n_dims)
        for d in t.dims:
            wr.u64(d)
        wr.u32(t.type)
        wr.u64(t.offset)
    raw = wr.getvalue()
    pad = (-len(raw)) % hdr.alignment
    return raw + b"\x00" * pad


# --------------------------------------------------------------------------- #
# edit operation
# --------------------------------------------------------------------------- #
def upsert(kv: list, key: str, vtype: int, value):
    """Insert-or-replace a key, preserving sorted order when the file is sorted."""
    for i, (k, _t, _v) in enumerate(kv):
        if k == key:
            kv[i] = (key, vtype, value)
            return "replaced", i
    keys = [k for k, _t, _v in kv]
    insert_at = len(kv)
    if keys == sorted(keys):
        for i, k in enumerate(keys):
            if key < k:
                insert_at = i
                break
    kv.insert(insert_at, (key, vtype, value))
    return "inserted", insert_at


def patch_file(src: str, dst: str, key: str, vtype: int, value,
               chunk_size: int = 8 * 1024 * 1024) -> GGUFHeader:
    """Rewrite `src` into `dst` with `key` set. Tensor data is copied verbatim."""
    hdr = read_header(src)
    kv = list(hdr.kv)
    action, idx = upsert(kv, key, vtype, value)
    new_header = build_header(hdr, kv)

    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("refusing to overwrite the source file; choose another --output")

    tmp = dst + ".tmp"
    with open(tmp, "wb") as out:
        out.write(new_header)
        with open(src, "rb") as inp:
            inp.seek(hdr.data_start)
            while True:
                b = inp.read(chunk_size)
                if not b:
                    break
                out.write(b)
    os.replace(tmp, dst)

    # verify the result parses and the key is present
    check = read_header(dst)
    got = check.get(key)
    if got != value:
        raise GGUFFormatError(f"post-write verification failed for {key}: {got!r} != {value!r}")
    expected_size = len(new_header) + (hdr.file_size - hdr.data_start)
    actual_size = os.path.getsize(dst)
    if actual_size != expected_size:
        raise GGUFFormatError(f"size mismatch: {actual_size} != {expected_size}")
    print(f"{action} {key!r} at index {idx} ({TYPE_NAMES[vtype]}); "
          f"header {hdr.data_start} -> {len(new_header)} bytes, "
          f"file {hdr.file_size} -> {actual_size} bytes", file=sys.stderr)
    return check


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_values(raw: str, etype: int):
    if raw is None:
        raise SystemExit("error: --values is required with --array-of")
    parts = [p for p in raw.split(",")]
    if etype == BOOL:
        out = []
        for p in parts:
            p = p.strip().lower()
            if p in ("1", "true", "yes"):
                out.append(True)
            elif p in ("0", "false", "no"):
                out.append(False)
            else:
                raise SystemExit(f"error: not a bool: {p!r}")
        return out
    if etype == STRING:
        return parts
    conv = float if etype in (FLOAT32, FLOAT64) else int
    return [conv(p.strip()) for p in parts]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Add or replace one GGUF metadata key at the byte level, "
                    "leaving tensor data untouched.")
    ap.add_argument("--file", required=True, help="source .gguf")
    ap.add_argument("--output", help="destination .gguf (default: <stem>-patched.gguf)")
    ap.add_argument("--key", help="metadata key to add/replace")
    ap.add_argument("--list", action="store_true", help="print the header and exit")
    ap.add_argument("--array-of", dest="array_of", choices=sorted(NAME_TO_TYPE),
                    help="element type for an array-valued key")
    ap.add_argument("--values", help="comma-separated array elements, e.g. true,true,false")
    for tname, tcode in NAME_TO_TYPE.items():
        if tcode in (ARRAY, STRING):
            continue
        ap.add_argument(f"--{tname}", help=f"scalar {tname} value")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing key")
    ap.add_argument("--dry-run", action="store_true", help="show the result without writing")
    args = ap.parse_args(argv)

    if args.list:
        print(read_header(args.file).describe())
        return 0
    if not args.key:
        ap.error("--key is required unless --list is given")

    if args.array_of:
        etype = NAME_TO_TYPE[args.array_of]
        vtype, value = ARRAY, (etype, _parse_values(args.values, etype))
    else:
        scalars = [(NAME_TO_TYPE[n], getattr(args, n.replace("_", "_")))
                   for n in NAME_TO_TYPE if n not in ("array", "string")
                   and getattr(args, n, None) is not None]
        scalars = [(t, v) for t, v in scalars if v is not None]
        if len(scalars) != 1:
            ap.error("give exactly one scalar type flag (e.g. --uint32) or --array-of")
        vtype, value = scalars[0][0], scalars[0][1]

    hdr = read_header(args.file)
    existing = [k for k, _t, _v in hdr.kv]
    if args.key in existing and not args.force and args.dry_run is False:
        print(f"error: {args.key!r} already exists; pass --force to replace it", file=sys.stderr)
        return 2

    if args.dry_run:
        kv = list(hdr.kv)
        action, idx = upsert(kv, args.key, vtype, value)
        print(f"would {action} {args.key!r} at index {idx} "
              f"({TYPE_NAMES[vtype]}{'[' + TYPE_NAMES[value[0]] + ']' if vtype == ARRAY else ''})")
        print(f"n_kv {len(hdr.kv)} -> {len(kv)}")
        return 0

    dst = args.output or (os.path.splitext(args.file)[0] + "-patched.gguf")
    patch_file(args.file, dst, args.key, vtype, value)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
