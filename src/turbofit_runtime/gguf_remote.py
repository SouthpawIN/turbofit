"""Measure a GGUF's expert footprint without downloading the weights.

``references/model-footprints.json`` knows how big every catalog main is, but
tier placement also needs to know how much of it is MoE expert tensors — that
is the only part ``--n-cpu-moe`` can move into system RAM. Publishing a table of
guessed expert shares would be exactly the kind of invented number the rest of
this repo avoids.

The header is enough: a few MiB per shard holding the metadata block and the
tensor descriptor list. Hugging Face serves pinned revisions over HTTP Range,
so this reads only the head of each shard and feeds the *same* parser
(``gguf_footprint.parse_gguf_header``) used for local files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import quote

from .gguf_footprint import (
    ExpertFootprint,
    GGUFFormatError,
    GGUFFileIndex,
    footprint_from_indices,
    parse_gguf_header,
)

MIB = 1024 * 1024
INITIAL_HEADER_BYTES = 8 * MIB
MAX_HEADER_BYTES = 512 * MIB

# fetch(url, start, length) -> bytes, injectable so tests never touch a network.
RangeFetcher = Callable[[str, int, int], bytes]


@dataclass(frozen=True)
class RemoteShard:
    """One shard of a pinned artifact, addressable by byte range."""

    path: str
    url: str
    size_bytes: int


def huggingface_url(repo_id: str, revision: str, path: str) -> str:
    """Resolve URL for ``path`` at a pinned revision (never a moving branch)."""
    return (
        f"https://huggingface.co/{repo_id}/resolve/{revision}/"
        f"{quote(path, safe='/')}"
    )


def _http_range(url: str, start: int, length: int, timeout: float = 90.0) -> bytes:
    import urllib.request

    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{start + length - 1}",
            "User-Agent": "turbofit-expert-footprint/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def read_remote_index(shard: RemoteShard, fetch: RangeFetcher | None = None) -> GGUFFileIndex:
    """Parse one shard's header over HTTP Range, growing the read on truncation."""
    fetch = fetch or _http_range
    length = min(INITIAL_HEADER_BYTES, shard.size_bytes)
    while True:
        data = fetch(shard.url, 0, length)
        try:
            return parse_gguf_header(data, file_size=shard.size_bytes, path=shard.path)
        except GGUFFormatError:
            if length >= shard.size_bytes or length >= MAX_HEADER_BYTES:
                raise
            length = min(length * 2, MAX_HEADER_BYTES, shard.size_bytes)


def measure_remote_expert_footprint(
    shards: Sequence[RemoteShard], fetch: RangeFetcher | None = None
) -> ExpertFootprint:
    """Measure a sharded artifact's expert footprint from its headers only."""
    if not shards:
        raise ValueError("no shards to measure")
    return footprint_from_indices([read_remote_index(shard, fetch) for shard in shards])
