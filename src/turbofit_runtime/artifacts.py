"""Presence, size and shard-set verification for downloaded model artifacts.

A file that exists is not an installed model. Interrupted downloads and
`curl -o` failures leave placeholder files behind (a few bytes of HTML error
page, or just the redirect body), and any presence check that only asks
``path.is_file()`` then reports the model as installed. The next campaign run
gets ``llama_model_load: error loading model`` and the user is told nothing
useful.

This module owns the one definition of "installed":

* a missing file is absent;
* a weights file below :data:`MIN_WEIGHT_BYTES` is a stub, therefore absent;
* a file whose size differs from its pinned/metadata size is incomplete;
* metadata files (``config.json``, ``README.md``, ``chat_template.jinja``, ...)
  are legitimately tiny and are never floor-checked;
* when the manifest has no ``size_bytes`` the size is fetched from the Hugging
  Face tree API instead of defaulting to "anything goes";
* a sharded repo is materialized as the whole sibling set, discovered by
  listing the repo rather than by guessing from a single basename.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

#: Anything smaller than this cannot be real weights. The smallest quantized
#: GGUF in the catalog is gigabytes; 1 MiB only rejects placeholders, empty
#: files and HTML error pages.
MIN_WEIGHT_BYTES = 1024 * 1024

WEIGHT_SUFFIXES = frozenset({
    ".gguf", ".ggml", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".npz",
})

# Metadata files legitimate at a few hundred bytes. Kept explicit so a new
# weights extension can never be silently floor-checked as metadata.
METADATA_SUFFIXES = frozenset({
    ".json", ".md", ".txt", ".jinja", ".yaml", ".yml", ".model", ".tiktoken",
    ".py", ".cfg", ".ini", ".toml",
})
METADATA_NAMES = frozenset({".gitattributes"})

_SHARD_RE = re.compile(
    r"^(?P<prefix>.+?)-(?P<index>\d{5})-of-(?P<count>\d{5})\.(?P<ext>gguf|safetensors)$",
    re.IGNORECASE,
)


class ArtifactMetadataError(ValueError):
    """The artifact's expected size could not be established."""


class ShardSetError(ValueError):
    """A sharded artifact's sibling set is incomplete or inconsistent."""


@dataclass(frozen=True)
class SiblingInfo:
    """One file in a Hugging Face repo revision."""

    path: str
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class FileState:
    """The verified state of one on-disk artifact."""

    path: Path
    present: bool
    size_bytes: int
    expected_bytes: int | None
    reason: str

    @property
    def shortfall(self) -> int:
        if self.expected_bytes is None:
            return 0
        return max(0, self.expected_bytes - self.size_bytes)

    @property
    def percent(self) -> float | None:
        if not self.expected_bytes:
            return None
        return 100.0 * self.size_bytes / self.expected_bytes

    def describe(self) -> str:
        actual = human_bytes(self.size_bytes)
        if self.expected_bytes is None:
            return f"{self.path} is {actual} on disk ({self.reason})"
        return (
            f"{self.path} is {actual} on disk but {human_bytes(self.expected_bytes)} "
            f"is expected ({self.percent:.0f}% present, {self.reason})"
        )


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def is_weight_path(path: str | Path) -> bool:
    """True when *path* names model weights (and so has a size floor)."""
    return Path(str(path)).suffix.lower() in WEIGHT_SUFFIXES


def is_metadata_path(path: str | Path) -> bool:
    name = Path(str(path)).name.lower()
    if name in METADATA_NAMES:
        return True
    return Path(name).suffix.lower() in METADATA_SUFFIXES


def minimum_size(path: str | Path) -> int:
    """The size floor for *path*: 1 MiB for weights, 0 for metadata/other."""
    return MIN_WEIGHT_BYTES if is_weight_path(path) else 0


def inspect_file(path: str | Path, expected_bytes: int | None = None) -> FileState:
    """Classify one artifact, treating placeholders as absent.

    ``expected_bytes`` comes from the pinned manifest when available. When it
    is ``None`` only the weight floor applies -- callers that need an exact
    answer must resolve the size first (:func:`resolve_expected_size`).
    """
    target = Path(path)
    if not target.exists():
        return FileState(target, False, 0, expected_bytes, "missing")
    if not target.is_file():
        return FileState(target, False, 0, expected_bytes, "not-a-regular-file")
    size = target.stat().st_size
    floor = minimum_size(target)
    if floor and size < floor:
        return FileState(target, False, size, expected_bytes, f"stub below {human_bytes(floor)} floor")
    if expected_bytes is not None and expected_bytes > 0 and size != expected_bytes:
        reason = "truncated download" if size < expected_bytes else "larger than pinned size"
        return FileState(target, False, size, expected_bytes, reason)
    return FileState(target, True, size, expected_bytes, "complete")


def is_complete(path: str | Path, expected_bytes: int | None = None) -> bool:
    return inspect_file(path, expected_bytes).present


class ArtifactTree:
    """Hugging Face sibling listing, cached per (repo, revision).

    Injecting ``api`` (anything with ``model_info``) keeps tests offline; the
    production default is ``huggingface_hub.HfApi``.
    """

    def __init__(self, api: Any | None = None) -> None:
        self._api = api
        self._cache: dict[tuple[str, str], dict[str, SiblingInfo]] = {}

    def _client(self) -> Any:
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ArtifactMetadataError(
                    "huggingface_hub is required to fetch artifact sizes; "
                    "install it or add size_bytes to references/artifact-manifest.json"
                ) from exc
            self._api = HfApi()
        return self._api

    def siblings(self, repo_id: str, revision: str | None = None) -> dict[str, SiblingInfo]:
        key = (str(repo_id), str(revision or "main"))
        if key not in self._cache:
            info = self._client().model_info(
                str(repo_id), revision=revision, files_metadata=True
            )
            files: dict[str, SiblingInfo] = {}
            for sibling in getattr(info, "siblings", None) or []:
                name = getattr(sibling, "rfilename", None)
                if not name:
                    continue
                files[str(name)] = SiblingInfo(
                    path=str(name),
                    size_bytes=getattr(sibling, "size", None),
                    sha256=getattr(getattr(sibling, "lfs", None), "sha256", None),
                )
            self._cache[key] = files
        return self._cache[key]


def resolve_expected_size(
    artifact: Mapping[str, Any],
    *,
    tree: ArtifactTree | None = None,
) -> int:
    """Pinned size for *artifact*, falling back to the HF tree API.

    Never returns ``None``: an artifact whose size cannot be established is
    rejected instead of being accepted at any size (the original bug).
    """
    raw = artifact.get("size_bytes")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    repo_id = str(artifact.get("repo_id") or "")
    path = str(artifact.get("path") or "")
    if not repo_id or not path:
        raise ArtifactMetadataError(
            f"artifact {artifact.get('destination') or path or '<unknown>'} has no size_bytes "
            "and no repo_id/path to resolve one from"
        )
    listing = (tree or ArtifactTree()).siblings(repo_id, artifact.get("revision"))
    sibling = listing.get(path)
    if sibling is None:
        raise ArtifactMetadataError(
            f"artifact {path} is not present in {repo_id} at the pinned revision; "
            "the manifest points at a file that was never published"
        )
    if not sibling.size_bytes or sibling.size_bytes <= 0:
        raise ArtifactMetadataError(
            f"Hugging Face returned no size for {repo_id}:{path}; "
            "refusing to treat any file size as complete"
        )
    return int(sibling.size_bytes)


def parse_shard(name: str) -> tuple[str, int, str] | None:
    """Return ``(prefix, count, extension)`` for a shard filename."""
    match = _SHARD_RE.match(PurePosixPath(str(name)).name)
    if not match:
        return None
    return match.group("prefix"), int(match.group("count")), match.group("ext").lower()


def shard_names(name: str, count: int) -> list[str]:
    """All sibling basenames for a shard set named by *name*."""
    parsed = parse_shard(name)
    if parsed is not None:
        prefix = parsed[0]
        ext = parsed[2]
    else:
        path = PurePosixPath(str(name))
        prefix, ext = path.stem, path.suffix.lstrip(".").lower() or "gguf"
    return [f"{prefix}-{index:05d}-of-{count:05d}.{ext}" for index in range(1, count + 1)]


def discover_shard_set(
    name: str,
    siblings: Mapping[str, Any] | Iterable[str],
) -> list[str]:
    """Enumerate the complete shard set for *name* from repo siblings.

    A sharded repo ships ``X-00001-of-00004.gguf`` while the catalog may name
    the single basename ``X.gguf``. Returning ``[name]`` there wrote a stub;
    this resolves ``X`` to the four real files. Paths must stay in one
    directory, as llama.cpp requires.
    """
    available = set(siblings.keys() if isinstance(siblings, Mapping) else siblings)
    path = PurePosixPath(str(name))
    directory = path.parent
    parsed = parse_shard(path.name)
    count: int | None = None
    if parsed is not None:
        count = parsed[1]
    else:
        # Catalog may name the unsharded basename even though the repo is sharded.
        stem = path.stem
        first_re = re.compile(
            rf"^{re.escape(stem)}-0*1-of-(?P<count>\d{{5}})\.(?P<ext>gguf|safetensors)$",
            re.IGNORECASE,
        )
        for candidate in available:
            candidate_path = PurePosixPath(str(candidate))
            if candidate_path.parent != directory:
                continue
            match = first_re.match(candidate_path.name)
            if match:
                count = int(match.group("count"))
                break
    if count is None:
        # Single-file repo, or a path the revision does not list: install what
        # was asked for (verification still applies after download).
        return [str(name)]

    names = shard_names(path.name, count)
    resolved = [str(directory / member) if str(directory) != "." else member for member in names]
    missing = [member for member in resolved if member not in available]
    if missing:
        raise ShardSetError(
            f"incomplete shard set for {name}: missing {', '.join(missing)} "
            f"(expected {count} shards in {directory or '.'})"
        )
    return resolved


def expand_artifact(
    artifact: Mapping[str, Any],
    siblings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand one manifest row into its complete, per-shard rows.

    Size and hash for each shard come from the repo listing, so callers get the
    exact metadata the manifest lacks. Destination directories are preserved
    and the destination names follow the repo shard names.
    """
    path = str(artifact.get("path") or "")
    members = discover_shard_set(path, siblings)
    if members == [path]:
        return [dict(artifact)]
    destination = PurePosixPath(str(artifact.get("destination") or path))
    rows: list[dict[str, Any]] = []
    for member in members:
        sibling = siblings.get(member)
        size = getattr(sibling, "size_bytes", None)
        if size is None and isinstance(sibling, Mapping):
            size = sibling.get("size_bytes")
        sha = getattr(sibling, "sha256", None)
        if sha is None and isinstance(sibling, Mapping):
            sha = sibling.get("sha256")
        if not size:
            raise ArtifactMetadataError(
                f"no size metadata for shard {member} in {artifact.get('repo_id')}; "
                "refusing to materialize an unverifiable shard"
            )
        row = dict(artifact)
        row["path"] = member
        row["destination"] = str(destination.parent / PurePosixPath(member).name)
        row["size_bytes"] = int(size)
        if sha:
            row["sha256"] = str(sha)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Failure explanation
# ---------------------------------------------------------------------------

_LOAD_ERROR_MARKERS = (
    "error loading model",
    "failed to load model",
    "unable to load model",
    "invalid magic",
    "tensor data is not within the file bounds",
    "error while handling argument",
)


def _log_evidence(log_path: str | Path | None, limit: int = 3) -> str:
    if not log_path:
        return ""
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    hits = [line.strip() for line in lines if any(marker in line.lower() for marker in _LOAD_ERROR_MARKERS)]
    if not hits:
        return ""
    return " Engine said: " + " | ".join(hits[-limit:])


def likely_cause(state: FileState) -> str:
    if state.reason == "missing":
        return "the weights were never downloaded"
    if state.reason.startswith("stub"):
        return (
            "a placeholder file was written by an interrupted download and then counted "
            "as installed"
        )
    if state.reason in {"truncated download", "size-mismatch"}:
        if state.percent is not None and state.percent >= 90:
            return "the download stopped before the last shard/bytes landed"
        return "the download is incomplete or was overwritten by an error page"
    if state.reason == "larger than pinned size":
        return "the file on disk is not the pinned artifact"
    if state.reason == "not-a-regular-file":
        return "the path is a directory or special file, not weights"
    return "the file is the right size, so the failure is more likely a tokenizer, architecture or projector mismatch"


def describe_load_failure(
    *,
    role: str,
    model: str,
    file: str | Path | None,
    expected_bytes: int | None = None,
    log_path: str | Path | None = None,
    missing_shards: Iterable[str] | None = None,
) -> str:
    """Build the operator-facing message for an engine load exit.

    Names the model, the file, actual versus expected size and the likely
    cause, instead of a bare ``exited during load``.
    """
    identity = f"{role} failed to load model {model or '<unknown>'}"
    if not file:
        return f"{identity}: the recipe did not resolve a model file.{_log_evidence(log_path)}"
    target = Path(file)
    if target.is_dir():
        return (
            f"{identity}: {target} is a checkpoint directory, so the failure is inside the "
            f"checkpoint rather than a truncated single-file download."
            f"{_log_evidence(log_path)}"
        )
    state = inspect_file(file, expected_bytes)
    detail = state.describe()
    shards = list(missing_shards or ())
    if shards:
        detail += f"; shard set is incomplete, missing {', '.join(shards)}"
        cause = "a sharded model is only partially materialized"
    else:
        cause = likely_cause(state)
    remedy = "Re-run `scripts/download-artifacts --family <family>` (it replaces stub files)"
    if log_path:
        remedy += " and inspect the engine log" + _log_evidence(log_path)
    return (
        f"{identity}: {detail}. Likely cause: {cause}. {remedy}"
    )


_MANIFEST_CACHE: dict[str, list[dict[str, Any]]] = {}


def manifest_rows(manifest_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Cached artifact-manifest rows (best effort; never raises)."""
    path = Path(manifest_path) if manifest_path else (
        Path(__file__).resolve().parents[2] / "references/artifact-manifest.json"
    )
    key = str(path)
    if key not in _MANIFEST_CACHE:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _MANIFEST_CACHE[key] = []
        else:
            rows = payload.get("artifacts") if isinstance(payload, Mapping) else None
            _MANIFEST_CACHE[key] = [dict(row) for row in rows or [] if isinstance(row, Mapping)]
    return _MANIFEST_CACHE[key]


def expected_size_for_path(
    file: str | Path,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    manifest_path: str | Path | None = None,
) -> int | None:
    """Pinned size for an on-disk artifact, matched by destination basename."""
    candidate = Path(file)
    names = {candidate.name, candidate.as_posix()}
    for row in (rows if rows is not None else manifest_rows(manifest_path)):
        destination = str(row.get("destination") or "")
        if not destination:
            continue
        if destination == candidate.as_posix() or Path(destination).name == candidate.name:
            raw = row.get("size_bytes")
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return raw
    return None


def expected_size_for_model_root(
    file: str | Path,
    *,
    model_root: str | Path | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> int | None:
    """Pinned size matched by path relative to the model root."""
    root = Path(
        model_root or os.environ.get("TURBOFIT_MODEL_ROOT", "~/Models/storage/gguf")
    ).expanduser()
    try:
        relative = Path(file).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        relative = Path(file).name
    for row in rows if rows is not None else manifest_rows():
        if str(row.get("destination")) == relative:
            raw = row.get("size_bytes")
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                return raw
    return expected_size_for_path(file, rows=rows)


__all__ = [
    "ArtifactMetadataError",
    "ArtifactTree",
    "FileState",
    "MIN_WEIGHT_BYTES",
    "METADATA_SUFFIXES",
    "SiblingInfo",
    "ShardSetError",
    "WEIGHT_SUFFIXES",
    "describe_load_failure",
    "discover_shard_set",
    "expand_artifact",
    "expected_size_for_model_root",
    "expected_size_for_path",
    "human_bytes",
    "inspect_file",
    "is_complete",
    "is_metadata_path",
    "is_weight_path",
    "likely_cause",
    "manifest_rows",
    "minimum_size",
    "parse_shard",
    "resolve_expected_size",
    "shard_names",
]
