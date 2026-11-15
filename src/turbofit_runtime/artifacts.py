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
import subprocess
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
    # The floor catches placeholders; it must not condemn a file whose own pinned
    # size is tiny (a 2-byte weight-scale sidecar is legitimate, not a stub).
    floor_applies = bool(floor) and (expected_bytes is None or expected_bytes >= floor)
    if floor_applies and size < floor:
        return FileState(target, False, size, expected_bytes, f"stub below {human_bytes(floor)} floor")
    if expected_bytes is not None and expected_bytes > 0:
        # Sizes are pinned from repo metadata; a 1% band absorbs repack rounding
        # without letting a genuinely truncated or substituted file pass.
        tolerance = max(1024, int(expected_bytes * 0.01))
        if size < expected_bytes - tolerance:
            return FileState(target, False, size, expected_bytes, "truncated download")
        if size > expected_bytes + tolerance:
            return FileState(target, False, size, expected_bytes, "larger than pinned size")
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


def shard_set_members(
    name: str,
    siblings: Mapping[str, Any] | Iterable[str],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Enumerate the shard set for *name* and the members the listing lacks.

    Returns ``(members, missing)`` where *members* is the complete set the
    filename promises and *missing* are those the repo listing does not carry.
    With ``strict`` (the verifier's contract) an incomplete set raises
    :class:`ShardSetError`; the downloader passes ``strict=False`` because a
    shard absent from a listing is still a work item to attempt -- lists can be
    stale, and only the fetch itself proves a shard is really gone.
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
        return [str(name)], []

    names = shard_names(path.name, count)
    resolved = [str(directory / member) if str(directory) != "." else member for member in names]
    missing = [member for member in resolved if member not in available]
    if missing and strict:
        raise ShardSetError(
            f"incomplete shard set for {name}: missing {', '.join(missing)} "
            f"(expected {count} shards in {directory or '.'})"
        )
    return resolved, missing


def discover_shard_set(
    name: str,
    siblings: Mapping[str, Any] | Iterable[str],
    *,
    strict: bool = True,
) -> list[str]:
    """Enumerate the complete shard set for *name* from repo siblings.

    A sharded repo ships ``X-00001-of-00004.gguf`` while the catalog may name
    the single basename ``X.gguf``. Returning ``[name]`` there wrote a stub;
    this resolves ``X`` to the four real files. Paths must stay in one
    directory, as llama.cpp requires.
    """
    return shard_set_members(name, siblings, strict=strict)[0]


def expand_artifact(
    artifact: Mapping[str, Any],
    siblings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand one manifest row into its complete, per-shard rows.

    Size and hash for each shard come from the repo listing, so callers get the
    exact metadata the manifest lacks. Destination directories are preserved
    and the destination names follow the repo shard names.

    A shard the listing does not carry is still emitted as a row (with
    ``size_bytes=None`` and ``unlisted_shard=True``): an incomplete upstream
    listing used to abort the whole run, and the fetch itself is the only way
    to distinguish a stale list from a genuinely removed shard. Those rows are
    unverifiable until downloaded, so they must be attempted before the rest of
    the set rather than after gigabytes of siblings.
    """
    path = str(artifact.get("path") or "")
    members, unlisted = shard_set_members(path, siblings, strict=False)
    if members == [path]:
        return [dict(artifact)]
    destination = PurePosixPath(str(artifact.get("destination") or path))
    unlisted_set = set(unlisted)
    rows: list[dict[str, Any]] = []
    for member in sorted(members, key=lambda name: name not in unlisted_set):
        sibling = siblings.get(member)
        size = getattr(sibling, "size_bytes", None)
        if size is None and isinstance(sibling, Mapping):
            size = sibling.get("size_bytes")
        sha = getattr(sibling, "sha256", None)
        if sha is None and isinstance(sibling, Mapping):
            sha = sibling.get("sha256")
        row = dict(artifact)
        row["path"] = member
        row["destination"] = str(destination.parent / PurePosixPath(member).name)
        if member in unlisted_set:
            # No pinned size or hash exists for a shard the repo does not list;
            # the manifest's size belongs to a different member, so it must not
            # be reused here.
            row["size_bytes"] = None
            row.pop("sha256", None)
            row["unlisted_shard"] = True
            rows.append(row)
            continue
        if not size:
            raise ArtifactMetadataError(
                f"no size metadata for shard {member} in {artifact.get('repo_id')}; "
                "refusing to materialize an unverifiable shard"
            )
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


def log_tail(log_path: str | Path | None, limit: int = 15) -> list[str]:
    """The last ``limit`` non-empty lines of a process log (best effort).

    A readiness or load failure must be explained with the engine's own words.
    Summarising the log as a guessed cause is how a healthy engine that exited
    after serving got reported as a tokenizer/architecture mismatch.
    """
    if not log_path:
        return []
    try:
        raw = Path(log_path).read_bytes()
    except OSError:
        return []
    if not raw:
        return []
    # Read a bounded tail: engine logs can be many megabytes.
    window = raw[-65536:]
    if len(window) < len(raw):
        newline = window.find(b"\n")
        window = window[newline + 1:] if newline >= 0 else window
    lines = window.decode("utf-8", errors="replace").splitlines()
    return [line.rstrip() for line in lines if line.strip()][-limit:]


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


SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 6: "SIGABRT", 7: "SIGBUS",
    8: "SIGFPE", 9: "SIGKILL", 11: "SIGSEGV", 13: "SIGPIPE", 15: "SIGTERM",
}


def signal_name(exit_code: int) -> str:
    """Human name for a negative ``Popen`` return code (``-9`` -> ``SIGKILL``)."""
    number = abs(int(exit_code))
    return SIGNAL_NAMES.get(number, f"signal {number}")


def kernel_oom_killed(pid: int | None, *, lines: int = 400) -> bool:
    """Best-effort: did the kernel OOM killer name *pid* recently?

    Only a best effort: it needs a readable kernel log. A ``False`` answer means
    "no evidence of an OOM kill", never "the host had enough memory".
    """
    if not pid:
        return False
    commands = (
        ["journalctl", "-k", "-n", str(lines), "--no-pager", "-o", "cat"],
        ["dmesg"],
    )
    marker = f"killed process {pid}"
    for command in commands:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        blob = (completed.stdout or "") + (completed.stderr or "")
        if marker in blob.lower():
            return True
    return False


def describe_signal_exit(
    *,
    role: str,
    model: str,
    exit_code: int,
    log_path: str | Path | None = None,
    pid: int | None = None,
    log_tail_lines: Iterable[str] | None = None,
) -> str:
    """Explain a *killed* engine, which is not a load failure.

    A negative exit code means the engine was terminated by a signal. Saying
    "failed to load model (exit code -9)" blames the weights for a process that
    something else killed, and the operator then re-downloads a complete model.
    This names the signal, decides whether the kernel OOM killer sent it, and
    otherwise says plainly that SIGKILL is external by definition.
    """
    name = signal_name(exit_code)
    number = abs(int(exit_code))
    tail = list(log_tail_lines if log_tail_lines is not None else log_tail(log_path))
    if number == 9:
        if kernel_oom_killed(pid):
            cause = (
                "the Linux OOM killer sent SIGKILL because the host ran out of "
                "system RAM; the engine never got a chance to fail on its own"
            )
        else:
            cause = (
                "no OOM-killer record names this pid, and an engine cannot SIGKILL "
                "itself, so an external process sent the signal: an operator or agent "
                "`pkill -9 -f llama-server`, a concurrent campaign reclaiming these "
                "ports, or a supervisor timeout/cleanup. GPU exhaustion does not look "
                "like this: it aborts with a CUDA out-of-memory error (SIGABRT or exit 1)"
            )
    elif number == 6:
        cause = (
            "the engine aborted (SIGABRT), which is what a CUDA/ggml assertion such as "
            "out-of-memory does after it prints the failing allocation"
        )
    elif number == 11:
        cause = "the engine dereferenced invalid memory (SIGSEGV)"
    else:
        cause = f"the engine was terminated by {name} (signal {number})"
    message = (
        f"{role} engine for model {model} was killed by {name} (signal {number})"
        f"{f' (pid {pid})' if pid else ''}; this is not a model load failure: {cause}."
    )
    if tail:
        message += f" Engine log {log_path} (last {len(tail)} lines):\n" + "\n".join(tail)
    return message


def describe_clean_exit(
    *,
    role: str,
    model: str,
    log_path: str | Path | None = None,
    pid: int | None = None,
    log_tail_lines: Iterable[str] | None = None,
) -> str:
    """Explain an engine that exited 0 *during load* — a shutdown, not a failure.

    llama-server handles SIGTERM and exits 0, so an external ``pkill``, a
    supervisor stop or a concurrent campaign reclaiming the port looks like
    "failed to load model (exit code 0)" next to a complete weights file.
    """
    tail = list(log_tail_lines if log_tail_lines is not None else log_tail(log_path))
    message = (
        f"{role} engine for model {model} exited with code 0 while loading"
        f"{f' (pid {pid})' if pid else ''}: a clean exit during load is a shutdown"
        " (SIGTERM from a supervisor, an operator `pkill`, or a concurrent campaign"
        " stopping engines), not a rejected model — the weights were not the cause."
    )
    if tail:
        message += f" Engine log {log_path} (last {len(tail)} lines):\n" + "\n".join(tail)
    return message


def describe_load_failure(
    *,
    role: str,
    model: str,
    file: str | Path | None,
    expected_bytes: int | None = None,
    log_path: str | Path | None = None,
    missing_shards: Iterable[str] | None = None,
    log_tail_lines: Iterable[str] | None = None,
    exit_code: int | None = None,
    pid: int | None = None,
) -> str:
    """Build the operator-facing message for an engine load exit.

    Names the model, the file and actual versus expected size, then quotes the
    engine's own last log lines. It never guesses a cause that the log already
    answers: a complete file plus a working engine is not a tokenizer mismatch.
    A negative exit code is a signal death, delegated to
    :func:`describe_signal_exit` so a killed process is never reported as a bad
    model file.
    """
    if exit_code is not None and int(exit_code) < 0:
        return describe_signal_exit(
            role=role, model=model, exit_code=int(exit_code), log_path=log_path,
            pid=pid, log_tail_lines=log_tail_lines,
        )
    if exit_code is not None and int(exit_code) == 0:
        return describe_clean_exit(
            role=role, model=model, log_path=log_path, pid=pid,
            log_tail_lines=log_tail_lines,
        )
    identity = f"{role} failed to load model {model or '<unknown>'}"
    tail = list(log_tail_lines if log_tail_lines is not None else log_tail(log_path))
    exit_note = f" (exit code {exit_code})" if exit_code is not None else ""
    tail_note = ""

    if not file:
        detail = "the recipe did not resolve a model file"
        cause = "no weights file reached the engine"
        state = None
    elif Path(file).is_dir():
        detail = (
            f"{Path(file)} is a checkpoint directory, so the failure is inside the "
            "checkpoint rather than a truncated single-file download"
        )
        cause = "the checkpoint itself is rejected by the engine"
        state = None
    else:
        state = inspect_file(file, expected_bytes)
        detail = state.describe()
        shards = list(missing_shards or ())
        if shards:
            detail += f"; shard set is incomplete, missing {', '.join(shards)}"
            cause = "a sharded model is only partially materialized"
        else:
            cause = likely_cause(state)

    remedy = (
        "Re-run `scripts/download-artifacts --family <family>` (it replaces stub files) "
        "and inspect the engine log"
    )
    if tail and not (state is not None and state.present):
        # The file is not verifiably complete: quote the engine and the cause.
        return (
            f"{identity}{exit_note}: {detail}. Likely cause: {cause}. {remedy}."
            f" Engine log {log_path} (last {len(tail)} lines):\n" + "\n".join(tail)
        )
    if tail:
        return (
            f"{identity}{exit_note}: {detail}. The weights are present and the right size, "
            f"so this is not a truncated-model failure. Engine log {log_path} "
            f"(last {len(tail)} lines):\n" + "\n".join(tail)
        )
    if state is not None and state.present:
        # Nothing to quote, but the file is verifiably complete: say so instead
        # of inventing a tokenizer/architecture explanation for the exit.
        cause = (
            "the weights are present and the right size, so the engine exited for "
            "its own reason rather than because of a truncated model "
            f"(no engine log content at {log_path or '<path unavailable>'})"
        )
    return f"{identity}{exit_note}: {detail}. Likely cause: {cause}. {remedy}."


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
    "shard_set_members",
]
