"""Row-level artifact gate for the catalog campaign.

The campaign used to attempt rows whose weights were still downloading (or were
only a stub file left by a failed ``curl``), then spent a full attempt, a GPU
clear gate and a failure-evidence directory rediscovering what
:mod:`turbofit_runtime.artifacts` already knows. This module turns that
knowledge into row selection: a row whose engine weights are not verifiably on
disk is reported as ``waiting-download`` with the missing byte count, instead of
being launched and failed.

The definition of "installed" is the one owned by
:func:`turbofit_runtime.artifacts.inspect_file`: a missing file, a sub-1 MiB
stub, a size mismatch against the pinned manifest, or an empty checkpoint
directory is *not* installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import (
    expected_size_for_model_root,
    human_bytes,
    inspect_file,
    parse_shard,
    shard_names,
)

#: Component fields that name on-disk artifacts the engine will open.
_ARTIFACT_FIELDS = ("model_path", "projector_path", "draft_path")


@dataclass(frozen=True)
class ArtifactProblem:
    """One artifact that is not (fully) installed."""

    role: str
    path: str
    missing_bytes: int
    reason: str

    def describe(self) -> str:
        if self.missing_bytes:
            return f"{self.role} {self.reason} ({human_bytes(self.missing_bytes)} missing)"
        return f"{self.role} {self.reason}"


def shard_members(path: Path) -> list[Path]:
    """Every file the engine will open for *path* (all shards, or just it)."""
    parsed = parse_shard(path.name)
    if parsed is None:
        return [path]
    return [path.with_name(name) for name in shard_names(path.name, parsed[1])]


def artifact_problems(
    components: Iterable[Any], *, model_root: str | Path | None = None
) -> list[ArtifactProblem]:
    """Artifacts of *components* that are not verifiably installed."""
    problems: list[ArtifactProblem] = []
    for component in components:
        role = str(getattr(component, "role", "") or "main")
        for field in _ARTIFACT_FIELDS:
            value = str(getattr(component, field, "") or "")
            if not value:
                continue
            target = Path(value)
            if target.is_dir():
                # vLLM/MLX checkpoints are directories; an empty one is a
                # download that never started.
                try:
                    empty = not any(target.iterdir())
                except OSError:
                    empty = False
                if empty:
                    problems.append(ArtifactProblem(
                        role=f"{role}:{field.removesuffix('_path')}",
                        path=str(target),
                        missing_bytes=0,
                        reason=f"directory {target} is empty",
                    ))
                continue
            for member in shard_members(target):
                expected = expected_size_for_model_root(member, model_root=model_root)
                state = inspect_file(member, expected)
                if state.present:
                    continue
                problems.append(ArtifactProblem(
                    role=f"{role}:{field.removesuffix('_path')}",
                    path=str(member),
                    missing_bytes=state.shortfall if state.expected_bytes else 0,
                    reason=state.describe(),
                ))
    return problems


def describe_problems(problems: Iterable[ArtifactProblem]) -> str:
    """One line naming every missing artifact and the bytes still to fetch."""
    items = list(problems)
    missing = sum(item.missing_bytes for item in items)
    detail = "; ".join(item.describe() for item in items)
    if missing:
        return f"waiting-download: {human_bytes(missing)} missing — {detail}"
    return f"waiting-download: {detail}"


@dataclass(frozen=True)
class WaitingDownload:
    """A row that cannot run until its weights finish downloading."""

    row_id: str
    reason: str
    missing_bytes: int
    problems: tuple[ArtifactProblem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "status": "waiting-download",
            "reason": self.reason,
            "missing_bytes": self.missing_bytes,
            "artifacts": [problem.path for problem in self.problems],
        }


def waiting_download_rows(
    configuration_index: Mapping[str, dict[str, Any]],
    recipes: Any,
) -> dict[str, WaitingDownload]:
    """Rows whose engine artifacts are not fully installed, with a reason each.

    Resolution failures are skipped: they are not proof of an incomplete
    artifact, and a broken recipe is the row's own failure to report.
    """
    model_root = getattr(recipes, "model_root", None)
    result: dict[str, WaitingDownload] = {}
    for row_id, row in configuration_index.items():
        try:
            recipe = recipes.resolve_catalog_configuration(dict(row))
        except Exception:  # noqa: BLE001 - an unresolved recipe is not this gate's verdict
            continue
        problems = artifact_problems(recipe.components, model_root=model_root)
        if problems:
            result[row_id] = WaitingDownload(
                row_id=row_id,
                reason=describe_problems(problems),
                missing_bytes=sum(item.missing_bytes for item in problems),
                problems=tuple(problems),
            )
    return result


__all__ = [
    "ArtifactProblem",
    "WaitingDownload",
    "artifact_problems",
    "describe_problems",
    "shard_members",
    "waiting_download_rows",
]
