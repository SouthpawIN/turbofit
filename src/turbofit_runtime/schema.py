"""Canonical Main:Aux matrix schema."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VALID_STATUSES = frozenset({"pending", "success", "blocked"})
VALID_CONTEXTS = frozenset({65_536, 131_072, 262_144, 1_048_576})
DEFAULT_METHOD_PRIORITY = ("dspark", "mtp", "nextn")


def _slug(value: str) -> str:
    value = value.lower().replace("1 bit", "1-bit")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _context_slug(context: int) -> str:
    return {
        65_536: "64k",
        131_072: "128k",
        262_144: "262k",
        1_048_576: "1m",
    }[context]


@dataclass(frozen=True)
class MatrixRow:
    id: str
    main: str
    aux: str
    context: int
    status: str = "pending"
    method_priority: tuple[str, ...] = DEFAULT_METHOD_PRIORITY

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {self.status}")
        if self.context not in VALID_CONTEXTS:
            raise ValueError(f"invalid context: {self.context}")
        if not self.method_priority:
            raise ValueError("method_priority cannot be empty")
        expected = self.make_id(self.main, self.aux, self.context)
        if self.id != expected:
            raise ValueError(f"non-deterministic id: {self.id}; expected {expected}")

    @staticmethod
    def make_id(main: str, aux: str, context: int) -> str:
        return f"{_slug(main)}-{_slug(aux)}-{_context_slug(context)}"

    @classmethod
    def from_dict(cls, data: dict) -> "MatrixRow":
        return cls(
            id=str(data["id"]),
            main=str(data["main"]),
            aux=str(data["aux"]),
            context=int(data["context"]),
            status=str(data.get("status", "pending")),
            method_priority=tuple(str(item) for item in data.get("method_priority", DEFAULT_METHOD_PRIORITY)),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "main": self.main,
            "aux": self.aux,
            "context": self.context,
            "status": self.status,
            "method_priority": list(self.method_priority),
        }


@dataclass(frozen=True)
class Matrix:
    rows: tuple[MatrixRow, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[str, str, int]] = set()
        seen_ids: set[str] = set()
        for row in self.rows:
            key = (row.main, row.aux, row.context)
            if key in seen or row.id in seen_ids:
                raise ValueError(f"duplicate matrix row: {row.id}")
            seen.add(key)
            seen_ids.add(row.id)

    @classmethod
    def from_rows(cls, rows: Iterable[MatrixRow]) -> "Matrix":
        return cls(rows=tuple(rows))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "rows": [row.to_dict() for row in self.rows]}


def load_matrix(path: Path | str) -> Matrix:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported matrix schema: {data.get('schema_version')}")
    return Matrix.from_rows(MatrixRow.from_dict(row) for row in data.get("rows", []))
