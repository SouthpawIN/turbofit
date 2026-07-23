from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.schema import Matrix, MatrixRow, load_matrix


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "references/main-aux-matrix.json"


def test_canonical_matrix_contains_exactly_75_unique_rows() -> None:
    matrix = load_matrix(MATRIX_PATH)

    assert len(matrix.rows) == 75
    assert len({row.id for row in matrix.rows}) == 75
    assert len({(row.main, row.aux, row.context) for row in matrix.rows}) == 75


def test_bonsai_never_appears_in_a_1m_row() -> None:
    matrix = load_matrix(MATRIX_PATH)

    one_million = [row for row in matrix.rows if row.context == 1_048_576]
    assert one_million
    assert all("Bonsai" not in row.main and "Bonsai" not in row.aux for row in one_million)


def test_contexts_are_normalized_integers() -> None:
    matrix = load_matrix(MATRIX_PATH)

    assert {row.context for row in matrix.rows} == {65_536, 131_072, 262_144, 1_048_576}
    assert all(isinstance(row.context, int) for row in matrix.rows)


def test_every_row_has_deterministic_id_and_method_priority() -> None:
    matrix = load_matrix(MATRIX_PATH)

    for row in matrix.rows:
        assert row.id == MatrixRow.make_id(row.main, row.aux, row.context)
        assert row.method_priority
        assert row.method_priority[0] == "dspark"
        assert row.status in {"pending", "success", "blocked"}


def test_duplicate_tuple_is_rejected() -> None:
    row = MatrixRow(
        id="a-b-64k",
        main="A",
        aux="B",
        context=65_536,
        status="pending",
        method_priority=("dspark", "mtp", "nextn"),
    )

    with pytest.raises(ValueError, match="duplicate matrix row"):
        Matrix(rows=(row, row))


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        MatrixRow(
            id="a-b-64k",
            main="A",
            aux="B",
            context=65_536,
            status="maybe",
            method_priority=("dspark",),
        )
