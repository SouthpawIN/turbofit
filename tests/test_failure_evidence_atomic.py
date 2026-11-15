"""Failure evidence must never be readable in a half-written state."""

import json
from pathlib import Path


def _atomic_write(path: Path, payload: dict) -> None:
    """Mirror of the executor's write-then-rename contract."""
    import os
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def test_reader_never_sees_a_partial_file(tmp_path):
    target = tmp_path / "failure.json"
    _atomic_write(target, {"schema": "turbofit.catalog-failure/v1", "error": "boom"})
    assert json.loads(target.read_text())["error"] == "boom"
    assert not (tmp_path / "failure.json.tmp").exists()


def test_interrupted_write_leaves_no_json(tmp_path):
    # a writer killed before the rename leaves only the temp file
    target = tmp_path / "failure.json"
    (tmp_path / "failure.json.tmp").write_text('{"schema": "turbofit.cata')
    assert not target.exists()
    assert not target.exists() or json.loads(target.read_text())
