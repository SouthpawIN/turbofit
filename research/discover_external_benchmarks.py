#!/usr/bin/env python3
"""Ingest the canonical DeepSWE leaderboard as provenance-bound external evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Mapping


DEFAULT_URL = "https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json"
ROOT = Path(__file__).resolve().parents[1]
MODEL_ALIASES = {
    "glm-5-2": "glm-5-2-2-788bpw",
    "minimax-m3": "minimax-m3-q4",
    "deepseek-v4-flash": "deepseek-v4-flash-0731-q8-dspark",
    "deepseek-v4-flash-0731": "deepseek-v4-flash-0731-q8-dspark",
}


def normalize_deepswe(payload: Any, source_url: str, artifact_identity: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("DeepSWE payload must be an object")
    generated_at = payload.get("generated_at")
    task_count = payload.get("n_tasks_in_set")
    rows = payload.get("rows")
    if not isinstance(generated_at, str) or not generated_at or not isinstance(task_count, int) or task_count <= 0:
        raise ValueError("DeepSWE payload has invalid metadata")
    if not isinstance(rows, list) or not artifact_identity.startswith("sha256:") or len(artifact_identity) != 71:
        raise ValueError("DeepSWE rows or artifact identity are invalid")
    normalized = [_row(row) for row in rows]
    best: dict[str, dict[str, Any]] = {}
    for row in normalized:
        current = best.get(row["model"])
        rank = (row["pass_at_1"], -row["mean_cost_usd"], row["configuration"])
        current_rank = None if current is None else (
            current["pass_at_1"], -current["mean_cost_usd"], current["configuration"]
        )
        if current_rank is None or rank > current_rank:
            best[row["model"]] = row
    return {
        "schema": "turbofit.external-benchmarks/v1",
        "source": {
            "name": "DeepSWE",
            "url": source_url,
            "generated_at": generated_at,
            "task_count": task_count,
            "artifact_identity": artifact_identity,
        },
        "benchmarks": [best[model] for model in sorted(best)],
    }


def _row(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("DeepSWE row must be an object")
    model = _token(raw.get("model"), "model")
    harness = _token(raw.get("harness"), "harness")
    configuration = _token(raw.get("config"), "config")
    effort = raw.get("reasoning_effort")
    if effort is not None and not isinstance(effort, str):
        raise ValueError(f"invalid reasoning_effort for {model}")
    pass_at_1 = _fraction(raw.get("pass_at_1"), "pass_at_1", model)
    pass_at_4 = _fraction(raw.get("pass_at_4"), "pass_at_4", model)
    passed = _nonnegative_int(raw.get("n_passed"), "n_passed", model)
    attempted = _positive_int(raw.get("n_attempted"), "n_attempted", model)
    if not math.isclose(pass_at_1, passed / attempted, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"inconsistent pass_at_1 for {model}")
    return {
        "benchmark": "deep-swe-v1.1",
        "model": model,
        "turbofit_model": MODEL_ALIASES.get(model),
        "harness": harness,
        "reasoning_effort": effort,
        "configuration": configuration,
        "pass_at_1": pass_at_1,
        "pass_at_4": pass_at_4,
        "passed": passed,
        "attempted": attempted,
        "mean_cost_usd": _nonnegative_number(raw.get("mean_cost_usd"), "mean_cost_usd", model),
        "median_output_tokens": _nonnegative_number(raw.get("median_output_tokens"), "median_output_tokens", model),
        "median_peak_context_tokens": _nonnegative_number(
            raw.get("median_peak_context_tokens"), "median_peak_context_tokens", model
        ),
    }


def _token(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise ValueError(f"DeepSWE {field} must be a token")
    return value


def _fraction(value: Any, field: str, model: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"invalid {field} for {model}")
    return float(value)


def _nonnegative_number(value: Any, field: str, model: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"invalid {field} for {model}")
    return value


def _nonnegative_int(value: Any, field: str, model: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {field} for {model}")
    return value


def _positive_int(value: Any, field: str, model: str) -> int:
    result = _nonnegative_int(value, field, model)
    if result <= 0:
        raise ValueError(f"invalid {field} for {model}")
    return result


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "external-benchmarks.json")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url, timeout=30) as response:
        raw = response.read()
    identity = "sha256:" + hashlib.sha256(raw).hexdigest()
    payload = normalize_deepswe(json.loads(raw), args.url, identity)
    _write_atomic(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "rows": len(payload["benchmarks"]),
        "generated_at": payload["source"]["generated_at"],
        "artifact_identity": identity,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
