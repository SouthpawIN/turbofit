#!/usr/bin/env python3
"""Compatibility wrappers around the canonical Turbofit GPU primitives."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turbofit_runtime.gpu import GPUClearGate, probe_gpus  # noqa: E402

GPU_CLEAR_LOG = ROOT / "references/results/gpu-clear-events.jsonl"


def gpu_snapshot() -> list[dict[str, Any]]:
    return [sample.to_dict() for sample in probe_gpus()]


def compute_processes() -> list[dict[str, Any]]:
    result = subprocess.run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",", 3)]
        rows.append({
            "gpu_uuid": values[0],
            "pid": int(values[1]),
            "used_mb": int(values[2]),
            "process": values[3],
        })
    return rows


def wait_for_gpu_clear(
    *,
    max_used_mb: dict[int, int] | None = None,
    timeout: int = 180,
    settle_samples: int = 3,
    label: str = "between-configurations",
) -> dict[str, Any]:
    event = GPUClearGate().wait(
        ceilings_mb=max_used_mb or {0: 1024, 1: 1024},
        timeout_s=timeout,
        settle_samples=settle_samples,
        label=label,
    )
    payload = event.to_dict()
    payload["compute_processes"] = compute_processes()
    GPU_CLEAR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GPU_CLEAR_LOG.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    return payload
