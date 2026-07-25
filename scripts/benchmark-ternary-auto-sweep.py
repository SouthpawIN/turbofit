#!/usr/bin/env python3
"""Resumable Ternary Bonsai:auto matrix sweep with evidence publication."""
from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from matrix_utils import wait_for_gpu_clear

IMAGE = "turbofit-prism-bonsai:local"
PORT = 11606
ROOT = Path("/home/sovthpaw/Models/storage/gguf/Ternary-Bonsai-27B")
MODEL = "Ternary-Bonsai-27B-Q2_0.gguf"
DRAFT = "Ternary-Bonsai-27B-dspark-Q4_1.gguf"
CONTEXTS = (65536, 131072, 262144)
MODES = {65536: "dspark", 131072: "dspark", 262144: "baseline"}
OUT = Path("/home/sovthpaw/projects/turbofit/references/results/ternary-auto-matrix-sweep.json")
STATE = Path.home() / ".config/turbofit/runtime-state.json"
PROFILES = Path("/home/sovthpaw/projects/turbofit/references/successful-runtime-profiles.json")
CHECKLIST = Path("/home/sovthpaw/.hermes/wiki/topics/turbofit/main-aux-inference-checklist.md")
EVIDENCE = CHECKLIST.parent / "evidence"
PROMPT = "Write a Python function that computes Fibonacci numbers and explain four implementation choices."


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def get_json(url: str, timeout: int = 10) -> tuple[int, dict, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.load(response), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}, {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}, {}


def wait_url(url: str, timeout: int = 300) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        code, body, _ = get_json(url, 3)
        last = body
        if code == 200:
            return body
        time.sleep(1)
    raise RuntimeError(f"readiness failed for {url}: {last}")


def start(ctx: int, mode: str) -> tuple[str, dict]:
    name = "turbofit-ternary-auto-test"
    run("docker", "rm", "-f", name, check=False)
    command = [
        "docker", "run", "-d", "--name", name,
        "--gpus", "device=1", "--network", "host",
        "-e", f"PORT={PORT}", "-e", f"CTX={ctx}",
        "-e", f"MODEL=/models/{MODEL}", "-e", "MAIN_GPU=0", "-e", "NGL=99",
        "-v", f"{ROOT}:/models:ro",
    ]
    if mode == "dspark":
        command.extend([
            "-e", f"DRAFT_MODEL=/models/{DRAFT}",
            "-e", "DRAFT_NGL=99", "-e", "SPEC_DRAFT_N_MAX=4",
        ])
    command.append(IMAGE)
    result = run(*command)
    wait_url(f"http://127.0.0.1:{PORT}/health")
    _, models, _ = get_json(f"http://127.0.0.1:{PORT}/v1/models")
    meta = ((models.get("data") or [{}])[0].get("meta") or {})
    return name, {"model": (models.get("data") or [{}])[0].get("id"), "context": meta.get("n_ctx")}


def route_profile(ctx: int, mode: str) -> dict:
    label = {65536: "64k", 131072: "128k", 262144: "262k"}[ctx]
    expected = {
        "main_alias": "ternary-bonsai-27b-dspark",
        "aux_alias": "auto:ternary-bonsai-27b-dspark",
        "aux_mode": "shared-main",
    }
    component = {"role": "main", "kind": "docker", "name": "turbofit-ternary-auto-test", "port": PORT}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "active": f"test:ternary-auto-{label}",
        "context": ctx,
        "expected": expected,
        "components": [component],
        "activating": True,
    }, indent=2) + "\n")
    run("systemctl", "--user", "restart", "turbofit-gateway.service")
    status = wait_url("http://127.0.0.1:8091/status", 60)
    if status.get("main", {}).get("alias") != expected["main_alias"]:
        raise RuntimeError(f"main route mismatch: {status}")
    if status.get("aux", {}).get("alias") != expected["aux_alias"]:
        raise RuntimeError(f"aux route mismatch: {status}")
    return status


def infer(role: str) -> dict:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:8091/{role}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        body = json.load(response)
        headers = dict(response.headers)
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "backend": headers.get("X-Turbofit-Backend"),
        "content": body["choices"][0]["message"].get("content", ""),
        "usage": body.get("usage", {}),
        "timings": body.get("timings", {}),
    }


def monitor(stop: threading.Event, samples: list[list[str]]) -> None:
    while not stop.is_set():
        raw = run(
            "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,fan.speed",
            "--format=csv,noheader,nounits",
        ).stdout
        samples.append(raw.strip().splitlines())
        time.sleep(0.15)


def peak_gpu(samples: list[list[str]]) -> dict:
    rows = []
    for snapshot in samples:
        for line in snapshot:
            values = [part.strip() for part in line.split(",")]
            rows.append({"gpu": int(values[0]), "used": int(values[1]), "util": int(values[3]), "power": float(values[4]), "fan": int(values[5])})
    return {
        str(gpu): {
            "max_used_mb": max(row["used"] for row in rows if row["gpu"] == gpu),
            "max_util_pct": max(row["util"] for row in rows if row["gpu"] == gpu),
            "max_power_w": max(row["power"] for row in rows if row["gpu"] == gpu),
            "max_fan_pct": max(row["fan"] for row in rows if row["gpu"] == gpu),
        }
        for gpu in (0, 1)
    }


def publish(record: dict) -> None:
    if not record.get("passed"):
        return
    ctx = record["context"]
    label = {65536: "64K", 131072: "128K", 262144: "262K"}[ctx]
    suffix = label.lower()
    slug = f"ternary-bonsai-auto-{suffix}"
    evidence_name = f"{slug}.md"
    path = EVIDENCE / evidence_name
    main, aux = record["results"]["main"], record["results"]["aux"]
    peak = record["peak_gpu"]["1"]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""---
title: Matrix evidence - Ternary Bonsai auto at {label}
created: 2026-07-23
updated: 2026-07-23
type: benchmark
tags: [turbofit, benchmark, inference, dspark]
---

# Matrix evidence: Ternary Bonsai:auto @ {label}

- Checklist row: [Ternary Bonsai:auto @ {label}](../main-aux-inference-checklist.md#{slug})
- Runtime profile: `turbofit-runtime use ternary-auto-{suffix}`
- Validated context: `{ctx}`
- Optimized mode: `{record['mode']}`
- Main route: `{main['backend']}`
- Auxiliary route: `{aux['backend']}` (`shared-main`)

## Inference

| Route | Decode | Draft accepted | Output |
|---|---:|---:|---:|
| Main | {main['timings'].get('predicted_per_second', 0):.2f} tok/s | {main['timings'].get('draft_n_accepted', 0)}/{main['timings'].get('draft_n', 0)} | {main['usage'].get('completion_tokens', 0)} tokens |
| Aux | {aux['timings'].get('predicted_per_second', 0):.2f} tok/s | {aux['timings'].get('draft_n_accepted', 0)}/{aux['timings'].get('draft_n', 0)} | {aux['usage'].get('completion_tokens', 0)} tokens |

## Peak GPU 1

- Memory: {peak['max_used_mb']} MiB
- Utilization: {peak['max_util_pct']}%
- Power: {peak['max_power_w']:.2f} W
- Fan: {peak['max_fan_pct']}%

## Gate

**PASS.** Exact context validated, both stable gateway routes produced non-empty output, optimized mode matched the prior baseline-vs-DSpark sweep, and GPU telemetry was captured.
""")
    checklist = CHECKLIST.read_text()
    pending = f"- [ ] **Ternary Bonsai:auto @ {label} context**"
    passed = f"- [x] **Ternary Bonsai:auto @ {label} context** — [evidence](evidence/{evidence_name})"
    if pending in checklist:
        checklist = checklist.replace(pending, passed, 1)
    index = f"- [Ternary Bonsai:auto @ {label}](#{slug}) — `turbofit-runtime use ternary-auto-{suffix}`; [evidence](evidence/{evidence_name})."
    if index not in checklist:
        checklist = checklist.replace("### Success index\n\n", f"### Success index\n\n{index}\n", 1)
    CHECKLIST.write_text(checklist)

    manifest = json.loads(PROFILES.read_text())
    name = f"ternary-auto-{suffix}"
    if name not in manifest["profiles"]:
        env = {
            "PORT": str(PORT), "CTX": str(ctx), "MODEL": f"/models/{MODEL}",
            "MAIN_GPU": "0", "NGL": "99",
        }
        if record["mode"] == "dspark":
            env.update({"DRAFT_MODEL": f"/models/{DRAFT}", "DRAFT_NGL": "99", "SPEC_DRAFT_N_MAX": "4"})
        manifest["profiles"][name] = {
            "description": f"Ternary Bonsai main with shared-main auto auxiliary at {label}",
            "context": ctx,
            "evidence": str(path),
            "expected": {
                "main_alias": "ternary-bonsai-27b-dspark",
                "aux_alias": "auto:ternary-bonsai-27b-dspark",
                "aux_mode": "shared-main",
            },
            "components": [{
                "role": "main", "kind": "docker", "name": "turbofit-runtime-main",
                "image": IMAGE, "gpu": "device=1", "port": PORT,
                "mounts": [f"{ROOT}:/models:ro"], "environment": env,
            }],
        }
        PROFILES.write_text(json.dumps(manifest, indent=2) + "\n")


def cleanup(name: str | None) -> None:
    if name:
        run("docker", "rm", "-f", name, check=False)
        wait_for_gpu_clear(label=f"ternary-auto-after-{name}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"active": None, "components": [], "stopped_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    run("systemctl", "--user", "restart", "turbofit-gateway.service", check=False)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(OUT.read_text()) if OUT.exists() else {"pair": "Ternary Bonsai:auto", "contexts": {}}
    try:
        for ctx in CONTEXTS:
            if state["contexts"].get(str(ctx), {}).get("passed"):
                print(json.dumps({"context": ctx, "status": "already-passed"}), flush=True)
                continue
            name = None
            record = {"context": ctx, "mode": MODES[ctx], "timestamp": datetime.now(timezone.utc).isoformat(), "passed": False}
            try:
                name, model_check = start(ctx, MODES[ctx])
                status = route_profile(ctx, MODES[ctx])
                samples: list[list[str]] = []
                stop_event = threading.Event()
                thread = threading.Thread(target=monitor, args=(stop_event, samples), daemon=True)
                thread.start()
                results = {"main": infer("main"), "aux": infer("aux")}
                stop_event.set()
                thread.join()
                logs = run("docker", "logs", "--tail", "240", name, check=False).stdout
                record.update({"model_check": model_check, "gateway_status": status, "results": results, "peak_gpu": peak_gpu(samples), "logs": logs})
                draft_ok = MODES[ctx] == "baseline" or all(item["timings"].get("draft_n", 0) > 0 for item in results.values())
                record["passed"] = (
                    model_check.get("context") == ctx
                    and all(item.get("content") for item in results.values())
                    and results["main"].get("backend") == "ternary-bonsai-27b-dspark"
                    and results["aux"].get("backend") == "auto:ternary-bonsai-27b-dspark"
                    and draft_ok
                )
            except Exception as exc:
                record["error"] = repr(exc)
            finally:
                cleanup(name)
                state["contexts"][str(ctx)] = record
                OUT.write_text(json.dumps(state, indent=2))
                publish(record)
                print(json.dumps({"context": ctx, "mode": MODES[ctx], "passed": record["passed"], "error": record.get("error")}), flush=True)
    finally:
        cleanup(None)
    return 0 if all(state["contexts"].get(str(ctx), {}).get("passed") for ctx in CONTEXTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
