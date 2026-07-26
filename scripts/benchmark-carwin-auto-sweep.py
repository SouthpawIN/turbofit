#!/usr/bin/env python3
"""Resumable Carwin Nano:auto matrix sweep with GPU-clear gates."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from matrix_utils import wait_for_gpu_clear

BIN = "/home/sovthpaw/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
MODEL = "/home/sovthpaw/Models/storage/gguf/Carwin-MoE-Nano/carwin-moe-Nano.gguf"
PORT = 11607
CONTEXTS = (65536, 131072, 262144, 1048576)
PROMPT = "Implement a bounded thread-safe queue in Python and explain six design choices."
STATE = Path.home() / ".config/turbofit/runtime-state.json"
OUT = Path("/home/sovthpaw/projects/turbofit/references/results/carwin-auto-matrix-sweep.json")
PROFILES = Path("/home/sovthpaw/projects/turbofit/references/successful-runtime-profiles.json")
CHECKLIST = Path("/home/sovthpaw/.hermes/wiki/topics/turbofit/main-aux-inference-checklist.md")
EVIDENCE = CHECKLIST.parent / "evidence"
ALIAS = "carwin-moe-nano"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def command(ctx: int) -> list[str]:
    return [
        BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
        "-c", str(ctx), "-ngl", "99", "--fit", "on", "-fa", "on",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1",
        "--spec-type", "draft-mtp",
    ]


def get_json(url: str, timeout: int = 10) -> tuple[int, dict, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.load(response), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}, {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}, {}


def wait_url(url: str, timeout: int = 360) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        code, body, _ = get_json(url, 3)
        last = body
        if code == 200:
            return body
        time.sleep(1)
    raise RuntimeError(f"readiness failed for {url}: {last}")


def start(ctx: int) -> tuple[subprocess.Popen[str], TextIO, dict]:
    log_path = OUT.parent / f"carwin-auto-{ctx}.log"
    log = log_path.open("w")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "0"
    process = subprocess.Popen(command(ctx), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        wait_url(f"http://127.0.0.1:{PORT}/health")
        _, models, _ = get_json(f"http://127.0.0.1:{PORT}/v1/models")
        data = (models.get("data") or [{}])[0]
        return process, log, {"model": data.get("id"), "context": (data.get("meta") or {}).get("n_ctx"), "log": str(log_path)}
    except Exception:
        stop_process(process, log)
        raise


def stop_process(process: subprocess.Popen[str] | None, log: TextIO | None) -> None:
    if process and process.poll() is None:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
    if log:
        log.close()


def route(ctx: int, pid: int) -> dict:
    label = {65536: "64k", 131072: "128k", 262144: "262k", 1048576: "1m"}[ctx]
    expected = {"main_alias": ALIAS, "aux_alias": f"auto:{ALIAS}", "aux_mode": "shared-main"}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "active": f"test:carwin-auto-{label}", "context": ctx, "expected": expected,
        "components": [{"role": "main", "kind": "process", "name": "carwin-auto-test", "pid": pid, "port": PORT}],
        "activating": True,
    }, indent=2) + "\n")
    run("systemctl", "--user", "restart", "turbofit-gateway.service")
    status = wait_url("http://127.0.0.1:8091/status", 60)
    if status.get("main", {}).get("alias") != ALIAS or status.get("aux", {}).get("alias") != f"auto:{ALIAS}":
        raise RuntimeError(f"route mismatch: {status}")
    return status


def infer(role: str) -> dict:
    payload = {
        "model": "auto", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:8091/{role}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response); headers = dict(response.headers)
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "backend": headers.get("X-Turbofit-Backend"),
        "content": body["choices"][0]["message"].get("content", ""),
        "usage": body.get("usage", {}), "timings": body.get("timings", {}),
    }


def monitor(stop: threading.Event, samples: list[list[str]]) -> None:
    while not stop.is_set():
        samples.append(run(
            "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,fan.speed",
            "--format=csv,noheader,nounits",
        ).stdout.strip().splitlines())
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


def reset_route() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"active": None, "components": [], "stopped_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    run("systemctl", "--user", "restart", "turbofit-gateway.service", check=False)


def publish(record: dict) -> None:
    if not record.get("passed"): return
    ctx = record["context"]
    label = {65536: "64K", 131072: "128K", 262144: "262K", 1048576: "1M"}[ctx]
    suffix = label.lower(); slug = f"carwin-nano-auto-{suffix}"; evidence_name = f"{slug}.md"
    path = EVIDENCE / evidence_name
    main, aux = record["results"]["main"], record["results"]["aux"]
    peak = record["peak_gpu"]["0"]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""---
title: Matrix evidence - Carwin Nano auto at {label}
created: 2026-07-23
updated: 2026-07-23
type: benchmark
tags: [turbofit, benchmark, inference, mtp]
---

# Matrix evidence: Carwin Nano:auto @ {label}

- Checklist row: [Carwin Nano:auto @ {label}](../main-aux-inference-checklist.md#{slug})
- Runtime profile: `turbofit-runtime use carwin-auto-{suffix}`
- Validated context: `{ctx}`

| Route | Decode | Draft accepted | Output |
|---|---:|---:|---:|
| Main | {main['timings'].get('predicted_per_second', 0):.2f} tok/s | {main['timings'].get('draft_n_accepted', 0)}/{main['timings'].get('draft_n', 0)} | {main['usage'].get('completion_tokens', 0)} tokens |
| Aux shared-main | {aux['timings'].get('predicted_per_second', 0):.2f} tok/s | {aux['timings'].get('draft_n_accepted', 0)}/{aux['timings'].get('draft_n', 0)} | {aux['usage'].get('completion_tokens', 0)} tokens |

Peak GPU 0: {peak['max_used_mb']} MiB, {peak['max_util_pct']}%, {peak['max_power_w']:.2f} W, fan {peak['max_fan_pct']}%.

**PASS.** Exact context, both gateway routes, MTP counters, output, telemetry, and post-run GPU clearing passed. GPU-clear event: `{record['gpu_clear_after']['timestamp']}`.
""")
    checklist = CHECKLIST.read_text()
    pending = f"- [ ] **Carwin Nano:auto @ {label} context**"
    passed = f"- [x] **Carwin Nano:auto @ {label} context** — [evidence](evidence/{evidence_name})"
    if pending in checklist: checklist = checklist.replace(pending, passed, 1)
    index = f"- [Carwin Nano:auto @ {label}](#{slug}) — `turbofit-runtime use carwin-auto-{suffix}`; [evidence](evidence/{evidence_name})."
    if index not in checklist: checklist = checklist.replace("### Success index\n\n", f"### Success index\n\n{index}\n", 1)
    CHECKLIST.write_text(checklist)

    manifest = json.loads(PROFILES.read_text()); name = f"carwin-auto-{suffix}"
    if name not in manifest["profiles"]:
        manifest["profiles"][name] = {
            "description": f"Carwin Nano MTP main with shared-main auto auxiliary at {label}",
            "context": ctx, "evidence": str(path),
            "expected": {"main_alias": ALIAS, "aux_alias": f"auto:{ALIAS}", "aux_mode": "shared-main"},
            "components": [{"role": "main", "kind": "process", "name": "turbofit-runtime-main", "gpu": "0", "port": PORT, "command": command(ctx)}],
        }
        PROFILES.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(OUT.read_text()) if OUT.exists() else {"pair": "Carwin Nano:auto", "contexts": {}}
    for ctx in CONTEXTS:
        if state["contexts"].get(str(ctx), {}).get("passed"):
            print(json.dumps({"context": ctx, "status": "already-passed"}), flush=True); continue
        wait_for_gpu_clear(label=f"carwin-auto-before-{ctx}")
        record = {"context": ctx, "timestamp": datetime.now(timezone.utc).isoformat(), "passed": False}
        process = None; log = None
        try:
            process, log, check = start(ctx)
            status = route(ctx, process.pid)
            samples: list[list[str]] = []; stop_event = threading.Event()
            mon = threading.Thread(target=monitor, args=(stop_event, samples), daemon=True); mon.start()
            results = {"main": infer("main"), "aux": infer("aux")}
            stop_event.set(); mon.join()
            record.update({"check": check, "gateway_status": status, "results": results, "peak_gpu": peak_gpu(samples)})
            record["passed"] = (
                check.get("context") == ctx
                and all(item.get("content") for item in results.values())
                and results["main"].get("backend") == ALIAS
                and results["aux"].get("backend") == f"auto:{ALIAS}"
                and all(item.get("timings", {}).get("draft_n", 0) > 0 for item in results.values())
            )
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            stop_process(process, log); reset_route()
            record["gpu_clear_after"] = wait_for_gpu_clear(label=f"carwin-auto-after-{ctx}")
            if record.get("check", {}).get("log"):
                try: record["logs"] = Path(record["check"]["log"]).read_text()
                except OSError: pass
            state["contexts"][str(ctx)] = record; OUT.write_text(json.dumps(state, indent=2)); publish(record)
            print(json.dumps({"context": ctx, "passed": record["passed"], "error": record.get("error")}), flush=True)
    return 0 if all(state["contexts"].get(str(ctx), {}).get("passed") for ctx in CONTEXTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
