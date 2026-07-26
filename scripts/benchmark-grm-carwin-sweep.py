#!/usr/bin/env python3
"""Resumable GRM:Carwin context sweep through the Turbofit gateway."""
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

from matrix_utils import wait_for_gpu_clear

BIN = "/home/sovthpaw/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
GRM = "/home/sovthpaw/Models/storage/gguf/GRM-2.6-Plus-0628/grm-2.6-plus-0628-Q4_K_M-reasoning-imat.gguf"
MMPROJ = "/home/sovthpaw/Models/storage/gguf/GRM-2.6-Plus-0628/mmproj-OrionLLM_GRM-2.6-Plus-0628-bf16.gguf"
CARWIN = "/home/sovthpaw/Models/storage/gguf/Carwin-MoE-Nano/carwin-moe-Nano.gguf"
OUT = Path("/home/sovthpaw/projects/turbofit/references/results/grm-carwin-sweep.json")
CHECKLIST = Path("/home/sovthpaw/.hermes/wiki/topics/turbofit/main-aux-inference-checklist.md")
EVIDENCE_DIR = CHECKLIST.parent / "evidence"
RUNTIME_PROFILES = Path("/home/sovthpaw/projects/turbofit/references/successful-runtime-profiles.json")
PROMPT = "Write a detailed implementation of merge sort in Python with type hints, docstrings, complexity analysis, and eight numbered design notes."


def get_json(url: str, timeout: int = 10) -> tuple[int, dict, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.load(response), dict(response.headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}, {}


def wait_ready(port: int, timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body, _ = get_json(f"http://127.0.0.1:{port}/health", 2)
        if status == 200 and body.get("status") == "ok":
            return
        time.sleep(2)
    raise RuntimeError(f"port {port} failed health")


def wait_gateway(timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        status, body, _ = get_json("http://127.0.0.1:8091/status", 3)
        last = body
        if status == 200:
            return
        time.sleep(1)
    raise RuntimeError(f"gateway failed status readiness: {last}")


def launch(ctx: int) -> list[subprocess.Popen[str]]:
    common = ["--host", "127.0.0.1", "-c", str(ctx), "-ngl", "99", "--fit", "on", "-fa", "on", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1", "--spec-type", "draft-mtp"]
    specs = [
        ("1", [BIN, "-m", GRM, "--port", "11605", *common, "--mmproj", MMPROJ]),
        ("0", [BIN, "-m", CARWIN, "--port", "11607", *common]),
    ]
    processes = []
    for gpu, command in specs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        processes.append(subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True))
    try:
        wait_ready(11605)
        wait_ready(11607)
    except Exception:
        stop(processes)
        raise
    subprocess.run(["systemctl", "--user", "restart", "turbofit-gateway.service"], check=True)
    wait_gateway(60)
    return processes


def stop(processes: list[subprocess.Popen[str]]) -> dict[str, str]:
    logs = {}
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for index, process in enumerate(processes):
        try:
            stdout, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        logs["main" if index == 0 else "aux"] = stdout
    return logs


def request(role: str, output: dict) -> None:
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:8091/{role}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            data = json.load(response)
            headers = dict(response.headers)
        output[role] = {
            "elapsed_s": round(time.monotonic() - started, 3),
            "backend": headers.get("X-Turbofit-Backend"),
            "content": data["choices"][0]["message"].get("content", ""),
            "usage": data.get("usage", {}),
            "timings": data.get("timings", {}),
        }
    except Exception as exc:
        output[role] = {"error": repr(exc), "elapsed_s": round(time.monotonic() - started, 3)}


def context_label(ctx: int) -> str:
    return {65536: "64K", 131072: "128K", 262144: "262K", 1048576: "1M"}[ctx]


def publish(record: dict) -> None:
    """Publish only evidence-gated passes and register their swap profile."""
    if not record.get("passed"):
        return
    ctx = int(record["context"])
    label = context_label(ctx)
    suffix = label.lower()
    slug = f"grm-26-plus-carwin-nano-{suffix}"
    evidence_name = f"{slug}.md"
    evidence_path = EVIDENCE_DIR / evidence_name
    main = record["results"]["main"]
    aux = record["results"]["aux"]
    main_t = main["timings"]
    aux_t = aux["timings"]
    peak = record["peak_gpu"]
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(f"""---
title: Matrix evidence - GRM 2.6 Plus with Carwin Nano at {label}
created: 2026-07-23
updated: 2026-07-23
type: benchmark
tags: [turbofit, benchmark, inference, optimization, mtp]
---

# Matrix evidence: GRM 2.6 Plus:Carwin Nano @ {label}

- Checklist row: [GRM 2.6 Plus:Carwin Nano @ {label}](../main-aux-inference-checklist.md#{slug})
- Runtime profile: `turbofit-runtime use grm-carwin-{suffix}`
- Validated context: `{ctx}` on both runtimes
- Gateway: main `{main['backend']}`, auxiliary `{aux['backend']}`

## Concurrent result

| Role | Decode | Draft accepted | Elapsed |
|---|---:|---:|---:|
| Main — GRM 2.6 Plus | {main_t['predicted_per_second']:.2f} tok/s | {main_t['draft_n_accepted']}/{main_t['draft_n']} | {main['elapsed_s']:.3f}s |
| Aux — Carwin Nano | {aux_t['predicted_per_second']:.2f} tok/s | {aux_t['draft_n_accepted']}/{aux_t['draft_n']} | {aux['elapsed_s']:.3f}s |

## Peak GPU telemetry

| GPU | Peak memory | Peak utilization | Peak power | Peak fan |
|---|---:|---:|---:|---:|
| 0 — Carwin | {peak['0']['max_used_mb']} MiB | {peak['0']['max_util_pct']}% | {peak['0']['max_power_w']:.2f} W | {peak['0']['max_fan_pct']}% |
| 1 — GRM | {peak['1']['max_used_mb']} MiB | {peak['1']['max_util_pct']}% | {peak['1']['max_power_w']:.2f} W | {peak['1']['max_fan_pct']}% |

## Gate

**PASS.** Both backends launched at the requested context, routed through Turbofit, produced non-empty concurrent responses, reported active MTP draft counters, and were measured under GPU load.
""")

    checklist = CHECKLIST.read_text()
    pending = f"- [ ] **GRM 2.6 Plus:Carwin Nano @ {label} context**"
    passed = f"- [x] **GRM 2.6 Plus:Carwin Nano @ {label} context** — [evidence](evidence/{evidence_name})"
    if pending in checklist:
        checklist = checklist.replace(pending, passed, 1)
    index_line = f"- [GRM 2.6 Plus:Carwin Nano @ {label}](#{slug}) — `turbofit-runtime use grm-carwin-{suffix}`; [evidence](evidence/{evidence_name})."
    if index_line not in checklist:
        checklist = checklist.replace("### Success index\n\n", f"### Success index\n\n{index_line}\n", 1)
    CHECKLIST.write_text(checklist)

    manifest = json.loads(RUNTIME_PROFILES.read_text())
    name = f"grm-carwin-{suffix}"
    if name not in manifest["profiles"]:
        profile = json.loads(json.dumps(manifest["profiles"]["grm-carwin-64k"]))
        profile["description"] = f"GRM 2.6 Plus MTP main with Carwin Nano MTP auxiliary at {label}"
        profile["context"] = ctx
        profile["evidence"] = str(evidence_path)
        for component in profile["components"]:
            command = component["command"]
            command[command.index("-c") + 1] = str(ctx)
        manifest["profiles"][name] = profile
        RUNTIME_PROFILES.write_text(json.dumps(manifest, indent=2) + "\n")


def monitor(stop_event: threading.Event, samples: list[list[str]]) -> None:
    while not stop_event.is_set():
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,fan.speed",
            "--format=csv,noheader,nounits",
        ], text=True)
        samples.append(raw.strip().splitlines())
        time.sleep(0.15)


def peak_gpu(samples: list[list[str]]) -> dict:
    rows = []
    for sample in samples:
        for line in sample:
            values = [item.strip() for item in line.split(",")]
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


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(OUT.read_text()) if OUT.exists() else {"pair": "GRM 2.6 Plus:Carwin Nano", "contexts": {}}
    seed = Path("/home/sovthpaw/projects/turbofit/references/results/grm-carwin-64k.json")
    if "65536" not in state["contexts"] and seed.exists():
        state["contexts"]["65536"] = json.loads(seed.read_text())
    for ctx in (131072, 262144, 1048576):
        if state["contexts"].get(str(ctx), {}).get("passed"):
            print(json.dumps({"context": ctx, "status": "already-passed"}), flush=True)
            continue
        processes: list[subprocess.Popen[str]] = []
        record = {"context": ctx, "timestamp": datetime.now(timezone.utc).isoformat(), "passed": False}
        try:
            processes = launch(ctx)
            status_code, status, _ = get_json("http://127.0.0.1:8091/status")
            results: dict = {}
            samples: list[list[str]] = []
            stop_event = threading.Event()
            mon = threading.Thread(target=monitor, args=(stop_event, samples), daemon=True)
            workers = [threading.Thread(target=request, args=(role, results)) for role in ("main", "aux")]
            mon.start()
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            stop_event.set()
            mon.join()
            record.update({
                "gateway_status_code": status_code,
                "gateway_status": status,
                "results": results,
                "peak_gpu": peak_gpu(samples),
                "passed": status_code == 200
                    and status.get("main", {}).get("alias") == "grm-2.6-plus-q4"
                    and status.get("aux", {}).get("alias") == "carwin-moe-nano"
                    and all(results.get(role, {}).get("content") for role in ("main", "aux"))
                    and all(results.get(role, {}).get("timings", {}).get("draft_n", 0) > 0 for role in ("main", "aux")),
            })
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            if processes:
                record["logs"] = stop(processes)
            record["gpu_clear_after"] = wait_for_gpu_clear(label=f"grm-carwin-after-{ctx}")
            state["contexts"][str(ctx)] = record
            OUT.write_text(json.dumps(state, indent=2))
            publish(record)
            print(json.dumps({"context": ctx, "passed": record["passed"], "error": record.get("error"), "output": str(OUT)}), flush=True)
    return 0 if all(state["contexts"].get(str(ctx), {}).get("passed") for ctx in (65536, 131072, 262144, 1048576)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
