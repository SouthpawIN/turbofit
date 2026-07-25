#!/usr/bin/env python3
"""Resumable Ternary Bonsai:1 Bit Bonsai matrix sweep."""
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
CONTEXTS = (65536, 131072, 262144)
MODES = {65536: "dspark", 131072: "dspark", 262144: "baseline"}
PROMPT = "Implement binary search in Python with type hints and explain six design decisions."
STATE = Path.home() / ".config/turbofit/runtime-state.json"
OUT = Path("/home/sovthpaw/projects/turbofit/references/results/ternary-1bit-matrix-sweep.json")
PROFILES = Path("/home/sovthpaw/projects/turbofit/references/successful-runtime-profiles.json")
CHECKLIST = Path("/home/sovthpaw/.hermes/wiki/topics/turbofit/main-aux-inference-checklist.md")
EVIDENCE = CHECKLIST.parent / "evidence"
COMPONENTS = {
    "main": {
        "name": "turbofit-ternary-main-test", "gpu": "1", "port": 11606,
        "root": Path("/home/sovthpaw/Models/storage/gguf/Ternary-Bonsai-27B"),
        "model": "Ternary-Bonsai-27B-Q2_0.gguf",
        "draft": "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
        "alias": "ternary-bonsai-27b-dspark",
    },
    "aux": {
        "name": "turbofit-1bit-aux-test", "gpu": "0", "port": 11610,
        "root": Path("/home/sovthpaw/Models/storage/gguf/Bonsai-27B"),
        "model": "Bonsai-27B-Q1_0.gguf",
        "draft": "Bonsai-27B-dspark-Q4_1.gguf",
        "alias": "bonsai-27b-1bit",
    },
}


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


def docker_command(role: str, ctx: int, mode: str, runtime_name: str | None = None) -> list[str]:
    spec = COMPONENTS[role]
    name = runtime_name or spec["name"]
    command = [
        "docker", "run", "-d", "--name", str(name),
        "--gpus", f"device={spec['gpu']}", "--network", "host",
        "-e", f"PORT={spec['port']}", "-e", f"CTX={ctx}",
        "-e", f"MODEL=/models/{spec['model']}", "-e", "MAIN_GPU=0", "-e", "NGL=99",
        "-v", f"{spec['root']}:/models:ro",
    ]
    if mode == "dspark":
        command.extend([
            "-e", f"DRAFT_MODEL=/models/{spec['draft']}",
            "-e", "DRAFT_NGL=99", "-e", "SPEC_DRAFT_N_MAX=4",
        ])
    command.append(IMAGE)
    return command


def start_pair(ctx: int, mode: str) -> dict:
    checks = {}
    for role, spec in COMPONENTS.items():
        run("docker", "rm", "-f", str(spec["name"]), check=False)
        run(*docker_command(role, ctx, mode))
    for role, spec in COMPONENTS.items():
        wait_url(f"http://127.0.0.1:{spec['port']}/health")
        _, models, _ = get_json(f"http://127.0.0.1:{spec['port']}/v1/models")
        data = (models.get("data") or [{}])[0]
        checks[role] = {"model": data.get("id"), "context": (data.get("meta") or {}).get("n_ctx")}
    return checks


def route_pair(ctx: int) -> dict:
    label = {65536: "64k", 131072: "128k", 262144: "262k"}[ctx]
    expected = {
        "main_alias": COMPONENTS["main"]["alias"],
        "aux_alias": COMPONENTS["aux"]["alias"],
        "aux_mode": "dedicated",
    }
    components = [
        {"role": role, "kind": "docker", "name": spec["name"], "port": spec["port"]}
        for role, spec in COMPONENTS.items()
    ]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "active": f"test:ternary-1bit-{label}", "context": ctx,
        "expected": expected, "components": components, "activating": True,
    }, indent=2) + "\n")
    run("systemctl", "--user", "restart", "turbofit-gateway.service")
    status = wait_url("http://127.0.0.1:8091/status", 60)
    if status.get("main", {}).get("alias") != expected["main_alias"]:
        raise RuntimeError(f"main route mismatch: {status}")
    if status.get("aux", {}).get("alias") != expected["aux_alias"]:
        raise RuntimeError(f"aux route mismatch: {status}")
    return status


def infer(role: str, output: dict) -> None:
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
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.load(response)
            headers = dict(response.headers)
        output[role] = {
            "elapsed_s": round(time.monotonic() - started, 3),
            "backend": headers.get("X-Turbofit-Backend"),
            "content": body["choices"][0]["message"].get("content", ""),
            "usage": body.get("usage", {}), "timings": body.get("timings", {}),
        }
    except Exception as exc:
        output[role] = {"error": repr(exc), "elapsed_s": round(time.monotonic() - started, 3)}


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


def cleanup(label: str) -> dict:
    for spec in COMPONENTS.values():
        run("docker", "rm", "-f", str(spec["name"]), check=False)
    clear = wait_for_gpu_clear(label=label)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"active": None, "components": [], "stopped_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    run("systemctl", "--user", "restart", "turbofit-gateway.service", check=False)
    return clear


def runtime_component(role: str, ctx: int, mode: str) -> dict:
    spec = COMPONENTS[role]
    env = {
        "PORT": str(spec["port"]), "CTX": str(ctx),
        "MODEL": f"/models/{spec['model']}", "MAIN_GPU": "0", "NGL": "99",
    }
    if mode == "dspark":
        env.update({"DRAFT_MODEL": f"/models/{spec['draft']}", "DRAFT_NGL": "99", "SPEC_DRAFT_N_MAX": "4"})
    return {
        "role": role, "kind": "docker", "name": f"turbofit-runtime-{role}",
        "image": IMAGE, "gpu": f"device={spec['gpu']}", "port": spec["port"],
        "mounts": [f"{spec['root']}:/models:ro"], "environment": env,
    }


def publish(record: dict) -> None:
    if not record.get("passed"):
        return
    ctx = record["context"]
    label = {65536: "64K", 131072: "128K", 262144: "262K"}[ctx]
    suffix = label.lower()
    slug = f"ternary-bonsai-1-bit-bonsai-{suffix}"
    evidence_name = f"{slug}.md"
    evidence_path = EVIDENCE / evidence_name
    main, aux = record["results"]["main"], record["results"]["aux"]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(f"""---
title: Matrix evidence - Ternary Bonsai with 1 Bit Bonsai at {label}
created: 2026-07-23
updated: 2026-07-23
type: benchmark
tags: [turbofit, benchmark, inference, dspark]
---

# Matrix evidence: Ternary Bonsai:1 Bit Bonsai @ {label}

- Checklist row: [Ternary Bonsai:1 Bit Bonsai @ {label}](../main-aux-inference-checklist.md#{slug})
- Runtime profile: `turbofit-runtime use ternary-1bit-{suffix}`
- Validated context: `{ctx}` on both runtimes
- Optimized mode: `{record['mode']}`

| Role | Backend | Decode | Draft accepted | Peak VRAM |
|---|---|---:|---:|---:|
| Main | `{main['backend']}` | {main['timings'].get('predicted_per_second', 0):.2f} tok/s | {main['timings'].get('draft_n_accepted', 0)}/{main['timings'].get('draft_n', 0)} | {record['peak_gpu']['1']['max_used_mb']} MiB |
| Aux | `{aux['backend']}` | {aux['timings'].get('predicted_per_second', 0):.2f} tok/s | {aux['timings'].get('draft_n_accepted', 0)}/{aux['timings'].get('draft_n', 0)} | {record['peak_gpu']['0']['max_used_mb']} MiB |

## Gate

**PASS.** Both isolated GPU runtimes launched at the exact context, routed through Turbofit, produced concurrent non-empty output, reported expected speculative counters, and were fully removed before the next configuration. GPU-clear event: `{record['gpu_clear_after']['timestamp']}`.
""")
    checklist = CHECKLIST.read_text()
    pending = f"- [ ] **Ternary Bonsai:1 Bit Bonsai @ {label} context**"
    passed = f"- [x] **Ternary Bonsai:1 Bit Bonsai @ {label} context** — [evidence](evidence/{evidence_name})"
    if pending in checklist:
        checklist = checklist.replace(pending, passed, 1)
    index = f"- [Ternary Bonsai:1 Bit Bonsai @ {label}](#{slug}) — `turbofit-runtime use ternary-1bit-{suffix}`; [evidence](evidence/{evidence_name})."
    if index not in checklist:
        checklist = checklist.replace("### Success index\n\n", f"### Success index\n\n{index}\n", 1)
    CHECKLIST.write_text(checklist)

    manifest = json.loads(PROFILES.read_text())
    name = f"ternary-1bit-{suffix}"
    if name not in manifest["profiles"]:
        manifest["profiles"][name] = {
            "description": f"Ternary Bonsai main with 1 Bit Bonsai auxiliary at {label}",
            "context": ctx, "evidence": str(evidence_path),
            "expected": {
                "main_alias": COMPONENTS["main"]["alias"],
                "aux_alias": COMPONENTS["aux"]["alias"], "aux_mode": "dedicated",
            },
            "components": [runtime_component("main", ctx, record["mode"]), runtime_component("aux", ctx, record["mode"])],
        }
        PROFILES.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(OUT.read_text()) if OUT.exists() else {"pair": "Ternary Bonsai:1 Bit Bonsai", "contexts": {}}
    for ctx in CONTEXTS:
        if state["contexts"].get(str(ctx), {}).get("passed"):
            print(json.dumps({"context": ctx, "status": "already-passed"}), flush=True)
            continue
        cleanup(f"ternary-1bit-before-{ctx}")
        record = {"context": ctx, "mode": MODES[ctx], "timestamp": datetime.now(timezone.utc).isoformat(), "passed": False}
        try:
            checks = start_pair(ctx, MODES[ctx])
            status = route_pair(ctx)
            samples: list[list[str]] = []
            stop_event = threading.Event()
            mon = threading.Thread(target=monitor, args=(stop_event, samples), daemon=True)
            results: dict = {}
            workers = [threading.Thread(target=infer, args=(role, results)) for role in ("main", "aux")]
            mon.start()
            for worker in workers: worker.start()
            for worker in workers: worker.join()
            stop_event.set(); mon.join()
            logs = {role: run("docker", "logs", "--tail", "240", str(spec["name"]), check=False).stdout for role, spec in COMPONENTS.items()}
            record.update({"checks": checks, "gateway_status": status, "results": results, "peak_gpu": peak_gpu(samples), "logs": logs})
            draft_ok = MODES[ctx] == "baseline" or all(results.get(role, {}).get("timings", {}).get("draft_n", 0) > 0 for role in COMPONENTS)
            record["passed"] = (
                all(checks[role].get("context") == ctx for role in COMPONENTS)
                and all(results.get(role, {}).get("content") for role in COMPONENTS)
                and results["main"].get("backend") == COMPONENTS["main"]["alias"]
                and results["aux"].get("backend") == COMPONENTS["aux"]["alias"]
                and draft_ok
            )
        except Exception as exc:
            record["error"] = repr(exc)
        finally:
            record["gpu_clear_after"] = cleanup(f"ternary-1bit-after-{ctx}")
            state["contexts"][str(ctx)] = record
            OUT.write_text(json.dumps(state, indent=2))
            publish(record)
            print(json.dumps({"context": ctx, "mode": MODES[ctx], "passed": record["passed"], "error": record.get("error")}), flush=True)
    return 0 if all(state["contexts"].get(str(ctx), {}).get("passed") for ctx in CONTEXTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
