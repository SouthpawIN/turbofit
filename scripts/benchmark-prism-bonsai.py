#!/usr/bin/env python3
"""Benchmark Prism Bonsai baseline vs DSpark across supported context tiers."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

IMAGE = "turbofit-prism-bonsai:local"
PORT = 11606
CONTEXTS = (65536, 131072, 262144)
PROMPT = "Write a detailed implementation of merge sort in Python with type hints, docstrings, complexity analysis, and eight numbered design notes."
FAMILIES = {
    "ternary": {
        "root": "/home/sovthpaw/Models/storage/gguf/Ternary-Bonsai-27B",
        "model": "Ternary-Bonsai-27B-Q2_0.gguf",
        "draft": "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
    },
    "1bit": {
        "root": "/home/sovthpaw/Models/storage/gguf/Bonsai-27B",
        "model": "Bonsai-27B-Q1_0.gguf",
        "draft": "Bonsai-27B-dspark-Q4_1.gguf",
    },
}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def get_json(url: str, timeout: int = 10) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}


def launch(family: str, ctx: int, dspark: bool) -> str:
    cfg = FAMILIES[family]
    name = f"turbofit-{family}-{'dspark' if dspark else 'baseline'}"
    for old in (
        "turbofit-1bit-dspark", "turbofit-1bit-baseline",
        "turbofit-ternary-dspark", "turbofit-ternary-baseline",
    ):
        run("docker", "rm", "-f", old, check=False)
    cmd = [
        "docker", "run", "-d", "--name", name,
        "--gpus", "device=1", "--network", "host",
        "-e", f"PORT={PORT}", "-e", f"CTX={ctx}",
        "-e", f"MODEL=/models/{cfg['model']}",
        "-e", "MAIN_GPU=0", "-e", "NGL=99",
        "-v", f"{cfg['root']}:/models:ro", IMAGE,
    ]
    if dspark:
        cmd[cmd.index(IMAGE):cmd.index(IMAGE)] = [
            "-e", f"DRAFT_MODEL=/models/{cfg['draft']}",
            "-e", "DRAFT_NGL=99", "-e", "SPEC_DRAFT_N_MAX=4",
        ]
    run(*cmd)
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        status, body = get_json(f"http://127.0.0.1:{PORT}/health", timeout=2)
        if status == 200 and body.get("status") == "ok":
            return name
        time.sleep(2)
    logs = run("docker", "logs", "--tail", "200", name, check=False)
    raise RuntimeError(f"{name} failed health at {ctx}:\n{logs.stdout}\n{logs.stderr}")


def benchmark(model: str) -> dict:
    samples: list[list[str]] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            raw = run(
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,fan.speed",
                "--format=csv,noheader,nounits",
            ).stdout
            samples.append(raw.strip().splitlines())
            time.sleep(0.15)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    elapsed = time.monotonic() - started
    stop.set()
    thread.join()
    parsed = []
    for snapshot in samples:
        for line in snapshot:
            values = [part.strip() for part in line.split(",")]
            parsed.append({
                "gpu": int(values[0]), "used": int(values[1]), "free": int(values[2]),
                "util": int(values[3]), "power": float(values[4]), "fan": int(values[5]),
            })
    peak = {
        str(gpu): {
            "max_util_pct": max(row["util"] for row in parsed if row["gpu"] == gpu),
            "max_power_w": max(row["power"] for row in parsed if row["gpu"] == gpu),
            "max_fan_pct": max(row["fan"] for row in parsed if row["gpu"] == gpu),
            "max_used_mb": max(row["used"] for row in parsed if row["gpu"] == gpu),
        }
        for gpu in (0, 1)
    }
    return {
        "elapsed_s": round(elapsed, 3),
        "content": data["choices"][0]["message"].get("content", ""),
        "usage": data.get("usage", {}),
        "timings": data.get("timings", {}),
        "peak_gpu": peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    args = parser.parse_args()
    cfg = FAMILIES[args.family]
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "image": IMAGE,
        "model": f"{cfg['root']}/{cfg['model']}",
        "draft": f"{cfg['root']}/{cfg['draft']}",
        "contexts": [],
    }
    out_path = Path(f"/home/sovthpaw/projects/turbofit/references/results/{args.family}-dspark-sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for context in CONTEXTS:
        record = {"context": context, "runs": {}}
        for mode in ("baseline", "dspark"):
            name = launch(args.family, context, mode == "dspark")
            status, models = get_json(f"http://127.0.0.1:{PORT}/v1/models")
            meta = ((models.get("data") or [{}])[0].get("meta") or {})
            result = benchmark(cfg["model"])
            logs = run("docker", "logs", "--tail", "220", name, check=False).stdout
            result.update({
                "health_status": status,
                "server_context": meta.get("n_ctx"),
                "dspark_initialized": bool(result.get("timings", {}).get("draft_n", 0)) or "speculative decoding context initialized" in logs,
                "logs": logs,
            })
            record["runs"][mode] = result
        baseline_tps = float(record["runs"]["baseline"]["timings"].get("predicted_per_second", 0))
        dspark_tps = float(record["runs"]["dspark"]["timings"].get("predicted_per_second", 0))
        record["optimized_mode"] = "dspark" if dspark_tps > baseline_tps else "baseline"
        record["speedup"] = round(dspark_tps / baseline_tps, 4) if baseline_tps else 0
        record["passed"] = all(run_data["health_status"] == 200 and run_data["server_context"] == context and bool(run_data["content"]) for run_data in record["runs"].values()) and record["runs"]["dspark"]["dspark_initialized"]
        output["contexts"].append(record)
        out_path.write_text(json.dumps(output, indent=2))
        print(json.dumps({"family": args.family, "context": context, "passed": record["passed"], "baseline_tps": baseline_tps, "dspark_tps": dspark_tps, "speedup": record["speedup"], "optimized_mode": record["optimized_mode"]}), flush=True)
    output["passed"] = all(record["passed"] for record in output["contexts"])
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"passed": output["passed"], "output": str(out_path)}, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
