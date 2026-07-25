#!/usr/bin/env python3
"""Exercise Bonsai 1-bit main + Turbofit auto aux at every supported context."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

IMAGE = "turbofit-bonsai-1bit:local"
CONTAINER = "turbofit-bonsai-test"
MODEL_ROOT = "/home/sovthpaw/Models/storage/gguf/Bonsai-27B"
MODEL = "Bonsai-27B-Q1_0.gguf"
PORT = 11610
GATEWAY = "http://127.0.0.1:8091"
CONTEXTS = (65536, 131072, 262144)
OUT = Path("/home/sovthpaw/projects/turbofit/references/results/bonsai-auto-contexts.json")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def get(url: str, timeout: int = 10) -> tuple[int, dict, dict]:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}, dict(exc.headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}, {}


def post(url: str, marker: str, timeout: int = 180) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply exactly: {marker}"}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            headers = dict(response.headers)
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        data = {"error": exc.read().decode(errors="replace")}
        headers = dict(exc.headers)
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    candidate = first.get("message")
    message: dict[str, object] = candidate if isinstance(candidate, dict) else {}
    return {
        "status": status,
        "elapsed_s": round(time.monotonic() - started, 3),
        "backend": headers.get("X-Turbofit-Backend"),
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "usage": data.get("usage", {}),
        "timings": data.get("timings", {}),
        "raw": data,
    }


def gpu_snapshot() -> list[dict]:
    raw = run(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ).stdout
    fields = ("index", "name", "memory_used_mb", "memory_free_mb", "utilization_pct", "power_w")
    return [dict(zip(fields, (part.strip() for part in line.split(",")))) for line in raw.splitlines()]


def launch(ctx: int) -> None:
    run("docker", "rm", "-f", CONTAINER, check=False)
    run(
        "docker", "run", "-d", "--name", CONTAINER,
        "--gpus", "all", "-p", f"{PORT}:{PORT}",
        "-e", f"CTX={ctx}", "-e", f"PORT={PORT}",
        "-v", f"{MODEL_ROOT}:/models:ro", IMAGE,
    )
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        status, body, _ = get(f"http://127.0.0.1:{PORT}/health", timeout=2)
        if status == 200 and body.get("status") == "ok":
            return
        time.sleep(2)
    raise RuntimeError(f"container did not become healthy at context {ctx}:\n{run('docker', 'logs', '--tail', '120', CONTAINER, check=False).stdout}")


def main() -> int:
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": IMAGE,
        "container": CONTAINER,
        "model": f"{MODEL_ROOT}/{MODEL}",
        "contexts": [],
    }
    for ctx in CONTEXTS:
        launch(ctx)
        model_status, models, _ = get(f"http://127.0.0.1:{PORT}/v1/models")
        status_status, gateway_status, _ = get(f"{GATEWAY}/status")
        main = post(f"http://127.0.0.1:{PORT}/v1/chat/completions", f"BONSAI-MAIN-{ctx}-OK")
        aux = post(f"{GATEWAY}/aux/v1/chat/completions", f"BONSAI-AUTO-AUX-{ctx}-OK")
        logs = run("docker", "logs", "--tail", "160", CONTAINER, check=False)
        meta = (((models.get("data") or [{}])[0]).get("meta") or {})
        passed = (
            model_status == 200
            and int(meta.get("n_ctx", 0)) == ctx
            and main["status"] == 200
            and main["content"].strip() == f"BONSAI-MAIN-{ctx}-OK"
            and aux["status"] == 200
            and aux["content"].strip() == f"BONSAI-AUTO-AUX-{ctx}-OK"
            and aux["backend"] == "auto:bonsai-27b-1bit"
        )
        record = {
            "context": ctx,
            "passed": passed,
            "model_status": model_status,
            "server_context": meta.get("n_ctx"),
            "gateway_status_status": status_status,
            "gateway_status": gateway_status,
            "main": main,
            "aux": aux,
            "gpu": gpu_snapshot(),
            "container_logs": logs.stdout + logs.stderr,
        }
        results["contexts"].append(record)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))
        print(json.dumps({"context": ctx, "passed": passed, "main_tps": main["timings"].get("predicted_per_second"), "aux_tps": aux["timings"].get("predicted_per_second"), "backend": aux["backend"]}), flush=True)
    results["passed"] = all(record["passed"] for record in results["contexts"])
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps({"passed": results["passed"], "output": str(OUT)}, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
