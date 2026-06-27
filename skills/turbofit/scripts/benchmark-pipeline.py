#!/usr/bin/env python3
"""
turbofit benchmark pipeline — benchmarks all catalog models and publishes to GitHub.

Reads the turbofit catalog (~/.config/turbofit/models.yaml), launches each model
one at a time on a dedicated port, runs a prompt, measures tok/s, writes results
to references/benchmark-results.json, and optionally commits + pushes to GitHub.

Usage:
  python3 benchmark-pipeline.py                    # benchmark all, write JSON
  python3 benchmark-pipeline.py --push              # benchmark + push to GitHub
  python3 benchmark-pipeline.py --filter "27b"      # only benchmark models matching filter
  python3 benchmark-pipeline.py --gpu 1             # use specific GPU
  python3 benchmark-pipeline.py --ctx 65536         # use smaller ctx for faster bench

Installs of turbofit can pull the latest results:
  serve bench pull    # downloads latest benchmark-results.json from GitHub
"""

import json
import time
import os
import sys
import subprocess
import argparse
import yaml
from datetime import datetime, timezone
from pathlib import Path

HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
SKILL_DIR = f"{HOME}/.hermes/skills/turbofit"
RESULTS_PATH = f"{SKILL_DIR}/references/benchmark-results.json"
REPO_DIR = f"{HOME}/projects/turbofit"

ATOMIC = f"{HOME}/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
STOCK = f"{HOME}/projects/LLM-Infra/llama.cpp/build/bin/llama-server"
LIB_ATOMIC = f"{HOME}/projects/LLM-Infra/llama.cpp-atomic/build/bin"
LIB_STOCK = f"{HOME}/projects/LLM-Infra/llama.cpp/build/bin"

BENCH_PORT = 11999
PROMPT = "Write a Python function that implements merge sort with type hints and docstrings. Include a brief explanation of the algorithm's time and space complexity."


def load_catalog():
    with open(CATALOG) as f:
        return yaml.safe_load(f) or {}


def kill_port(port):
    os.system(f"fuser -k {port}/tcp 2>/dev/null; sleep 3")


def wait_for_ready(port, timeout=120):
    for _ in range(timeout // 2):
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "2", f"http://127.0.0.1:{port}/health"],
                capture_output=True, text=True, timeout=5
            )
            if "ok" in result.stdout:
                return True
        except:
            pass
        time.sleep(2)
    return False


def run_bench(port, prompt, max_tokens=256):
    start = time.time()
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "120",
             f"http://127.0.0.1:{port}/v1/completions",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": False})],
            capture_output=True, text=True, timeout=180
        )
        elapsed = time.time() - start
        data = json.loads(result.stdout)
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0 and elapsed > 0:
            return {"status": "ok", "tok_s": round(completion_tokens / elapsed, 1),
                    "tokens": completion_tokens, "time_s": round(elapsed, 2)}
        return {"status": "error", "error": "no tokens in response", "time_s": round(elapsed, 2)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def build_launch_args(entry, ctx):
    binary = entry.get("binary", STOCK)
    path = entry["path"]
    mmproj = entry.get("mmproj", "")
    presets = entry.get("presets", [])

    if binary and "atomic" in str(binary):
        actual_binary = ATOMIC
        lib_dir = LIB_ATOMIC
    else:
        actual_binary = STOCK
        lib_dir = LIB_STOCK

    preset_map = {
        "turbo4-kv": ["-ctk", "turbo4", "-ctv", "turbo4"],
        "turbo3-kv": ["-ctk", "turbo3", "-ctv", "turbo3"],
        "q8-kv": ["-ctk", "q8_0", "-ctv", "q8_0"],
        "no-mmap": ["--no-mmap"],
        "split-none": ["--split-mode", "none"],
        "mlock": ["--mlock"],
        "cpu-moe-2": ["--n-cpu-moe", "2"],
        "cpu-moe-4": ["--n-cpu-moe", "4"],
        "cpu-moe-8": ["--n-cpu-moe", "8"],
        "nextn": ["--spec-type", "nextn", "--draft-block-size", "3"],
        "draft-mtp": ["--spec-type", "draft-mtp"],
    }

    flags = []
    for p in presets:
        if p in preset_map:
            flags.extend(preset_map[p])

    extra_args = entry.get("extra_args", [])
    if extra_args:
        flags.extend(extra_args)

    if mmproj and os.path.exists(mmproj):
        flags.extend(["--mmproj", mmproj])

    cmd = [actual_binary, "-m", path, "--host", "127.0.0.1", "--port", str(BENCH_PORT),
           "-ngl", "999", "-fa", "on", "-c", str(ctx), "--jinja", "--main-gpu", "0"] + flags
    return cmd, lib_dir


def benchmark_model(alias, entry, ctx, gpu):
    print(f"\n=== {alias} ===", flush=True)
    if not os.path.exists(entry["path"]):
        print(f"  SKIP: file not found", flush=True)
        return {"name": alias, "status": "skip", "reason": "file not found"}

    size_gb = os.path.getsize(entry["path"]) / (1024**3)
    print(f"  Size: {size_gb:.1f} GB", flush=True)

    kill_port(BENCH_PORT)
    cmd, lib_dir = build_launch_args(entry, ctx)
    print(f"  Launching on GPU {gpu}...", flush=True)

    env = os.environ.copy()
    env["LLAMA_LIB_DIR"] = lib_dir
    env["LD_LIBRARY_PATH"] = lib_dir
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    if not wait_for_ready(BENCH_PORT, timeout=120):
        print(f"  FAILED: startup timeout", flush=True)
        proc.kill()
        kill_port(BENCH_PORT)
        return {"name": alias, "status": "failed", "reason": "startup timeout", "size_gb": round(size_gb, 1)}

    print(f"  Ready, benchmarking...", flush=True)
    run_bench(BENCH_PORT, "Hello", max_tokens=5)  # warmup
    result = run_bench(BENCH_PORT, PROMPT, max_tokens=256)
    result["name"] = alias
    result["alias"] = alias
    result["size_gb"] = round(size_gb, 1)
    result["tier"] = entry.get("tier", "?")
    result["role"] = entry.get("role", "?")
    result["has_mtp"] = any(p in ["nextn", "draft-mtp"] for p in entry.get("presets", []))
    result["has_vision"] = entry.get("vision", False)

    if result["status"] == "ok":
        print(f"  {result['tok_s']} tok/s ({result['tokens']} tokens in {result['time_s']}s)", flush=True)
    else:
        print(f"  ERROR: {result.get('error', '?')}", flush=True)

    proc.kill()
    kill_port(BENCH_PORT)
    return result


def push_to_github():
    os.chdir(REPO_DIR)
    subprocess.run(["git", "add", "-A"], check=False)
    date_str = datetime.now().strftime("%Y-%m-%d")
    subprocess.run(["git", "commit", "-m", f"bench: daily benchmark results {date_str}"], check=False)
    result = subprocess.run(["git", "push"], capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="turbofit benchmark pipeline")
    parser.add_argument("--push", action="store_true", help="Push results to GitHub")
    parser.add_argument("--filter", type=str, default="", help="Only benchmark matching models")
    parser.add_argument("--gpu", type=int, default=1, help="GPU to use")
    parser.add_argument("--ctx", type=int, default=131072, help="Context size")
    args = parser.parse_args()

    cfg = load_catalog()
    models = cfg.get("models", {})

    print(f"turbofit benchmark pipeline")
    print(f"  Catalog: {len(models)} models")
    print(f"  GPU: {args.gpu}, CTX: {args.ctx}")
    if args.filter:
        print(f"  Filter: '{args.filter}'")

    print("\nStopping turbofit daemons for clean benchmark...")
    subprocess.run(["systemctl", "--user", "stop", "turbofit-scaling-watcher.service"],
                   capture_output=True, timeout=10)
    for alias in list(models.keys()):
        subprocess.run(["systemctl", "--user", "stop", f"turbofit-{alias}.service"],
                       capture_output=True, timeout=10)
    time.sleep(3)

    results = []
    for alias, entry in models.items():
        if args.filter and args.filter.lower() not in alias.lower():
            continue
        if entry.get("mode") == "embedding":
            continue
        results.append(benchmark_model(alias, entry, args.ctx, args.gpu))

    output = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ctx": args.ctx,
        "gpu": args.gpu,
        "gpu_name": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0],
        "prompt": PROMPT[:80] + "...",
        "results": results
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {ok_count}/{len(results)} models OK")
    print(f"Results: {RESULTS_PATH}")

    if args.push:
        print("\nPushing to GitHub...")
        if push_to_github():
            print("Pushed!")
        else:
            print("Push failed - check git config")

    print("\nRestarting scaling watcher...")
    subprocess.run(["systemctl", "--user", "start", "turbofit-scaling-watcher.service"],
                   capture_output=True, timeout=10)

    print(f"\n{'Model':<30} {'tok/s':>8} {'Size':>8} {'MTP':>5} {'Vision':>7}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x.get("tok_s", 0), reverse=True):
        if r["status"] == "ok":
            print(f"{r['name']:<30} {r['tok_s']:>8.1f} {r.get('size_gb',0):>7.1f}G "
                  f"{'Y' if r.get('has_mtp') else 'N':>5} {'Y' if r.get('has_vision') else 'N':>7}")
        else:
            print(f"{r['name']:<30} {'FAIL':>8}")


if __name__ == "__main__":
    main()
