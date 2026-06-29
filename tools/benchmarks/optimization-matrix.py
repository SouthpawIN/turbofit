#!/usr/bin/env python3
"""
turbofit optimization matrix — tests all flag combinations across all backends
to find the optimal configuration for each model at each context tier.

Tests:
  - llama.cpp atomic (turbo4, turbo3, turbo2 KV cache)
  - llama.cpp stock (q8_0, q4_0 KV cache)
  - Speculative decoding: none, nextn, draft-mtp
  - Context tiers: 262K, 131K, 65536
  - GPU: 0, 1

Outputs:
  - references/optimization-matrix.yaml (best config per model per tier)
  - references/optimization-results.json (full raw results)
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
import itertools

HOME = os.path.expanduser("~")
CATALOG = f"{HOME}/.config/turbofit/models.yaml"
SKILL_DIR = f"{HOME}/.hermes/skills/turbofit"
RESULTS_PATH = f"{SKILL_DIR}/references/optimization-results.json"
MATRIX_PATH = f"{SKILL_DIR}/references/optimization-matrix.yaml"

ATOMIC = f"{HOME}/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
STOCK = f"{HOME}/projects/LLM-Infra/llama.cpp/build/bin/llama-server"
LIB_ATOMIC = f"{HOME}/projects/LLM-Infra/llama.cpp-atomic/build/bin"
LIB_STOCK = f"{HOME}/projects/LLM-Infra/llama.cpp/build/bin"

BENCH_PORT = 11999
PROMPT = "Write a Python function that implements merge sort with type hints and docstrings."

# Flag combinations to test
KV_CACHE_TYPES = {
    "atomic": ["turbo4", "turbo3", "turbo2", "q8_0"],
    "stock": ["q8_0", "q4_0"],
}

SPEC_TYPES = {
    "atomic": ["none", "nextn", "draft-mtp"],
    "stock": ["none", "draft-mtp"],
}

CTX_TIERS = [262144, 131072, 65536]


def get_vram_free(gpu=0):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip()) / 1024
    except:
        return 0


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


def run_bench(port, max_tokens=256):
    start = time.time()
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "120",
             f"http://127.0.0.1:{port}/v1/completions",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0, "stream": False})],
            capture_output=True, text=True, timeout=180
        )
        elapsed = time.time() - start
        data = json.loads(result.stdout)
        usage = data.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        if tokens > 0 and elapsed > 0:
            return {"status": "ok", "tok_s": round(tokens / elapsed, 1),
                    "tokens": tokens, "time_s": round(elapsed, 2)}
        return {"status": "error", "error": "no tokens", "time_s": round(elapsed, 2)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def estimate_vram_need(model_size_gb, ctx, kv_type):
    """Rough VRAM estimate: model + KV cache + overhead."""
    model_vram = model_size_gb
    # KV cache sizes (rough, per 262K ctx):
    # turbo4: ~3.3GB, turbo3: ~4.5GB, turbo2: ~5.5GB, q8_0: ~8.7GB, q4_0: ~4.4GB
    kv_multipliers = {
        "turbo4": 3.3, "turbo3": 4.5, "turbo2": 5.5,
        "q8_0": 8.7, "q4_0": 4.4
    }
    kv_base = kv_multipliers.get(kv_type, 8.7)
    kv_scaled = kv_base * (ctx / 262144)
    overhead = 1.0  # GB overhead
    return model_vram + kv_scaled + overhead


def build_cmd(binary, model_path, ctx, kv_type, spec_type, mmproj="", gpu=0):
    if "atomic" in binary:
        actual_binary = ATOMIC
        lib_dir = LIB_ATOMIC
    else:
        actual_binary = STOCK
        lib_dir = LIB_STOCK

    cmd = [actual_binary, "-m", model_path, "--host", "127.0.0.1", "--port", str(BENCH_PORT),
           "-ngl", "999", "-fa", "on", "-c", str(ctx), "--jinja", "--main-gpu", "0"]

    # KV cache
    cmd.extend(["-ctk", kv_type, "-ctv", kv_type])

    # No mmap
    cmd.append("--no-mmap")

    # Spec decoding
    if spec_type == "nextn":
        cmd.extend(["--spec-type", "nextn", "--draft-block-size", "3"])
    elif spec_type == "draft-mtp":
        cmd.extend(["--spec-type", "draft-mtp"])

    # mmproj
    if mmproj and os.path.exists(mmproj):
        cmd.extend(["--mmproj", mmproj])

    return cmd, lib_dir


def test_config(model_alias, model_path, model_size_gb, mmproj, binary_type,
                kv_type, spec_type, ctx, gpu):
    """Test a single configuration. Returns result dict."""
    config_name = f"{model_alias}/{binary_type}/{kv_type}/{spec_type}/{ctx//1024}K"

    # Check VRAM
    free_vram = get_vram_free(gpu)
    needed_vram = estimate_vram_need(model_size_gb, ctx, kv_type)

    if needed_vram > free_vram:
        return {
            "config": config_name,
            "status": "skip_oom",
            "needed_gb": round(needed_vram, 1),
            "free_gb": round(free_vram, 1),
            "tok_s": 0
        }

    binary = ATOMIC if binary_type == "atomic" else STOCK
    cmd, lib_dir = build_cmd(binary, model_path, ctx, kv_type, spec_type, mmproj, gpu)

    kill_port(BENCH_PORT)

    env = os.environ.copy()
    env["LLAMA_LIB_DIR"] = lib_dir
    env["LD_LIBRARY_PATH"] = lib_dir
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    if not wait_for_ready(BENCH_PORT, timeout=120):
        proc.kill()
        kill_port(BENCH_PORT)
        return {"config": config_name, "status": "timeout", "tok_s": 0}

    # Warmup
    run_bench(BENCH_PORT, max_tokens=5)

    # Real benchmark
    result = run_bench(BENCH_PORT, max_tokens=256)
    result["config"] = config_name
    result["model"] = model_alias
    result["binary"] = binary_type
    result["kv_type"] = kv_type
    result["spec_type"] = spec_type
    result["ctx"] = ctx
    result["gpu"] = gpu
    result["model_size_gb"] = model_size_gb

    proc.kill()
    kill_port(BENCH_PORT)
    return result


def main():
    parser = argparse.ArgumentParser(description="turbofit optimization matrix")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to test on")
    parser.add_argument("--filter", type=str, default="", help="Filter models")
    parser.add_argument("--quick", action="store_true", help="Quick test: only stock+q8 at 131K")
    args = parser.parse_args()

    with open(CATALOG) as f:
        cfg = yaml.safe_load(f) or {}
    models = cfg.get("models", {})

    print(f"turbofit optimization matrix")
    print(f"  GPU: {args.gpu}")
    print(f"  Models: {len(models)}")

    # Stop daemons
    subprocess.run(["systemctl", "--user", "stop", "turbofit-scaling-watcher.service"],
                   capture_output=True, timeout=10)
    for alias in models:
        subprocess.run(["systemctl", "--user", "stop", f"turbofit-{alias}.service"],
                       capture_output=True, timeout=10)
    time.sleep(3)

    all_results = []

    for alias, entry in models.items():
        if args.filter and args.filter.lower() not in alias.lower():
            continue
        if entry.get("mode") == "embedding":
            continue

        model_path = entry["path"]
        if not os.path.exists(model_path):
            print(f"\nSKIP {alias}: file not found")
            continue

        model_size_gb = os.path.getsize(model_path) / (1024**3)
        mmproj = entry.get("mmproj", "")
        has_mtp = any(p in ["nextn", "draft-mtp"] for p in entry.get("presets", []))
        catalog_binary = entry.get("binary", STOCK)
        is_atomic = "atomic" in str(catalog_binary)

        print(f"\n{'='*60}")
        print(f"Model: {alias} ({model_size_gb:.1f}GB)")
        print(f"  Atomic: {is_atomic}, MTP: {has_mtp}, Vision: {entry.get('vision', False)}")
        print(f"{'='*60}")

        # Determine which backends to test
        backends = []
        if is_atomic:
            backends.append("atomic")
        backends.append("stock")

        # Determine spec types
        spec_types_for_model = ["none"]
        if has_mtp:
            if "atomic" in backends:
                spec_types_for_model.extend(["nextn", "draft-mtp"])
            spec_types_for_model.append("draft-mtp")  # stock supports draft-mtp

        # Context tiers
        ctx_tiers = [131072, 65536] if args.quick else CTX_TIERS

        for backend in backends:
            kv_types = KV_CACHE_TYPES.get(backend, ["q8_0"])
            for kv_type in kv_types:
                for spec_type in spec_types_for_model:
                    if backend == "stock" and spec_type == "nextn":
                        continue  # stock doesn't support nextn
                    if backend == "stock" and kv_type.startswith("turbo"):
                        continue  # stock doesn't support turbo

                    for ctx in ctx_tiers:
                        config_name = f"{alias}/{backend}/{kv_type}/{spec_type}/{ctx//1024}K"
                        print(f"\n  Testing: {config_name}", flush=True, end=" ")

                        result = test_config(alias, model_path, model_size_gb, mmproj,
                                           backend, kv_type, spec_type, ctx, args.gpu)
                        all_results.append(result)

                        if result["status"] == "ok":
                            print(f"-> {result['tok_s']} tok/s", flush=True)
                        elif result["status"] == "skip_oom":
                            print(f"-> SKIP (need {result['needed_gb']}GB, have {result['free_gb']}GB)", flush=True)
                        else:
                            print(f"-> {result['status']}", flush=True)

                        # Skip smaller ctx if we already OOMed at larger ctx with same flags
                        if result["status"] == "skip_oom" and ctx == max(ctx_tiers):
                            break
                    if result["status"] == "skip_oom" and ctx == max(ctx_tiers):
                        break

    # Write raw results
    output = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": args.gpu,
        "gpu_name": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0],
        "prompt": PROMPT[:80] + "...",
        "results": all_results
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Build optimization matrix (best config per model per ctx tier)
    matrix = {}
    for r in all_results:
        if r["status"] != "ok":
            continue
        model = r["model"]
        ctx = r["ctx"]
        key = f"{model}_{ctx//1024}K"
        if key not in matrix or r["tok_s"] > matrix[key]["tok_s"]:
            matrix[key] = {
                "model": model,
                "ctx": ctx,
                "tok_s": r["tok_s"],
                "binary": r["binary"],
                "kv_type": r["kv_type"],
                "spec_type": r["spec_type"],
                "size_gb": r.get("model_size_gb", 0),
            }

    matrix_yaml = {
        "date": output["date"],
        "gpu": output["gpu_name"],
        "optimal_configs": matrix,
    }

    with open(MATRIX_PATH, "w") as f:
        yaml.safe_dump(matrix_yaml, f, default_flow_style=False, sort_keys=True)

    # Print summary
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION MATRIX COMPLETE")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'CTX':>8} {'tok/s':>8} {'Backend':>10} {'KV':>8} {'Spec':>10}")
    print("-" * 75)
    for key in sorted(matrix.keys()):
        m = matrix[key]
        print(f"{m['model']:<25} {m['ctx']//1024:>7}K {m['tok_s']:>8.1f} "
              f"{m['binary']:>10} {m['kv_type']:>8} {m['spec_type']:>10}")

    print(f"\nResults: {RESULTS_PATH}")
    print(f"Matrix:  {MATRIX_PATH}")

    # Restart watcher
    subprocess.run(["systemctl", "--user", "start", "turbofit-scaling-watcher.service"],
                   capture_output=True, timeout=10)


if __name__ == "__main__":
    main()
