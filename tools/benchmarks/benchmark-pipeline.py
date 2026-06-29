#!/usr/bin/env python3.12
"""
turbofit benchmark pipeline v2 — auto-discovers all GGUF models, benchmarks with
lm-eval-harness (MMLU, GSM8K, GPQA), measures tok/s, ranks by:
  1. Smartest (MMLU + GSM8K + GPQA composite)
  2. Fastest (tok/s)
  3. Smallest weight (GB on disk)
Favoring smartest, then fastest, then smallest.

Results published to GitHub as benchmark-results.json. All turbofit installations
pull from this file via `serve bench pull`.

Usage:
  python3.12 benchmark-pipeline.py                    # benchmark all, write JSON
  python3.12 benchmark-pipeline.py --push              # benchmark + push to GitHub
  python3.12 benchmark-pipeline.py --filter "27b"      # only benchmark matching models
  python3.12 benchmark-pipeline.py --gpu 0             # use specific GPU
  python3.12 benchmark-pipeline.py --ctx 65536         # smaller ctx for faster bench
  python3.12 benchmark-pipeline.py --tasks mmlu,gsm8k  # override lm-eval tasks
  python3.12 benchmark-pipeline.py --limit 50          # limit questions per task
  python3.12 benchmark-pipeline.py --speed-only        # skip lm-eval, just tok/s
"""

import json
import time
import os
import sys
import subprocess
import argparse
import yaml
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
SKILL_DIR = f"{HOME}/.hermes/skills/turbofit"
RESULTS_PATH = f"{SKILL_DIR}/references/benchmark-results.json"
REPO_DIR = f"{HOME}/projects/turbofit"
GITHUB_RAW = "https://raw.githubusercontent.com/SouthpawIN/turbofit/main/skills/turbofit/references/benchmark-results.json"

ATOMIC = f"{HOME}/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
STOCK = f"{HOME}/projects/LLM-Infra/llama.cpp/build/bin/llama-server"
LIB_ATOMIC = f"{HOME}/projects/LLM-Infra/llama.cpp-atomic/build/bin"
LIB_STOCK = f"{HOME}/projects/LLM-Infra/llama.cpp/build/bin"

BENCH_PORT = 11999

# Default lm-eval tasks: reasoning + math + coding
DEFAULT_TASKS = "mmlu,gsm8k,gpqa,humaneval"
DEFAULT_LIMIT = 100  # questions per task (full MMLU is 14k, 100 is representative)

SPEED_PROMPT = "Write a Python function that implements merge sort with type hints and docstrings. Include a brief explanation of the algorithm's time and space complexity."


def load_catalog():
    with open(CATALOG) as f:
        return yaml.safe_load(f) or {}


def kill_port(port):
    """Kill any process listening on the given port."""
    os.system(f"fuser -k {port}/tcp 2>/dev/null")
    time.sleep(1)


def wait_for_ready(port, timeout=180):
    """Wait for llama-server to respond on /health."""
    for i in range(timeout):
        try:
            urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def run_speed_bench(port, prompt, max_tokens=256):
    """Run a single completion and measure tok/s."""
    data = json.dumps({
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                 data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urlopen(req, timeout=120)
        d = json.loads(resp.read())
        timings = d.get("timings", {})
        tok_s = timings.get("predicted_per_second", 0)
        if not tok_s:
            # Fallback: compute from usage
            usage = d.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            predicted_ms = timings.get("predicted_ms", 0)
            if predicted_ms > 0:
                tok_s = comp_tokens / (predicted_ms / 1000)
        return {
            "status": "ok",
            "tok_s": round(tok_s, 1),
            "tokens": d.get("usage", {}).get("completion_tokens", 0),
            "time_s": round(timings.get("predicted_ms", 0) / 1000, 1),
            "prompt_per_second": round(timings.get("prompt_per_second", 0), 1),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "tok_s": 0}


def run_lm_eval(port, tasks, limit, model_name="bench"):
    """Run lm-eval-harness against the server."""
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", f"model={model_name},base_url=http://127.0.0.1:{port}/v1/completions,num_concurrent=1",
        "--tasks", tasks,
        "--limit", str(limit),
        "--output_path", "/tmp/lm-eval-results",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        # Parse results from stdout
        scores = {}
        for line in result.stdout.split("\n"):
            # lm-eval prints lines like: "|mmlu       |acc     |0.6532|"
            for task in tasks.split(","):
                task = task.strip()
                if f"|{task}" in line and "acc" in line:
                    parts = line.split("|")
                    if len(parts) >= 4:
                        try:
                            val = float(parts[-2].strip())
                            scores[task] = val
                        except ValueError:
                            pass

        # Also try reading from the results file
        if not scores:
            results_file = Path("/tmp/lm-eval-results")
            if results_file.exists():
                for f in results_file.rglob("*.json"):
                    try:
                        data = json.loads(f.read_text())
                        for task_name, task_results in data.get("results", {}).items():
                            for metric, value in task_results.items():
                                if metric in ("acc", "acc_norm", "exact_match"):
                                    scores[task_name] = value
                                    break
                    except Exception:
                        pass

        return scores
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


def build_launch_args(entry, ctx, gpu):
    """Build llama-server launch command from catalog entry."""
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
        "draft-mtp": ["--spec-type", "mtp"],
        "mtp": ["--spec-type", "mtp"],
        "nextn": ["--spec-type", "nextn", "--draft-block-size", "3"],
        "nextn-tight": ["--spec-type", "nextn", "--draft-block-size", "2"],
    }

    flags = []
    for p in presets:
        if p in preset_map:
            flags.extend(preset_map[p])

    # nextn needs --model-draft
    if any(p in ("nextn", "nextn-tight") for p in presets):
        flags.extend(["--model-draft", path])

    extra_args = entry.get("extra_args", [])
    if extra_args:
        flags.extend(extra_args)

    if mmproj and os.path.exists(mmproj):
        flags.extend(["--mmproj", mmproj])

    cmd = [
        actual_binary, "-m", path,
        "--host", "127.0.0.1", "--port", str(BENCH_PORT),
        "-ngl", "99", "-fa", "on", "-c", str(ctx),
        "--jinja", "--main-gpu", "0",
    ] + flags
    return cmd, lib_dir


def benchmark_model(alias, entry, ctx, gpu, tasks, limit, speed_only=False):
    """Benchmark a single model: speed + lm-eval."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {alias}", flush=True)
    print(f"{'='*60}", flush=True)

    if not os.path.exists(entry["path"]):
        print(f"  SKIP: file not found", flush=True)
        return {"name": alias, "alias": alias, "status": "skip", "reason": "file not found"}

    size_gb = os.path.getsize(entry["path"]) / (1024**3)
    print(f"  Size: {size_gb:.1f} GB", flush=True)

    kill_port(BENCH_PORT)
    cmd, lib_dir = build_launch_args(entry, ctx, gpu)

    # For nextn models: need both GPUs
    has_nextn = any(p in ("nextn", "nextn-tight") for p in entry.get("presets", []))
    if has_nextn:
        gpu_setting = "0,1"
        # Kill Carnice to free GPU 1
        os.system("systemctl --user stop turbofit-carnice.service 2>/dev/null")
        time.sleep(3)
    else:
        gpu_setting = str(gpu)

    print(f"  Launching on GPU {gpu_setting}...", flush=True)

    env = os.environ.copy()
    env["LLAMA_LIB_DIR"] = lib_dir
    env["LD_LIBRARY_PATH"] = lib_dir
    env["CUDA_VISIBLE_DEVICES"] = gpu_setting

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

    if not wait_for_ready(BENCH_PORT, timeout=180):
        print(f"  FAILED: startup timeout", flush=True)
        proc.kill()
        kill_port(BENCH_PORT)
        return {"name": alias, "alias": alias, "status": "failed",
                "reason": "startup timeout", "size_gb": round(size_gb, 1)}

    print(f"  Ready!", flush=True)

    # Warmup
    run_speed_bench(BENCH_PORT, "Hello", max_tokens=5)

    # Speed benchmark
    print(f"  Speed benchmarking...", flush=True)
    speed = run_speed_bench(BENCH_PORT, SPEED_PROMPT, max_tokens=256)

    result = {
        "name": alias,
        "alias": alias,
        "status": speed["status"],
        "size_gb": round(size_gb, 1),
        "tier": entry.get("tier", "?"),
        "role": entry.get("role", "?"),
        "has_mtp": any(p in ["nextn", "draft-mtp", "mtp"] for p in entry.get("presets", [])),
        "has_vision": entry.get("vision", False),
        "tok_s": speed.get("tok_s", 0),
        "speed_tokens": speed.get("tokens", 0),
        "speed_time_s": speed.get("time_s", 0),
        "prompt_per_second": speed.get("prompt_per_second", 0),
    }

    if speed["status"] == "ok":
        print(f"  Speed: {result['tok_s']} tok/s", flush=True)
    else:
        print(f"  Speed ERROR: {speed.get('error', '?')}", flush=True)

    # lm-eval benchmarks
    if not speed_only and speed["status"] == "ok":
        print(f"  Running lm-eval ({tasks}, limit={limit})...", flush=True)
        scores = run_lm_eval(BENCH_PORT, tasks, limit, model_name=alias)

        if "error" in scores:
            print(f"  lm-eval ERROR: {scores['error']}", flush=True)
            result["lm_eval_error"] = scores["error"]
        else:
            result["scores"] = scores
            for task, score in scores.items():
                print(f"  {task}: {score:.4f}", flush=True)

            # Compute composite intelligence score (0-100)
            composite = 0
            weights = {"mmlu": 30, "gsm8k": 25, "gpqa": 25, "humaneval": 20}
            total_weight = 0
            for task, weight in weights.items():
                if task in scores and isinstance(scores[task], (int, float)):
                    composite += scores[task] * weight
                    total_weight += weight
            if total_weight > 0:
                composite = round((composite / total_weight) * 100, 1)
            result["intelligence_score"] = composite
            print(f"  Intelligence: {composite}/100", flush=True)

    proc.kill()
    kill_port(BENCH_PORT)

    # Restart Carnice if we killed it for nextn
    if has_nextn:
        os.system("systemctl --user start turbofit-carnice.service 2>/dev/null")

    time.sleep(2)
    return result


def compute_ranking(results):
    """Rank models: smartest → fastest → smallest.

    Returns tier assignments: S (top 2-3), SF, SD, F, C.
    """
    # Only rank models that have intelligence scores
    scored = [r for r in results if r.get("intelligence_score") is not None and r["status"] == "ok"]
    speed_only = [r for r in results if r["status"] == "ok" and r.get("intelligence_score") is None]

    # Sort: intelligence desc, then tok/s desc, then size asc
    scored.sort(key=lambda r: (-r["intelligence_score"], -r["tok_s"], r["size_gb"]))

    # Assign tiers based on ranking
    # Top 2-3 = S tier, next 2-3 = SF, next = SD, rest = F, tiny/weak = C
    tiers = {}
    for i, r in enumerate(scored):
        if i < 3:
            tiers[r["alias"]] = "s"
        elif i < 6:
            tiers[r["alias"]] = "sf"
        elif i < 10:
            tiers[r["alias"]] = "sd"
        else:
            tiers[r["alias"]] = "f"

    # Speed-only models get F or C based on size
    speed_only.sort(key=lambda r: r["size_gb"])
    for r in speed_only:
        if r["size_gb"] <= 6:
            tiers[r["alias"]] = "c"
        else:
            tiers[r["alias"]] = "f"

    return tiers, scored


def update_catalog_tiers(tiers):
    """Update models.yaml with benchmark-derived tiers."""
    cfg = load_catalog()
    changed = 0
    for alias, entry in cfg.get("models", {}).items():
        if alias in tiers:
            old_tier = entry.get("tier", "?")
            new_tier = tiers[alias]
            if old_tier != new_tier:
                entry["tier"] = new_tier
                changed += 1
    if changed:
        with open(CATALOG, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n  Updated {changed} tier assignments in catalog", flush=True)
    return changed


def push_to_github():
    """Push benchmark results to GitHub."""
    if not os.path.isdir(REPO_DIR):
        print(f"  Repo not found: {REPO_DIR}", flush=True)
        return False
    os.chdir(REPO_DIR)
    subprocess.run(["git", "add", "-A"], check=False)
    date_str = datetime.now().strftime("%Y-%m-%d")
    subprocess.run(["git", "commit", "-m", f"bench: daily benchmark results {date_str}"], check=False)
    result = subprocess.run(["git", "push"], capture_output=True, text=True)
    return result.returncode == 0


def pull_from_github():
    """Pull latest benchmark results from GitHub."""
    try:
        resp = urlopen(GITHUB_RAW, timeout=15)
        data = json.loads(resp.read())
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Pulled latest results from GitHub ({data.get('date', '?')})", flush=True)
        return data
    except Exception as e:
        print(f"  Failed to pull from GitHub: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="turbofit benchmark pipeline v2")
    parser.add_argument("--push", action="store_true", help="Push results to GitHub")
    parser.add_argument("--filter", type=str, default="", help="Only benchmark matching models")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use (default: 0)")
    parser.add_argument("--ctx", type=int, default=65536, help="Context size (default: 65536 for speed)")
    parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS, help="lm-eval tasks")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Questions per task")
    parser.add_argument("--speed-only", action="store_true", help="Skip lm-eval, just measure tok/s")
    parser.add_argument("--pull", action="store_true", help="Just pull latest results from GitHub, don't benchmark")
    parser.add_argument("--update-tiers", action="store_true", help="Update catalog tiers from results")
    args = parser.parse_args()

    if args.pull:
        print("Pulling latest benchmark results from GitHub...")
        data = pull_from_github()
        if data:
            print(f"\nLatest results: {data.get('date', '?')}")
            print(f"{'Model':<30} {'IQ':>6} {'tok/s':>7} {'Size':>7} {'Tier':>5}")
            print("-" * 60)
            for r in sorted(data.get("results", []),
                          key=lambda r: -(r.get("intelligence_score") or 0)):
                if r["status"] == "ok":
                    iq = f"{r.get('intelligence_score', 0):.0f}" if r.get("intelligence_score") else "—"
                    print(f"  {r['name']:<28} {iq:>6} {r['tok_s']:>7.1f} {r.get('size_gb',0):>6.1f}G {r.get('tier','?'):>5}")
        return

    cfg = load_catalog()
    models = cfg.get("models", {})

    print(f"\n turbofit benchmark pipeline v2", flush=True)
    print(f"  Catalog: {len(models)} models", flush=True)
    print(f"  GPU: {args.gpu}, CTX: {args.ctx}", flush=True)
    print(f"  Tasks: {args.tasks} (limit={args.limit})", flush=True)
    print(f"  Speed only: {args.speed_only}", flush=True)
    if args.filter:
        print(f"  Filter: '{args.filter}'", flush=True)

    # Stop turbofit daemons for clean benchmark
    print("\n  Stopping turbofit daemons for clean benchmark...", flush=True)
    subprocess.run(["systemctl", "--user", "stop", "turbofit-scaling-watcher.service"],
                   capture_output=True, timeout=10)
    for alias in list(models.keys()):
        subprocess.run(["systemctl", "--user", "stop", f"turbofit-{alias}.service"],
                       capture_output=True, timeout=10)
    # Also stop raw darwin.service if running
    subprocess.run(["systemctl", "--user", "stop", "darwin.service"],
                   capture_output=True, timeout=10)
    time.sleep(3)

    results = []
    for alias, entry in models.items():
        if args.filter and args.filter.lower() not in alias.lower():
            continue
        if entry.get("mode") == "embedding":
            continue
        results.append(benchmark_model(alias, entry, args.ctx, args.gpu,
                                       args.tasks, args.limit, args.speed_only))

    # Compute rankings and tiers
    tiers, scored = compute_ranking(results)

    # Build output
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    ).stdout.strip().split("\n")[0]

    output = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": 2,
        "ctx": args.ctx,
        "gpu": args.gpu,
        "gpu_name": gpu_name,
        "tasks": args.tasks,
        "limit": args.limit,
        "ranking_criteria": "intelligence_score desc, tok_s desc, size_gb asc (favoring smartest, then fastest, then smallest)",
        "tier_assignments": tiers,
        "results": results,
    }

    # Save results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Update catalog tiers
    if not args.speed_only:
        update_catalog_tiers(tiers)

    # Print summary table
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*70}", flush=True)
    print(f"  Benchmark complete: {ok_count}/{len(results)} models OK", flush=True)
    print(f"  Results: {RESULTS_PATH}", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Model':<30} {'IQ':>6} {'tok/s':>7} {'Size':>7} {'Tier':>5} {'MTP':>4} {'Vis':>4}", flush=True)
    print("-" * 70, flush=True)
    for r in sorted(results, key=lambda r: -(r.get("intelligence_score") or 0)):
        if r["status"] == "ok":
            iq = f"{r.get('intelligence_score', 0):.0f}" if r.get("intelligence_score") else "—"
            tier = tiers.get(r["alias"], r.get("tier", "?"))
            print(f"  {r['name']:<28} {iq:>6} {r['tok_s']:>7.1f} {r.get('size_gb',0):>6.1f}G "
                  f"{tier:>5} {'Y' if r.get('has_mtp') else 'N':>4} {'Y' if r.get('has_vision') else 'N':>4}", flush=True)
        else:
            print(f"  {r['name']:<28} {'FAIL':>6}", flush=True)

    # Print API recommendations
    print(f"\n{'='*70}", flush=True)
    print("  RECOMMENDATIONS (smartest→fastest→smallest):", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  Main candidates (S+SF tier):", flush=True)
    for r in scored[:6]:
        tier = tiers.get(r["alias"], "?")
        if tier in ("s", "sf"):
            print(f"    [{tier.upper()}] {r['name']:<28} IQ={r.get('intelligence_score',0):.0f} "
                  f"speed={r['tok_s']:.1f} size={r['size_gb']:.1f}G", flush=True)

    aux_candidates = [r for r in results if r["status"] == "ok" and r.get("has_vision")]
    aux_candidates.sort(key=lambda r: (-r.get("intelligence_score", 0), -r["tok_s"], r["size_gb"]))
    print(f"\n  Aux candidates (vision-capable, ranked):", flush=True)
    for r in aux_candidates[:5]:
        tier = tiers.get(r["alias"], "?")
        print(f"    [{tier.upper()}] {r['name']:<28} IQ={r.get('intelligence_score',0):.0f} "
              f"speed={r['tok_s']:.1f} size={r['size_gb']:.1f}G vis=Y", flush=True)

    # Push to GitHub
    if args.push:
        print("\n  Pushing to GitHub...", flush=True)
        if push_to_github():
            print("  Pushed!", flush=True)
        else:
            print("  Push failed - check git config", flush=True)

    # Restart services
    print("\n  Restarting turbofit services...", flush=True)
    subprocess.run(["systemctl", "--user", "start", "turbofit-scaling-watcher.service"],
                   capture_output=True, timeout=10)
    subprocess.run(["systemctl", "--user", "start", "turbofit-darwin-28b-reason.service"],
                   capture_output=True, timeout=10)
    subprocess.run(["systemctl", "--user", "start", "turbofit-carnice.service"],
                   capture_output=True, timeout=10)


if __name__ == "__main__":
    main()
