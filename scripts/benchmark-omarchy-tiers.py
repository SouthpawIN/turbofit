#!/usr/bin/env python3
"""Run tier benchmarks on Omarchy and collect evidence hashes.

Benchmarks each Fit List candidate on the Omarchy machine (2x RTX 3090).
Produces evidence JSON that feeds the tier tournament promotion pipeline.

Usage:
    python3 scripts/benchmark-omarchy-tiers.py [--tier 8gb] [--quick]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "omarchy"
REMOTE_MODEL_ROOT = "/home/sovthpaw/Models/storage/gguf"
REMOTE_RUNTIME_ROOT = "/home/sovthpaw/.local/share/turbofit/runtimes"

# Model -> (gguf_path, engine, port, extra_flags)
# Ordered by tier priority.
MODELS = {
    "ternary-bonsai-2-27b-ptq1": {
        "gguf": f"{REMOTE_MODEL_ROOT}/Ternary-Bonsai-2-27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        "engine": "prism",
        "port": 8201,
        "context": 262144,
        "tier": "8gb",
    },
    "underdog-27b-pq2": {
        "gguf": f"{REMOTE_MODEL_ROOT}/Underdog-27B/Underdog-PQ2_0.gguf",
        "engine": "prism",
        "port": 8202,
        "context": 262144,
        "tier": "12gb",
    },
    "qwen3-8-27b-gsq-rco-iq3s": {
        "gguf": f"{REMOTE_MODEL_ROOT}/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
        "engine": "llama.cpp",
        "port": 8203,
        "context": 262144,
        "tier": "16gb",
    },
    "qwen3-8-flash-next-strata": {
        "gguf": f"{REMOTE_MODEL_ROOT}/flash-next-gsq/Qwen3.8-Flash-Next-IQ2_XS.gguf",
        "engine": "strata",
        "port": 8082,
        "context": 131072,
        "tier": "24gb+",
    },
    "qwen3-8-27b-gsq-rco-iq3s": {
        "gguf": f"{REMOTE_MODEL_ROOT}/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-UD-Q3_K_XL.gguf",
        "engine": "prism",
        "port": 8100,
        "context": 262144,
        "tier": "48gb",
    },
    "cyber-tiel-coder-35b-a3b-ud-q4-k-xl": {
        "gguf": f"{REMOTE_MODEL_ROOT}/Cyber-Tiel-Coder-35B-A3B/Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "engine": "llama.cpp",
        "port": 8204,
        "context": 262144,
        "tier": "24gb-unified",
    },
    "maple-preview-tq2": {
        "gguf": f"{REMOTE_MODEL_ROOT}/deepgrove_maple-preview-GGUF/maple-preview-TQ2_0.gguf",
        "engine": "maple-llama.cpp",
        "port": 8205,
        "context": 131072,
        "tier": "8gb-unified",
    },
}

PROMPTS = [
    "Explain the difference between TCP and UDP in exactly 3 sentences.",
    "Write a Python function that checks if a string is a palindrome. Include type hints.",
    "What is the capital of France? Answer in one word.",
    "Summarize the causes of the 2008 financial crisis in 5 bullet points.",
    "Write a haiku about distributed systems.",
]


def ssh(cmd: str) -> str:
    """Run a command on the remote host."""
    result = subprocess.run(
        ["tailscale", "ssh", REMOTE, cmd],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout + result.stderr


def ssh_ok(cmd: str) -> bool:
    result = subprocess.run(
        ["tailscale", "ssh", REMOTE, cmd],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0


def stop_servers() -> None:
    """Kill any running llama-server or strata processes on the remote."""
    ssh("pkill -f llama-server 2>/dev/null; pkill -f strata 2>/dev/null; sleep 2")
    # Verify ports are free
    for _ in range(10):
        out = ssh("ss -tlnp 2>/dev/null | grep -E '808[0-9]|809[0-9]|81[0-9][0-9]|82[0-9][0-9]' | wc -l")
        if out.strip() == "0":
            return
        time.sleep(1)


def start_model(model_id: str) -> bool:
    """Start a model server on the remote host. Returns True on success."""
    spec = MODELS[model_id]
    port = spec["port"]
    ctx = spec["context"]

    if spec["engine"] == "strata":
        # Strata uses its own launcher
        cmd = (
            f"cd ~/Strata && nohup .venv/bin/python -m strata serve "
            f"--model {spec['gguf']} --port {port} "
            f"> /tmp/strata-{port}.log 2>&1 &"
        )
    elif spec["engine"] == "prism":
        binary = f"{REMOTE_RUNTIME_ROOT}/prism-llama.cpp-9ca265a57f85f2117942490f421f64a226dd9847/build-cuda/bin/llama-server"
        cmd = (
            f"nohup {binary} -m {spec['gguf']} "
            f"--port {port} -ngl 9999 -fa on -c {ctx} --jinja -t 48 "
            f"--host 127.0.0.1 > /tmp/llama-{port}.log 2>&1 &"
        )
    elif spec["engine"] == "maple-llama.cpp":
        # Maple uses its own runtime
        binary = f"{REMOTE_RUNTIME_ROOT}/maple-llama.cpp/build-cuda/bin/llama-server"
        cmd = (
            f"nohup {binary} -m {spec['gguf']} "
            f"--port {port} -ngl 9999 -fa on -c {ctx} --jinja -t 48 "
            f"--host 127.0.0.1 > /tmp/maple-{port}.log 2>&1 &"
        )
    else:
        binary = f"{REMOTE_RUNTIME_ROOT}/llama.cpp-1c3c9674de4d455f1e571bed808252af54932767/build-cuda/bin/llama-server"
        cmd = (
            f"nohup {binary} -m {spec['gguf']} "
            f"--port {port} -ngl 9999 -fa on -c {ctx} --jinja -t 48 "
            f"--host 127.0.0.1 > /tmp/llama-{port}.log 2>&1 &"
        )

    ssh(cmd)
    # Wait for the server to be ready
    for _ in range(60):
        if ssh_ok(f"curl -sf http://127.0.0.1:{port}/health > /dev/null 2>&1"):
            return True
        time.sleep(2)
    return False


def bench_model(model_id: str, port: int, quick: bool = False) -> dict:
    """Run benchmark prompts against a model and collect timing data."""
    prompts = PROMPTS[:2] if quick else PROMPTS
    results = []
    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        payload = json.dumps({
            "model": Path(MODELS[model_id]["gguf"]).name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7,
        }).encode()

        req = urllib.request.Request(
            f"http://{REMOTE}.tail6ff78b.ts.net:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        # Use SSH tunnel instead — Tailscale HTTP may not be exposed
        # Fall back to direct curl via SSH
        cmd = (
            f"curl -s http://127.0.0.1:{port}/v1/chat/completions "
            f"-H 'Content-Type: application/json' "
            f"-d '{json.dumps({'model': Path(MODELS[model_id]['gguf']).name, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 200, 'temperature': 0.7})}'"
        )
        out = ssh(cmd)
        try:
            data = json.loads(out)
            timings = data.get("timings", {})
            tps = timings.get("predicted_per_second", 0)
            tokens = timings.get("predicted_n", 0)
            total_tokens += tokens
            total_time += timings.get("predicted_ms", 0) / 1000
            results.append({
                "prompt": prompt[:50],
                "tokens": tokens,
                "tok_per_sec": round(tps, 2),
            })
        except (json.JSONDecodeError, KeyError):
            results.append({"prompt": prompt[:50], "error": out[:200]})

    avg_tps = round(total_tokens / total_time, 2) if total_time > 0 else 0
    return {
        "model": model_id,
        "tier": MODELS[model_id]["tier"],
        "prompts": results,
        "avg_tok_per_sec": avg_tps,
        "total_tokens": total_tokens,
    }


def compute_evidence_hash(bench: dict) -> str:
    """Compute a sha256 evidence hash from benchmark results."""
    import hashlib
    canonical = json.dumps(bench, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Turbofit tier candidates on Omarchy")
    parser.add_argument("--tier", help="Only benchmark this tier (e.g. 8gb, 12gb, 16gb)")
    parser.add_argument("--quick", action="store_true", help="Only run 2 prompts per model")
    parser.add_argument("--skip-running", action="store_true", help="Skip stopping running servers")
    args = parser.parse_args()

    models_to_bench = list(MODELS.keys())
    if args.tier:
        models_to_bench = [m for m, s in MODELS.items() if s["tier"].startswith(args.tier)]
        if not models_to_bench:
            print(f"No models for tier {args.tier}")
            return 1

    print(f"Benchmarking {len(models_to_bench)} models on {REMOTE}...")
    if not args.skip_running:
        print("Stopping running servers...")
        stop_servers()

    all_results = []
    for model_id in models_to_bench:
        spec = MODELS[model_id]
        print(f"\n=== {model_id} ({spec['tier']}) ===")
        print(f"  Starting on port {spec['port']}...")

        if not start_model(model_id):
            print(f"  FAILED to start {model_id}")
            all_results.append({"model": model_id, "error": "failed to start"})
            continue

        print(f"  Running benchmarks...")
        bench = bench_model(model_id, spec["port"], quick=args.quick)
        bench["evidence"] = compute_evidence_hash(bench)
        all_results.append(bench)
        print(f"  avg_tok_per_sec: {bench.get('avg_tok_per_sec', 'N/A')}")
        print(f"  evidence: {bench['evidence'][:20]}...")

        # Stop this model before starting the next
        ssh(f"pkill -f 'port {spec['port']}' 2>/dev/null; pkill -f llama-server 2>/dev/null; sleep 2")

    # Write results
    out_path = ROOT / "references/results/omarchy-tier-benchmarks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "host": REMOTE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results,
    }, indent=2) + "\n")
    print(f"\nResults written to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for r in all_results:
        if "error" in r:
            print(f"  {r['model']}: ERROR - {r['error']}")
        else:
            print(f"  {r['model']} ({r['tier']}): {r['avg_tok_per_sec']} tok/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
