#!/usr/bin/env python3
"""Benchmark every model in the live Turbofit catalog with one fixed workload.

Measures server-reported prompt/decode throughput when available and wall-clock
output throughput for every runtime. Production services are stopped during the
sweep and restored in a finally block. Incremental JSON is written after each
model so a partial long-running sweep remains inspectable.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HOME = Path.home()
CATALOG = HOME / ".config/turbofit/models.yaml"
PORT = 11999
PROMPT = (
    "Write a Python merge_sort function with type hints and a docstring, then "
    "briefly explain its time and space complexity."
)
PRODUCTION_SERVICES = (
    "turbofit-controller.service",
    "turbofit-gateway.service",
    "turbohaul-manager.service",
    "turbofit-ace-step.service",
)
CONFLICTING_SERVICES = (
    "turbofit-carnice.service",
    "turbofit-darwin-28b-reason.service",
    "glm52-colibri.service",
    "glm52-colibri-65k-fullpin.service",
    "glm52-colibri-gpumax.service",
    "glm52-colibri-simple.service",
    "glm52-llamacpp.service",
)


def run(*args: str, check: bool = False, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check, timeout=timeout)


def service_active(name: str) -> bool:
    return run("systemctl", "--user", "is-active", "--quiet", name).returncode == 0


def stop_services(names: tuple[str, ...]) -> None:
    run("systemctl", "--user", "stop", *names, timeout=120)


def restore_services(names: list[str]) -> None:
    if names:
        run("systemctl", "--user", "start", *names, timeout=120)


def cleanup_port() -> None:
    try:
        run("docker", "rm", "-f", "turbofit-catalog-speed-bench", timeout=180)
    except subprocess.TimeoutExpired:
        # Large mmap-backed containers can take minutes to release host pages.
        # Continue cleanup and never let Docker teardown suppress service restore.
        pass
    try:
        run("fuser", "-k", f"{PORT}/tcp", timeout=30)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(2)


def restore_after_sweep(active_before: list[str]) -> None:
    try:
        cleanup_port()
    finally:
        restore_services(active_before)


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 10) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(process: subprocess.Popen[bytes], timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        for path in ("health", "v1/models"):
            try:
                request_json(f"http://127.0.0.1:{PORT}/{path}", timeout=3)
                return True
            except Exception:
                pass
        time.sleep(2)
    return False


def fixed_llama_args(alias: str, entry: dict[str, Any], ctx: int) -> tuple[list[str], dict[str, str]]:
    binary_overrides = {
        "laguna-s2-1-fp16": HOME / "projects/LLM-Infra/llama.cpp-laguna/build/bin/llama-server",
        "laguna-s2-1-q4": HOME / "projects/LLM-Infra/llama.cpp-laguna/build/bin/llama-server",
        "minimax-m3-q4": HOME / "projects/LLM-Infra/llama.cpp-minimax-m3/build/bin/llama-server",
    }
    binary = str(binary_overrides.get(alias, Path(entry["binary"])))
    env = os.environ.copy()
    build_dir = Path(binary).parent.parent
    env["LD_LIBRARY_PATH"] = ":".join((
        str(build_dir / "src"),
        str(build_dir / "ggml/src"),
        str(build_dir / "examples/mtmd"),
        str(Path(binary).parent),
    ))

    if alias == "glm-5-2-2-788bpw":
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["GGML_CUDA_NO_PINNED"] = "1"
        return ([
            binary, "-m", str(entry["path"]), "--host", "127.0.0.1",
            "--port", str(PORT), "-c", str(ctx), "-ngl", "79",
            "-fa", "on", "-mla", "1", "-dsa", "-fidx",
            "-b", "512", "-ub", "256", "-t", "14",
            "-sm", "layer", "-ts", "1,1", "-mg", "0",
            "--cpu-moe", "--no-kv-offload", "-cram", "0",
        ], env)

    pinned_gpu = entry.get("gpu")
    if pinned_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(pinned_gpu)
        topology = ["--split-mode", "none", "--main-gpu", "0"]
    elif alias == "laguna-s2-1-fp16":
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Explicit tensor_split disables llama.cpp's automatic fitter.
        topology = ["--split-mode", "layer"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        topology = ["--split-mode", "layer", "--tensor-split", "1,1"]

    ngl = "999"
    threads = "16"
    extra: list[str] = []
    cache = ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]

    if alias == "laguna-s2-1-fp16":
        # Let --fit choose a safe CPU/GPU split for the 219 GiB FP16 model.
        ngl = "-1"
    elif alias == "laguna-s2-1-q4":
        # Keep attention/routing on the GPUs and only early MoE tensors on CPU.
        ngl = "999"
        threads = "10"
        extra += ["--n-cpu-moe", "46", "--no-kv-offload"]
    elif alias == "minimax-m3-q4":
        # Offload every attention/routing layer; 56 CPU MoE layers is the
        # maximum expert GPU residency that fits dual 24 GiB cards.
        ngl = "999"
        threads = "12"
        extra += ["--n-cpu-moe", "56", "--no-kv-offload"]
    elif alias == "glm-5-2-2-788bpw":
        # The current AtomicBot binary supports generic MLA/DSA detection;
        # legacy -mla/-dsa/-fidx/-muge switches were removed.
        extra += ["--cpu-moe", "--no-mmap", "-cram", "0"]
        cache = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
    if alias in {"grm-2-6-27b", "carwin-nano"}:
        extra += ["--spec-type", "draft-mtp"]

    args = [
        binary, "-m", str(entry["path"]), "--host", "127.0.0.1",
        "--port", str(PORT), "-c", str(ctx), "-ngl", ngl,
        "--fit", "on", "--flash-attn", "on", "--jinja", "--parallel", "1",
        "--threads", threads, "--threads-batch", "32", "-b", "512", "-ub", "256",
        *topology, *cache, *extra,
    ]
    mmproj = entry.get("mmproj")
    if mmproj and Path(mmproj).exists():
        args += ["--mmproj", str(mmproj)]
    return args, env


def launch(alias: str, entry: dict[str, Any], ctx: int, log_path: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    launcher = entry.get("launcher")
    if launcher == "ds4":
        args = [
            str(entry["binary"]), "--model", str(entry["path"]),
            "--host", "127.0.0.1", "--port", str(PORT), "--ctx", str(ctx),
            "--cuda", "--ssd-streaming", "--ssd-streaming-cache-experts", "48GB",
            "--kv-disk-dir", "/tmp/ds4-kv", "--kv-disk-space-mb", "8192",
        ]
    elif launcher == "colibri":
        env["COLI_MODEL"] = str(entry["path"])
        args = [
            str(entry["binary"]), "serve", "--host", "127.0.0.1", "--port", str(PORT),
            "--model-id", alias, "--ctx", str(ctx), "--auto-tier", "--ram", "280",
            "--gpu", "0,1", "--vram", "46", "--repin", "16", "--policy", "balanced",
            "--topp", "0.7",
        ]
    elif launcher == "prism-bonsai":
        model_dir = Path(entry["path"]).parent
        args = [
            "docker", "run", "--rm", "--name", "turbofit-catalog-speed-bench",
            "--gpus", "all", "-p", f"127.0.0.1:{PORT}:{PORT}",
            "-v", f"{model_dir}:/models:ro", "-e", f"PORT={PORT}", "-e", f"CTX={ctx}",
            "-e", f"MODEL=/models/{Path(entry['path']).name}",
        ]
        if entry.get("draft"):
            args += ["-e", f"DRAFT_MODEL=/models/{Path(entry['draft']).name}"]
        if entry.get("mmproj"):
            args += ["-e", f"MMPROJ=/models/{Path(entry['mmproj']).name}"]
        args += [str(entry.get("image") or "turbofit-prism-bonsai:local")]
        if "fp16" in alias:
            args += [
                "-ngl", "-1", "--split-mode", "layer", "--no-kv-offload",
            ]
    else:
        args, env = fixed_llama_args(alias, entry, ctx)

    log = log_path.open("wb")
    process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    process._turbofit_log = log  # type: ignore[attr-defined]
    return process


def terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
    log = getattr(process, "_turbofit_log", None)
    if log:
        log.close()
    cleanup_port()


def speed_request(alias: str, max_tokens: int = 256) -> dict[str, Any]:
    payload = {
        "model": alias, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0,
    }
    started = time.monotonic()
    response = request_json(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", payload, timeout=900
    )
    elapsed = time.monotonic() - started
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    tokens = int(usage.get("completion_tokens") or 0)
    decode = float(timings.get("predicted_per_second") or 0)
    prompt = float(timings.get("prompt_per_second") or 0)
    return {
        "completion_tokens": tokens,
        "wall_seconds": round(elapsed, 3),
        "wall_output_tok_s": round(tokens / elapsed, 3) if elapsed and tokens else 0,
        "decode_tok_s": round(decode, 3) if decode else None,
        "prompt_tok_s": round(prompt, 3) if prompt else None,
        "metric_source": "server_timings" if decode else "wall_clock",
    }


def save(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=65536)
    parser.add_argument("--only", default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("references/results/catalog-speed-sweep-latest.json"),
    )
    args = parser.parse_args()

    catalog = yaml.safe_load(CATALOG.read_text())["models"]
    only_terms = [term.strip().lower() for term in args.only.split(",") if term.strip()]
    selected = {
        alias: entry for alias, entry in catalog.items()
        if not only_terms or alias.lower() in only_terms
    }
    if args.list:
        for alias, entry in selected.items():
            print(f"{alias}\t{entry.get('launcher')}\t{entry.get('size_gb')} GB")
        return 0

    active_before = [name for name in PRODUCTION_SERVICES if service_active(name)]
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "2x NVIDIA GeForce RTX 3090 24GB",
        "context_tokens": args.ctx,
        "max_output_tokens": 256,
        "prompt": PROMPT,
        "results": [],
    }

    stop_services(PRODUCTION_SERVICES + CONFLICTING_SERVICES)
    cleanup_port()
    try:
        for index, (alias, entry) in enumerate(selected.items(), 1):
            print(f"[{index}/{len(selected)}] {alias}", flush=True)
            result: dict[str, Any] = {
                "model": alias,
                "launcher": entry.get("launcher"),
                "size_gb": entry.get("size_gb"),
                "status": "failed",
            }
            process: subprocess.Popen[bytes] | None = None
            started = time.monotonic()
            log_path = Path("/tmp") / f"turbofit-speed-{alias}.log"
            try:
                process = launch(alias, entry, args.ctx, log_path)
                timeout = 1800 if float(entry.get("size_gb") or 0) >= 200 else 900
                if not wait_ready(process, timeout):
                    raise RuntimeError(f"startup failed or exceeded {timeout}s; see {log_path}")
                result["startup_seconds"] = round(time.monotonic() - started, 3)
                # Warmup keeps model initialization out of the measured request.
                # Short generations keep very large/FP16 CPU-offloaded models
                # within the request timeout while still yielding stable decode TPS.
                is_very_large = float(entry.get("size_gb") or 0) >= 200
                is_fp16 = "fp16" in alias
                warmup_tokens = 4 if is_very_large or is_fp16 else 16
                speed_tokens = 32 if is_very_large or is_fp16 else 256
                request_json(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    {"model": alias, "messages": [{"role": "user", "content": "Reply OK"}],
                     "max_tokens": warmup_tokens, "temperature": 0},
                    timeout=600,
                )
                result.update(speed_request(alias, speed_tokens))
                result["requested_output_tokens"] = speed_tokens
                result["status"] = "ok"
                print(
                    f"  decode={result.get('decode_tok_s')} wall={result.get('wall_output_tok_s')} tok/s",
                    flush=True,
                )
            except Exception as exc:
                result["error"] = str(exc)
                result["log"] = str(log_path)
                print(f"  FAILED: {exc}", flush=True)
            finally:
                if process is not None:
                    terminate(process)
            payload["results"].append(result)
            save(args.output, payload)
    finally:
        restore_after_sweep(active_before)

    ok = sum(item["status"] == "ok" for item in payload["results"])
    print(f"complete: {ok}/{len(payload['results'])}; {args.output}", flush=True)
    return 0 if ok == len(payload["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
