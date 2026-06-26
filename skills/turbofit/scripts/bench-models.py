#!/usr/bin/env python3
"""Turbofit benchmark — launches each model, warms up, benchmarks, kills."""
import json, time, os, subprocess, sys

PORT = 11999
CTX = 131072
ATOMIC = "/home/sovthpaw/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
STOCK = "/home/sovthpaw/projects/LLM-Infra/llama.cpp/build/bin/llama-server"
LIB_ATOMIC = "/home/sovthpaw/projects/LLM-Infra/llama.cpp-atomic/build/bin"
LIB_STOCK = "/home/sovthpaw/projects/LLM-Infra/llama.cpp/build/bin"
GGUF = "/home/sovthpaw/Models/storage/gguf"

# All models use turbo4-kv for atomic (saves VRAM), q8-kv for stock
# No spec decoding (nextn loads model twice -> OOM at 131K)
MODELS = [
    {"name": "Qwopus Coder", "path": f"{GGUF}/Qwopus3.6-27B-Coder-MTP/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf",
     "binary": "atomic", "kv": "turbo4", "mmproj": f"{GGUF}/Qwopus3.6-27B-Coder-MTP/mmproj-F32.gguf"},
    {"name": "Qwopus v2", "path": f"{GGUF}/Qwopus3.6-27B-v2-MTP/Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf",
     "binary": "atomic", "kv": "turbo4", "mmproj": f"{GGUF}/Qwopus3.6-27B-v2-MTP/mmproj-F32.gguf"},
    {"name": "Abliterated Qwable", "path": f"{GGUF}/Huihui-Qwable-3.6-27b-abliterated-MTP/Huihui-Qwable-3.6-27b-abliterated-Q4_K_M_Q8-MTP.gguf",
     "binary": "atomic", "kv": "turbo4", "mmproj": f"{GGUF}/Huihui-Qwable-3.6-27b-abliterated-MTP/mmproj-model-f16.gguf"},
    {"name": "Carwin", "path": f"{GGUF}/Carwin-28B-MTP-GGUF/carwin-Q4_K_M.gguf",
     "binary": "atomic", "kv": "turbo4", "mmproj": f"{GGUF}/Carwin-28B-MTP-GGUF/carwin-mmproj-f16.gguf"},
    {"name": "Prism Eagle", "path": f"{GGUF}/Prism-Eagle-27B/Qwen3.6-27B-PRISM-PRO-DQ.gguf",
     "binary": "atomic", "kv": "turbo4", "mmproj": f"{GGUF}/Prism-Eagle-27B/mmproj-F32.gguf"},
    {"name": "Qwable Coder", "path": f"{GGUF}/Qwable-5-27B-Coder-GGUF/Qwable-5-27B-Coder-Q4_K_M.gguf",
     "binary": "stock", "kv": "q8_0", "mmproj": f"{GGUF}/Qwable-5-27B-Coder-GGUF/mmproj-Qwable-5-27B-Coder-f16.gguf"},
    {"name": "Carnice V2", "path": f"{GGUF}/Carnice-V2-27b-GGUF/carnice-v2-27b-Q4_K_M.gguf",
     "binary": "stock", "kv": "q8_0", "mmproj": f"{GGUF}/mmproj-BF16.gguf"},
    {"name": "Carwin Nano", "path": f"{GGUF}/Carwin-MoE-Nano-GGUF/carwin-moe-Nano.gguf",
     "binary": "stock", "kv": "q8_0", "mmproj": f"{GGUF}/mmproj-BF16.gguf", "extra": "--split-mode none --n-cpu-moe 4"},
    {"name": "Holo 3.1", "path": f"{GGUF}/Holo-3.1-35B-A3B-GGUF/q4_k_m.gguf",
     "binary": "stock", "kv": "q8_0", "mmproj": f"{GGUF}/Holo-3.1-35B-A3B-GGUF/mmproj.f16.gguf", "extra": "--split-mode none --n-cpu-moe 4"},
]

PROMPT = "Write a Python function that implements merge sort with type hints and docstrings."

def kill_port():
    os.system(f"fuser -k {PORT}/tcp 2>/dev/null; sleep 3")

def launch(model):
    binary = ATOMIC if model["binary"] == "atomic" else STOCK
    lib_dir = LIB_ATOMIC if model["binary"] == "atomic" else LIB_STOCK
    env = os.environ.copy()
    env["LLAMA_LIB_DIR"] = lib_dir
    env["LD_LIBRARY_PATH"] = lib_dir
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    cmd = [
        binary, "-m", model["path"],
        "--host", "127.0.0.1", "--port", str(PORT),
        "-ngl", "999", "-fa", "on", "-c", str(CTX), "--jinja",
        "-ctk", model["kv"], "-ctv", model["kv"],
        "--no-mmap", "--main-gpu", "0",
    ]
    if model.get("extra"):
        cmd.extend(model["extra"].split())
    if model.get("mmproj") and os.path.exists(model["mmproj"]):
        cmd.extend(["--mmproj", model["mmproj"]])
    
    print(f"  Launching ({model['binary']}, {model['kv']}-kv)...", end=" ", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc

def wait_ready(timeout=90):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3)
            return True
        except:
            pass
        time.sleep(2)
    return False

def warmup():
    import urllib.request
    for _ in range(10):
        try:
            data = json.dumps({"model":"t","messages":[{"role":"user","content":"hi"}],"max_tokens":5}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=data, headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req, timeout=30)
            return True
        except:
            time.sleep(3)
    return False

def bench():
    import urllib.request
    data = json.dumps({"model":"t","messages":[{"role":"user","content":PROMPT}],"max_tokens":256,"temperature":0.1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=data, headers={"Content-Type":"application/json"})
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    elapsed = time.time() - start
    result = json.loads(resp.read())
    tokens = result.get("usage",{}).get("completion_tokens",0)
    return tokens, elapsed, tokens/elapsed if elapsed > 0 else 0

def main():
    print(f"Turbofit Benchmark — {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Context: {CTX} | GPU1 | No spec decode (turbo4/q8 KV only)")
    print(f"{'='*60}")
    
    results = []
    for m in MODELS:
        print(f"\n[{m['name']}]")
        if not os.path.exists(m["path"]):
            print(f"  SKIP: file not found")
            results.append({"name": m["name"], "status": "missing"})
            continue
        
        size_gb = os.path.getsize(m["path"]) / 1e9
        kill_port()
        proc = launch(m)
        
        if not wait_ready():
            print(f"FAILED to start")
            proc.kill()
            results.append({"name": m["name"], "status": "failed"})
            continue
        
        print("ready. Warming up...", end=" ", flush=True)
        if not warmup():
            print("WARMUP FAILED")
            proc.kill()
            results.append({"name": m["name"], "status": "warmup_failed"})
            continue
        
        print("benching...", end=" ", flush=True)
        try:
            tokens, elapsed, tok_s = bench()
            print(f"{tok_s:.1f} tok/s ({tokens} tok, {elapsed:.1f}s)")
            results.append({"name": m["name"], "status": "ok", "tok_s": round(tok_s,1), "tokens": tokens, "time_s": round(elapsed,2), "size_gb": round(size_gb,1)})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"name": m["name"], "status": "error", "error": str(e)[:100]})
        
        proc.kill()
        proc.wait()
        kill_port()
    
    # Darwin Reason (already running on GPU0)
    print(f"\n[Darwin Reason] (GPU0, already running)")
    try:
        import urllib.request
        # Warmup
        data = json.dumps({"model":"t","messages":[{"role":"user","content":"hi"}],"max_tokens":5}).encode()
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11500/v1/chat/completions", data=data, headers={"Content-Type":"application/json"}), timeout=30)
        tokens, elapsed, tok_s = bench_darwin()
        print(f"  {tok_s:.1f} tok/s ({tokens} tok, {elapsed:.1f}s)")
        results.append({"name": "Darwin Reason", "status": "ok", "tok_s": round(tok_s,1), "tokens": tokens, "time_s": round(elapsed,2), "size_gb": 16.6})
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS — tok/s at {CTX//1000}K context")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'tok/s':>8} {'Size':>6}")
    print(f"{'-'*45}")
    for r in sorted(results, key=lambda x: x.get("tok_s",0), reverse=True):
        if r["status"] == "ok":
            print(f"{r['name']:<25} {r['tok_s']:>7.1f} {r.get('size_gb','?'):>5.1f}G")
        else:
            print(f"{r['name']:<25} {'—':>8} {'?':>6}  ({r['status']})")
    
    # Save
    out = os.path.expanduser("~/.hermes/skills/turbofit/references/benchmark-results.json")
    with open(out, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d"), "ctx": CTX, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")

def bench_darwin():
    import urllib.request
    data = json.dumps({"model":"t","messages":[{"role":"user","content":PROMPT}],"max_tokens":256,"temperature":0.1}).encode()
    req = urllib.request.Request("http://127.0.0.1:11500/v1/chat/completions", data=data, headers={"Content-Type":"application/json"})
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    elapsed = time.time() - start
    result = json.loads(resp.read())
    tokens = result.get("usage",{}).get("completion_tokens",0)
    return tokens, elapsed, tokens/elapsed if elapsed > 0 else 0

if __name__ == "__main__":
    main()
