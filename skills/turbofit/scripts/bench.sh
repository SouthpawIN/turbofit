#!/usr/bin/env bash
# Bench one model at a time on GPU1
set -uo pipefail

GGUF="/home/sovthpaw/Models/storage/gguf"
ATOMIC="/home/sovthpaw/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server"
STOCK="/home/sovthpaw/projects/LLM-Infra/llama.cpp/build/bin/llama-server"
LIB_ATOMIC="/home/sovthpaw/projects/LLM-Infra/llama.cpp-atomic/build/bin"
LIB_STOCK="/home/sovthpaw/projects/LLM-Infra/llama.cpp/build/bin"
PORT=11999
CTX=131072
PROMPT="Write a Python function that implements merge sort with type hints and docstrings."

bench_model() {
    local name="$1" path="$2" binary="$3" presets="$4" mmproj="$5"
    
    echo "=== $name ==="
    if [ ! -f "$path" ]; then
        echo "  SKIP: file not found"
        echo ""
        return
    fi
    
    local size_gb=$(du -h "$path" | cut -f1)
    local lib_dir
    if [ "$binary" = "$ATOMIC" ]; then lib_dir="$LIB_ATOMIC"; else lib_dir="$LIB_STOCK"; fi
    
    fuser -k $PORT/tcp 2>/dev/null; sleep 2
    
    echo "  Launching ($size_gb, $(basename $binary))..."
    LLAMA_LIB_DIR="$lib_dir" LD_LIBRARY_PATH="$lib_dir" CUDA_VISIBLE_DEVICES=1 \
    timeout 120 "$binary" \
        -m "$path" --host 127.0.0.1 --port $PORT -ngl 999 -fa on -c $CTX --jinja \
        $presets --main-gpu 0 $([ -f "$mmproj" ] && echo "--mmproj $mmproj") \
        > /dev/null 2>&1 &
    local pid=$!
    
    # Wait for ready
    local ready=false
    for i in $(seq 1 45); do
        if curl -s --max-time 2 http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
            ready=true; break
        fi
        sleep 2
    done
    
    if [ "$ready" = false ]; then
        echo "  FAILED to start (timeout)"
        kill $pid 2>/dev/null; sleep 3
        echo ""
        return
    fi
    
    # Warmup — send tiny request and wait for it to succeed
    echo -n "  Warming up... "
    warmup_ok=false
    for i in $(seq 1 10); do
        w=$(python3 -c "
import json
from urllib.request import urlopen, Request
try:
    data = json.dumps({'model':'t','messages':[{'role':'user','content':'hi'}],'max_tokens':5}).encode()
    req = Request('http://127.0.0.1:$PORT/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})
    urlopen(req, timeout=30)
    print('ok')
except Exception as e:
    print('fail')
" 2>&1)
        if [ "$w" = "ok" ]; then warmup_ok=true; break; fi
        sleep 3
    done
    echo "$warmup_ok"
    
    if [ "$warmup_ok" = false ]; then
        echo "  FAILED warmup (503 — model not ready)"
        kill $pid 2>/dev/null; sleep 3
        echo ""
        return
    fi
    
    echo -n "  Benchmarking... "
    result=$(python3 -c "
import json, time
from urllib.request import urlopen, Request
data = json.dumps({'model':'t','messages':[{'role':'user','content':'$PROMPT'}],'max_tokens':256,'temperature':0.1}).encode()
req = Request('http://127.0.0.1:$PORT/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})
start = time.time()
resp = urlopen(req, timeout=60)
elapsed = time.time() - start
result = json.loads(resp.read())
tokens = result.get('usage',{}).get('completion_tokens',0)
tok_s = tokens/elapsed if elapsed > 0 else 0
print(f'{tok_s:.1f}')
" 2>&1)
    
    echo "$result tok/s"
    kill $pid 2>/dev/null; sleep 5
    echo ""
}

echo "TURBOFIT BENCHMARK — $(date +%Y-%m-%d)"
echo "Context: $CTX tokens | GPU1 only | Prompt: 256 max tokens"
echo ""

bench_model "Qwopus Coder" \
    "$GGUF/Qwopus3.6-27B-Coder-MTP/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf" \
    "$ATOMIC" "--spec-type nextn --draft-block-size 3 --no-mmap" \
    "$GGUF/Qwopus3.6-27B-Coder-MTP/mmproj-F32.gguf"

bench_model "Qwopus v2" \
    "$GGUF/Qwopus3.6-27B-v2-MTP/Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf" \
    "$ATOMIC" "--spec-type nextn --draft-block-size 3 --no-mmap" \
    "$GGUF/Qwopus3.6-27B-v2-MTP/mmproj-F32.gguf"

bench_model "Abliterated Qwable" \
    "$GGUF/Huihui-Qwable-3.6-27b-abliterated-MTP/Huihui-Qwable-3.6-27b-abliterated-Q4_K_M_Q8-MTP.gguf" \
    "$ATOMIC" "--spec-type nextn --draft-block-size 3 --no-mmap" \
    "$GGUF/Huihui-Qwable-3.6-27b-abliterated-MTP/mmproj-model-f16.gguf"

bench_model "Qwable Coder" \
    "$GGUF/Qwable-5-27B-Coder-GGUF/Qwable-5-27B-Coder-Q4_K_M.gguf" \
    "$STOCK" "--no-mmap" \
    "$GGUF/Qwable-5-27B-Coder-GGUF/mmproj-Qwable-5-27B-Coder-f16.gguf"

bench_model "Carnice V2" \
    "$GGUF/Carnice-V2-27b-GGUF/carnice-v2-27b-Q4_K_M.gguf" \
    "$STOCK" "--no-mmap" \
    "$GGUF/mmproj-BF16.gguf"

bench_model "Carwin" \
    "$GGUF/Carwin-28B-MTP-GGUF/carwin-Q4_K_M.gguf" \
    "$ATOMIC" "--spec-type nextn --draft-block-size 3 --no-mmap" \
    "$GGUF/Carwin-28B-MTP-GGUF/carwin-mmproj-f16.gguf"

bench_model "Carwin Nano" \
    "$GGUF/Carwin-MoE-Nano-GGUF/carwin-moe-Nano.gguf" \
    "$STOCK" "-ctk q8_0 -ctv q8_0 --no-mmap --split-mode none --n-cpu-moe 4" \
    "$GGUF/mmproj-BF16.gguf"

# Check if Holo is downloaded
if [ -f "$GGUF/Holo-3.1-35B-A3B-GGUF/q4_k_m.gguf" ]; then
    bench_model "Holo 3.1" \
        "$GGUF/Holo-3.1-35B-A3B-GGUF/q4_k_m.gguf" \
        "$STOCK" "--no-mmap --split-mode none --n-cpu-moe 4" \
        "$GGUF/Holo-3.1-35B-A3B-GGUF/mmproj.f16.gguf"
else
    echo "=== Holo 3.1 ==="
    echo "  SKIP: still downloading"
    du -sh "$GGUF/Holo-3.1-35B-A3B-GGUF/" 2>/dev/null
fi

echo "=== Darwin Reason (already running, GPU0) ==="
echo -n "  "
python3 -c "
import json, time
from urllib.request import urlopen, Request
data = json.dumps({'model':'t','messages':[{'role':'user','content':'$PROMPT'}],'max_tokens':256,'temperature':0.1}).encode()
req = Request('http://127.0.0.1:11500/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})
start = time.time()
resp = urlopen(req, timeout=60)
elapsed = time.time() - start
result = json.loads(resp.read())
tokens = result.get('usage',{}).get('completion_tokens',0)
print(f'{tokens/elapsed:.1f} tok/s')
" 2>&1

echo ""
echo "DONE"
