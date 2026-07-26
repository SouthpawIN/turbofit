#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL="${MODEL:-${MODEL_DIR}/Bonsai-27B-Q1_0.gguf}"
MMPROJ="${MMPROJ:-${MODEL_DIR}/Bonsai-27B-mmproj-Q8_0.gguf}"
PORT="${PORT:-11610}"
CTX="${CTX:-262144}"

exec /opt/turbofit/runtime/llama-server \
  -m "$MODEL" \
  --host "${HOST:-0.0.0.0}" --port "$PORT" \
  -ngl "${NGL:-99}" -fa on -c "$CTX" --jinja -t "${THREADS:-$(nproc)}" \
  --fit on --split-mode none --main-gpu "${MAIN_GPU:-0}" \
  --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1 \
  --mmproj "$MMPROJ" \
  "$@"