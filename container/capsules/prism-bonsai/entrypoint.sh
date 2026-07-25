#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL="${MODEL:-${MODEL_DIR}/Ternary-Bonsai-27B-Q2_0.gguf}"
MMPROJ="${MMPROJ:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
PORT="${PORT:-11611}"
CTX="${CTX:-65536}"
MAIN_GPU="${MAIN_GPU:-0}"
NGL="${NGL:-99}"
DRAFT_NGL="${DRAFT_NGL:-99}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
THREADS="${THREADS:-32}"

args=(
  -m "$MODEL"
  --host "${HOST:-0.0.0.0}" --port "$PORT"
  -ngl "$NGL" -fa on -c "$CTX" --jinja -t "$THREADS"
  --fit on --split-mode none --main-gpu "$MAIN_GPU"
  --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1
)

if [[ -n "$MMPROJ" ]]; then
  args+=(--mmproj "$MMPROJ")
fi

if [[ -n "$DRAFT_MODEL" ]]; then
  args+=(
    --model-draft "$DRAFT_MODEL"
    --spec-type draft-dspark
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
    -ngld "$DRAFT_NGL"
  )
fi

exec /opt/prism-llama.cpp/build/bin/llama-server "${args[@]}" "$@"
