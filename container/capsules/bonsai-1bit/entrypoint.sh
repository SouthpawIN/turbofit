#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL="${MODEL:-${MODEL_DIR}/Bonsai-27B-Q1_0.gguf}"
MMPROJ="${MMPROJ:-${MODEL_DIR}/Bonsai-27B-mmproj-Q8_0.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
PORT="${PORT:-11610}"
CTX="${CTX:-262144}"
MAIN_GPU="${MAIN_GPU:-0}"
NGL="${NGL:-99}"
DRAFT_NGL="${DRAFT_NGL:-99}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-4}"
THREADS="${THREADS:-$(nproc)}"

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

exec /opt/turbofit/runtime/llama-server "${args[@]}" "$@"
