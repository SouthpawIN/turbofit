#!/bin/bash
set -euo pipefail
ROOT=/home/sovthpaw/projects/turbofit
cd "$ROOT"
export PYTHONPATH=src:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
LOG="$ROOT/logs/h3-verify-and-promo.log"
exec >>"$LOG" 2>&1
echo "START $(date -Is)"
"$ROOT/.venv-h3/bin/python" -u scripts/verify-h3-live --smoke --device cuda:0
if [ ! -f "$ROOT/promo/h3-verify/bonsai-8gb-smoke.mp4" ]; then
  echo "SMOKE FAILED"
  exit 1
fi
echo "SMOKE OK"
"$ROOT/.venv-h3/bin/python" -u scripts/generate-h3-promo-clips \
  --device cuda:0 \
  --prompts promo/h3-prompts-2.3.json \
  --output-dir promo/h3-local-2.3
"$ROOT/.venv-h3/bin/python" -u scripts/build-promo-video-2.3
OUT="$ROOT/promo/turbofit-2.3-h3-local-promo.mp4"
if [ ! -f "$OUT" ]; then
  echo "NO MASTER"
  exit 1
fi
# Discord free cap 25MB. Re-encode a DM copy if needed.
DM="$ROOT/promo/turbofit-2.3-h3-local-promo-discord.mp4"
SIZE=$(stat -c%s "$OUT")
if [ "$SIZE" -gt 24000000 ]; then
  ffmpeg -y -i "$OUT" -c:v libx264 -crf 28 -preset fast -c:a aac -b:a 128k -movflags +faststart "$DM"
else
  cp -f "$OUT" "$DM"
fi
HERMES_HOME=/home/sovthpaw/.hermes/profiles/senter hermes send --json \
  -t "discord:1358311720665219122" \
  "TurboFit 2.3 promo. Local MiniMax H3 only. New clips. MEDIA:$DM"
echo "DONE $(date -Is)"
