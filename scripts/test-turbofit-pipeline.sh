#!/usr/bin/env bash
# test-turbofit-pipeline.sh — End-to-end test of the Turbofit pipeline
# Tests: local gateway → remote backend (Omarchy) → Tailscale serve → chat completion
#
# Usage:
#   ./test-turbofit-pipeline.sh              # Test local gateway + Omarchy
#   ./test-turbofit-pipeline.sh --tailscale  # Also test Tailscale serve
#   ./test-turbofit-pipeline.sh --strata     # Test Strata (if installed)

set -euo pipefail

GATEWAY_PORT="${GATEWAY_PORT:-8091}"
OMARCHY_URL="${TURBOFIT_REMOTE_BACKEND_URL:-http://omarchy.tail6ff78b.ts.net:8080/v1}"
STRATA_URL="${TURBOFIT_STRATA_BACKEND_URL:-http://omarchy.tail6ff78b.ts.net:8082/v1}"
TAILSCALE_URL="https://christophers-macbook-pro.tail6ff78b.ts.net:9443/v1"

log() { echo "[test] $*"; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; FAILED=1; }

FAILED=0

# Print the assistant content from an OpenAI-compatible chat response, truncated
# to the optional character limit. Parsing happens out of the shell so a response
# body is never piped into an interpreter: Hermes' plugin installer blocks
# curl|python and echo|python as critical supply-chain findings, which made the
# plugin non-installable.
chat_content() {
    python3 - "$1" "${2:-100}" <<'PY' 2>/dev/null
import json
import sys

try:
    content = json.loads(sys.argv[1])["choices"][0]["message"]["content"]
except (ValueError, LookupError, KeyError, TypeError):
    raise SystemExit(1)
print(content[: int(sys.argv[2])])
PY
}

# Test 1: Local gateway is running
log "Test 1: Local gateway on :$GATEWAY_PORT"
if curl -s --max-time 3 "http://127.0.0.1:$GATEWAY_PORT/v1/models" &>/dev/null; then
    pass "Local gateway is up"
else
    fail "Local gateway not responding on :$GATEWAY_PORT"
fi

# Test 2: Omarchy remote backend
log "Test 2: Omarchy backend at $OMARCHY_URL"
if curl -s --max-time 5 "$OMARCHY_URL/models" &>/dev/null; then
    pass "Omarchy backend is reachable"
else
    fail "Omarchy backend not reachable"
fi

# Test 3: Chat completion through Omarchy
log "Test 3: Chat completion through Omarchy"
RESPONSE=$(curl -s --max-time 30 "$OMARCHY_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.8-27B-Unleashed-UD-Q3_K_XL.gguf",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 64,
        "temperature": 0
    }' 2>&1)
if CONTENT=$(chat_content "$RESPONSE" 100); then
    pass "Chat completion works: \"$CONTENT\""
else
    fail "Chat completion failed: $RESPONSE"
fi

# Test 4: Tailscale serve (optional)
if [[ "${1:-}" == "--tailscale" ]]; then
    log "Test 4: Tailscale serve at $TAILSCALE_URL"
    if curl -s --max-time 5 "$TAILSCALE_URL/models" &>/dev/null; then
        pass "Tailscale serve is up"
    else
        fail "Tailscale serve not responding"
    fi

    log "Test 5: Chat completion through Tailscale"
    RESPONSE=$(curl -s --max-time 30 "$TAILSCALE_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "auto",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 64,
            "temperature": 0
        }' 2>&1)
    if CONTENT=$(chat_content "$RESPONSE" 100); then
        pass "Tailscale chat completion works: \"$CONTENT\""
    else
        fail "Tailscale chat completion failed: $RESPONSE"
    fi
fi

# Test 6: Strata (optional)
if [[ "${1:-}" == "--strata" ]]; then
    log "Test 6: Strata backend at $STRATA_URL"
    if curl -s --max-time 5 "$STRATA_URL/models" &>/dev/null; then
        pass "Strata backend is reachable"
    else
        fail "Strata backend not reachable (install with: ./scripts/install-strata.sh)"
    fi

    log "Test 7: Chat completion through Strata"
    RESPONSE=$(curl -s --max-time 60 "$STRATA_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen3.8-flash-next-iq2_xs",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 64,
            "temperature": 0
        }' 2>&1)
    if CONTENT=$(chat_content "$RESPONSE" 100); then
        pass "Strata chat completion works: \"$CONTENT\""
    else
        fail "Strata chat completion failed: $RESPONSE"
    fi
fi

# Test 8: Engine check (includes Strata)
log "Test 8: Engine check includes Strata"
if grep -q "strata" ~/turbofit/src/turbofit_runtime/engine_check.py; then
    pass "Strata is in engine check"
else
    fail "Strata not found in engine check"
fi

# Test 9: GSQ-RCO in fit check
log "Test 9: GSQ-RCO in fit check"
if grep -q "gsq-rco" ~/turbofit/src/turbofit_runtime/fit_check.py; then
    pass "GSQ-RCO is in fit check"
else
    fail "GSQ-RCO not found in fit check"
fi

# Test 10: GSQ-RCO in recipes
log "Test 10: GSQ-RCO in recipes"
if grep -q "gsq-rco" ~/turbofit/references/model-recipes.json; then
    pass "GSQ-RCO is in recipes"
else
    fail "GSQ-RCO not found in recipes"
fi

# Summary
echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Check output above."
fi
exit $FAILED
