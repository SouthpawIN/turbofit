#!/usr/bin/env bash
# install.sh — wire turbofit into your shell
set -e
SHELL_RC="${SHELL_RC:-${HOME}/.bashrc}"

# Resolve the skill directory the same way the shim does
_turbofit_find_skill_dir() {
    if [ -n "${HERMES_HOME:-}" ] && [ -f "${HERMES_HOME}/skills/turbofit/scripts/serve" ]; then
        echo "${HERMES_HOME}/skills/turbofit"
        return
    fi
    if [ -f "${HOME}/.hermes/skills/turbofit/scripts/serve" ]; then
        echo "${HOME}/.hermes/skills/turbofit"
        return
    fi
    local _profile
    for _profile in "${HOME}/.hermes/profiles/"*; do
        if [ -f "${_profile}/skills/turbofit/scripts/serve" ]; then
            echo "${_profile}/skills/turbofit"
            return
        fi
    done
    echo ""
}

SKILL_DIR="$(_turbofit_find_skill_dir)"
if [ -z "${SKILL_DIR}" ]; then
    echo "ERROR: could not locate turbofit skill directory."
    echo "Set HERMES_HOME or reinstall with 'hermes skills install turbofit'."
    exit 1
fi

# Source the shim (which now resolves SKILL_DIR dynamically)
SNIPPET="
# turbofit — added by turbofit skill (auto-generated)
[ -f \"${SKILL_DIR}/scripts/turbofit.sharco\" ] && source \"${SKILL_DIR}/scripts/turbofit.sharco\"
"

if ! grep -q "turbofit.sharco" "${SHELL_RC}" 2>/dev/null; then
    echo "" >> "${SHELL_RC}"
    echo "${SNIPPET}" >> "${SHELL_RC}"
    echo "✓ Wired serve/name functions into ${SHELL_RC}"
    echo "  Run 'source ~/.bashrc' or open a new shell to use them."
else
    # Update the path in case the install location changed
    # Remove old turbofit block and re-add with current path
    sed -i '/# turbofit — added by turbofit skill/,/source.*turbofit\.sharco/d' "${SHELL_RC}" 2>/dev/null || true
    echo "" >> "${SHELL_RC}"
    echo "${SNIPPET}" >> "${SHELL_RC}"
    echo "✓ Updated serve/name path in ${SHELL_RC}"
fi

# Bootstrap config directory with empty catalog if missing
TURBOFIT_CONFIG_DIR="${HOME}/.config/turbofit"
mkdir -p "${TURBOFIT_CONFIG_DIR}"

if [ ! -f "${TURBOFIT_CONFIG_DIR}/models.yaml" ]; then
    cat > "${TURBOFIT_CONFIG_DIR}/models.yaml" <<'YAML'
# turbofit model catalog
# Register models with: serve register <alias> <path> [--launcher llama-cpp] [--port N]
# Or edit this file directly.
# Schema: https://github.com/SouthpawIN/turbofit#catalog-schema
models: {}
YAML
    echo "✓ Created empty catalog at ${TURBOFIT_CONFIG_DIR}/models.yaml"
fi

if [ ! -f "${TURBOFIT_CONFIG_DIR}/curated.yaml" ]; then
    cat > "${TURBOFIT_CONFIG_DIR}/curated.yaml" <<'YAML'
# turbofit curated slots — used by `serve auto` to pick models
# Define which model fills each role for your hardware.
# Leave slots empty to let serve auto fall back to API mode.
slots:
  main_local:
    # alias: my-27b-model        # uncomment and set to a catalog alias
    archetype: "27-28B dense (Q4)"
    tier: s
  aux_local:
    # alias: my-35b-moe
    archetype: "35B MoE (3B active)"
    tier: sd
  main_api:
    archetype: "DeepSeek V4 Pro"
    api_id: deepseek-ai/deepseek-v4-pro
    tier: s
  aux_api:
    archetype: "MiniMax M3"
    api_id: minimaxai/minimax-m3
    tier: sf
YAML
    echo "✓ Created curated slots at ${TURBOFIT_CONFIG_DIR}/curated.yaml"
fi

echo ""
echo "turbofit is ready. Key commands:"
echo "  serve auto main    # detect hardware, pick best model, launch"
echo "  serve catalog       # browse registered models"
echo "  serve vram          # check GPU VRAM"
echo "  serve help          # full command reference"
