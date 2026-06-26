#!/usr/bin/env bash
# install.sh — wire turbofit into your shell
set -e
SHELL_RC="${SHELL_RC:-${HOME}/.bashrc}"

# Resolve the skill directory relative to where THIS file lives
_turbofit_this_file="${BASH_SOURCE[0]:-$0}"
_turbofit_this_dir=""
if [ -n "${_turbofit_this_file}" ]; then
    _turbofit_this_dir="$(cd "$(dirname "${_turbofit_this_file}")" 2>/dev/null && pwd)"
fi

if [ -n "${_turbofit_this_dir}" ] && [ -f "${_turbofit_this_dir}/serve" ]; then
    SKILL_DIR="$(dirname "${_turbofit_this_dir}")"
elif [ -n "${HERMES_HOME:-}" ] && [ -f "${HERMES_HOME}/skills/turbofit/scripts/serve" ]; then
    SKILL_DIR="${HERMES_HOME}/skills/turbofit"
elif [ -f "${HOME}/.hermes/skills/turbofit/scripts/serve" ]; then
    SKILL_DIR="${HOME}/.hermes/skills/turbofit"
else
    # Walk profiles
    SKILL_DIR=""
    for _turbofit_profile in "${HOME}/.hermes/profiles/"*; do
        if [ -f "${_turbofit_profile}/skills/turbofit/scripts/serve" ]; then
            SKILL_DIR="${_turbofit_profile}/skills/turbofit"
            break
        fi
    done
fi

if [ -z "${SKILL_DIR}" ]; then
    echo "ERROR: could not locate turbofit skill directory."
    echo "Set HERMES_HOME or reinstall with 'hermes skills install turbofit'."
    exit 1
fi

# Source the shim (which resolves SKILL_DIR dynamically via BASH_SOURCE)
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
    sed -i '/# turbofit — added by turbofit skill/,/source.*turbofit\.sharco/d' "${SHELL_RC}" 2>/dev/null || true
    echo "" >> "${SHELL_RC}"
    echo "${SNIPPET}" >> "${SHELL_RC}"
    echo "✓ Updated serve/name path in ${SHELL_RC}"
fi

# Bootstrap config directory with catalog and curated slots if missing
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
# Each slot has tier keys (s/sf/sd/f/c) with lists of entries.
# Edit to match your local models and API preferences.
slots:
  # LOCAL MAIN — your GPU models, ranked by tier
  main_local:
    description: Local main — runs on your GPU. Register models with `serve register`.
    # Uncomment and set to your models:
    # s:
    # - alias: my-27b-model
    #   why: Smartest model I have
    #   tok_s_target: 38
    #   ctx_target: 262144
    #   vision: true
    #   vram_gb: 17

  # LOCAL AUX — secondary model for vision/aux tasks
  aux_local:
    description: Local aux — secondary model (vision, compression, etc.)
    # sd:
    # - alias: my-35b-moe
    #   why: MoE 35B/3B-active, vision, 1M ctx
    #   tok_s_target: 110
    #   ctx_target: 262144
    #   vision: true
    #   vram_gb: 11

  # API MAIN — used when no local GPU or --api flag
  main_api:
    description: API main picks — free models first, ranked by reasoning quality.
    s:
    - alias: deepseek-v4-pro
      provider: nvidia-nim
      model_id: deepseek-ai/deepseek-v4-pro
      why: Best free reasoning + coding. 1M ctx. Free on NIM.
      vision: false
      free: true
    sf:
    - alias: deepseek-v4-flash
      provider: nvidia-nim
      model_id: deepseek-ai/deepseek-v4-flash
      why: Fast reasoning, free tier.
      vision: false
      free: true

  # API AUX — vision-capable, free first
  aux_api:
    description: API aux picks — vision required, free first.
    sf:
    - alias: minimax-m3
      provider: nvidia-nim
      model_id: minimaxai/minimax-m3
      why: Free vision model. 1M ctx, image+video.
      vision: true
      free: true
YAML
    echo "✓ Created curated slots at ${TURBOFIT_CONFIG_DIR}/curated.yaml"
    echo "  Edit this file to add your local models."
fi

echo ""
echo "turbofit is ready. Key commands:"
echo "  serve auto main    # detect hardware, pick best model, launch"
echo "  serve catalog       # browse registered models"
echo "  serve vram          # check GPU VRAM"
echo "  serve help          # full command reference"
