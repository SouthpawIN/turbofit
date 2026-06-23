#!/usr/bin/env bash
# install.sh — wire turbofit into your shell
set -e
SHELL_RC="${HOME}/.bashrc"
SNIPPET='
# turbofit — added by turbofit skill (auto-generated)
[ -f "${HOME}/.hermes/skills/turbofit/scripts/turbofit.sharco" ] && source "${HOME}/.hermes/skills/turbofit/scripts/turbofit.sharco"
'

if ! grep -q "turbofit.sharco" "${SHELL_RC}" 2>/dev/null; then
    echo "" >> "${SHELL_RC}"
    echo "${SNIPPET}" >> "${SHELL_RC}"
    echo "✓ Wired serve/name functions into ${SHELL_RC}"
    echo "  Run 'source ~/.bashrc' or open a new shell to use them."
else
    echo "✓ serve/name already wired in ${SHELL_RC}"
fi
