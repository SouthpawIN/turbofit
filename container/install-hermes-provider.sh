#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${TURBOFIT_BASE_URL:-http://127.0.0.1:8091}"
hermes config set custom_providers.0.name turbofit
hermes config set custom_providers.0.base_url "${BASE_URL}/v1"
hermes config set custom_providers.0.api_key not-needed
hermes config set model.provider custom:turbofit
printf 'Hermes now routes through Turbofit at %s/v1\n' "$BASE_URL"