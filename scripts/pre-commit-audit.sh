#!/usr/bin/env bash
set -euo pipefail

# Simple pre-commit secret audit (Phase 8)
#
# Scans staged changes for:
# - NVIDIA API keys and env lines
# - Bearer tokens, http/https api keys
# - Common secret patterns
#
# Usage:
#   echo scripts/pre-commit-audit.sh > .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#   (or add as a local pre-commit repo hook)

files=$(git diff --cached --name-only | tr '\n' ' ')
if [[ -z "$files" ]]; then
  exit 0
fi

fail=false

scan() {
  local pattern="$1"; shift
  local desc="$1"; shift
  if rg -n "$pattern" $files >/dev/null 2>&1; then
    echo "[audit] Potential secret detected ($desc)."
    rg -n "$pattern" $files || true
    fail=true
  fi
}

# Patterns
scan 'NVIDIA_API_KEY\s*=' 'env var assignment'
scan 'nvapi-' 'NVIDIA API key prefix'
scan 'Bearer\s+[A-Za-z0-9._-]{10,}' 'Bearer token in code'
scan 'api_key\s*[:=]\s*[\"\'`][A-Za-z0-9._-]{10,}' 'api_key literal'
scan 'http(s)?://[^ ]*(token|key|secret)=' 'URL with token/key/secret parameter'

if [[ "$fail" == true ]]; then
  echo "\n[audit] Blocked by pre-commit audit. Mask or remove secrets from staged changes."
  exit 1
fi

exit 0
