#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

echo "🔐 Verifying GitHub auth…" | mask
run_gh api user --silent >/dev/null 2>&1 && echo "✅ Auth OK" || { echo "❌ Auth failed"; exit 1; }
