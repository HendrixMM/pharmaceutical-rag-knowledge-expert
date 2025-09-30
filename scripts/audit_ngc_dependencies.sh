#!/usr/bin/env bash
# Audit NGC Dependencies
# Purpose: Scan the repository for NGC-dependent patterns to ensure NGC independence.
# Exit codes: 0 = no NGC deps found, 1 = NGC deps detected
# Usage: scripts/audit_ngc_dependencies.sh [-v]

set -euo pipefail
VERBOSE=0
if [[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]]; then VERBOSE=1; fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*"; }

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
info "NGC audit started at $TS"

# File filters and exclusions
EXCLUDES=(backups \/venv\/ \/.git\/ \/.pytest_cache\/ \/__pycache__\/ docs/NGC_DEPRECATION_IMMUNITY.md)
EXPR_EXCLUDE=$(printf "|%s" "${EXCLUDES[@]}")
EXPR_EXCLUDE="${EXPR_EXCLUDE:1}"

scan() {
  local pattern="$1"
  rg -n --hidden --glob '!venv/*' --glob '!*__pycache__/*' --glob '!backups/*' "$pattern" || true
}

PATTERNS=(
  'NGC_API_KEY'
  'nvcr\.io'
  'ngc\.nvidia\.com'
  'ngc\-cli'
)

FOUND=0
TOTAL_FILES=$(rg -uu --files | wc -l | tr -d ' ')
info "Scanning $TOTAL_FILES files for NGC dependencies..."

for p in "${PATTERNS[@]}"; do
  MATCHES=$(scan "$p")
  if [[ -n "$MATCHES" ]]; then
    if [[ $VERBOSE -eq 1 ]]; then
      err "Pattern '$p' found in:\n$MATCHES"
    else
      err "Pattern '$p' found (use -v for details)"
    fi
    FOUND=1
  else
    ok "Pattern '$p' not found"
  fi
done

if [[ $FOUND -eq 0 ]]; then
  ok "No NGC dependencies detected. Repository is NGC-independent."
  exit 0
else
  err "NGC dependencies detected. Please remove or replace these references."
  exit 1
fi

