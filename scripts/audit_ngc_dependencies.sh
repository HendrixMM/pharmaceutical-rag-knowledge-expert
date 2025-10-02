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
EXCLUDES=(backups \/venv\/ \/.git\/ \/.pytest_cache\/ \/__pycache__\/)
EXPR_EXCLUDE=$(printf "|%s" "${EXCLUDES[@]}")
EXPR_EXCLUDE="${EXPR_EXCLUDE:1}"

scan() {
  local pattern="$1"
  rg -n --hidden \
    --glob '!venv/*' \
    --glob '!*__pycache__/*' \
    --glob '!backups/*' \
    --glob '!.git/*' \
    --glob '!scripts/audit_ngc_dependencies.sh' \
    "$pattern" || true
}

# Check if a line contains NGC pattern only in a comment or safe context
is_comment_line() {
  local full_line="$1"

  # Extract just the content part (after filename:linenum:)
  local content=$(echo "$full_line" | sed 's/^[^:]*:[^:]*://')

  # Check if line is a comment (starts with # after optional whitespace)
  # Works for YAML, shell/env files, and other config formats
  if echo "$content" | grep -qE '^\s*#'; then
    return 0  # Is comment
  fi

  # Check for NGC immunity feature flags (safe, not dependencies)
  # These variables are ABOUT NGC deprecation immunity, not NGC dependencies
  if echo "$content" | grep -qE '^\s*(ENABLE_NGC_DEPRECATION_WARNINGS|ENABLE_MIGRATION_ASSISTANCE)\s*='; then
    return 0  # Safe NGC immunity feature flag
  fi

  # Check if entire line content is within a comment (pattern appears after #)
  # This handles inline comments and comment blocks
  local before_hash=$(echo "$content" | sed 's/#.*//')
  if ! echo "$before_hash" | grep -qE 'NGC_API_KEY|nvcr\.io|ngc\.nvidia\.com|ngc-cli'; then
    # Pattern only appears after # - it's in a comment
    return 0
  fi

  return 1  # Not a comment - real dependency
}

PATTERNS=(
  'NGC_API_KEY'
  'nvcr\.io'
  'ngc\.nvidia\.com'
  'ngc\-cli'
)

FOUND=0
COUNTS=()
TOTAL_FILES=$(rg -uu --files | wc -l | tr -d ' ')
info "Scanning $TOTAL_FILES files for NGC dependencies..."

# Allowlist: files where NGC mentions are expected and educational/documentary
# - Documentation files explain NGC deprecation and immunity
# - compose-mock.yaml is for testing purposes
#
# Note: docker-compose.yml and .env are NOT blanket-allowlisted.
# Instead, we use comment-aware filtering (see is_comment_line function) to distinguish:
#   - NGC mentions in comments (allowed, educational)
#   - NGC immunity feature flags (allowed, e.g., ENABLE_NGC_DEPRECATION_WARNINGS)
#   - NGC dependencies in actual values (not allowed, would break NGC independence)
ALLOWLIST_REGEX='^(docs/NGC_DEPRECATION_IMMUNITY\.md|docs/NVIDIA_MODEL_ACCESS_GUIDE\.md|docs/DEPLOYMENT_GUIDE\.md|compose-mock\.yaml):'

idx=0
for p in "${PATTERNS[@]}"; do
  # Get all matches excluding allowlisted files
  MATCHES=$(scan "$p" | grep -Ev "$ALLOWLIST_REGEX" || true)

  # Apply comment-aware filtering for docker-compose.yml, .env, and similar config files
  if [[ -n "$MATCHES" ]]; then
    FILTERED_MATCHES=""
    while IFS= read -r line; do
      # Check if this match is from a config file that needs comment filtering
      if echo "$line" | grep -qE '^(docker-compose\.yml|.*\.yaml|.*\.yml|\.env|.*\.env):'; then
        # Apply comment and safe-context filtering
        if ! is_comment_line "$line"; then
          # Not a comment or safe context - this is a real dependency
          FILTERED_MATCHES="${FILTERED_MATCHES}${line}"$'\n'
        fi
        # If it's a comment or safe context, skip it (don't add to FILTERED_MATCHES)
      else
        # Non-config file, include the match
        FILTERED_MATCHES="${FILTERED_MATCHES}${line}"$'\n'
      fi
    done <<< "$MATCHES"

    # Remove trailing newline and check if we have any real matches left
    MATCHES=$(echo -n "$FILTERED_MATCHES" | sed '/^$/d')
  fi

  if [[ -n "$MATCHES" ]]; then
    if [[ $VERBOSE -eq 1 ]]; then
      err "Pattern '$p' found in:\n$MATCHES"
    else
      err "Pattern '$p' found (use -v for details)"
    fi
    FOUND=1
    # Count matches (lines)
    COUNT=$(echo "$MATCHES" | sed '/^$/d' | wc -l | tr -d ' ')
    COUNTS+=("$COUNT")
  else
    ok "Pattern '$p' not found"
    COUNTS+=("0")
  fi
  idx=$((idx+1))
done

info "--- Audit Summary ---"
info "Total files scanned: $TOTAL_FILES"
for i in "${!PATTERNS[@]}"; do
  p=${PATTERNS[$i]}
  c=${COUNTS[$i]}
  if [[ "$c" != "0" ]]; then
    err "Flagged pattern '$p': $c match(es)"
  else
    ok "Pattern '$p': 0 matches"
  fi
done

if [[ $FOUND -eq 0 ]]; then
  ok "No NGC dependencies detected. Repository is NGC-independent."
  exit 0
else
  err "NGC dependencies detected. Please remove or replace these references."
  exit 1
fi
