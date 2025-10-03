#!/usr/bin/env bash
set -euo pipefail

# Common helpers for GitHub CLI wrappers.

mask() { sed -E 's/(Bearer |gho_|github_pat_)[A-Za-z0-9_\-]+/\1REDACTED/g'; }

get_token() {
  if [ -n "${GITHUB_PAT:-}" ]; then
    echo -n "$GITHUB_PAT"
    return 0
  fi
  if [ -f .env ]; then
    local t
    t=$(grep -E '^GITHUB_PAT=' .env | tail -n1 | sed -e 's/^GITHUB_PAT=//' -e 's/^"//' -e 's/"$//') || true
    if [ -n "$t" ]; then
      echo -n "$t"
      return 0
    fi
  fi
  return 1
}

need_gh() {
  if ! command -v gh >/dev/null 2>&1; then
    echo "Error: GitHub CLI 'gh' not found. Install via: brew install gh" >&2
    exit 127
  fi
}

detect_repo() {
  if [ -n "${REPO:-}" ]; then
    echo -n "$REPO"
    return 0
  fi
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    local remote
    remote=$(git config --get remote.origin.url || true)
    if [ -n "$remote" ]; then
      # Convert git/https URL to owner/repo robustly
      local sanitized
      sanitized=$(echo "$remote" | sed -E 's#^(git@[^:]+:|https://[^/]+/)##')
      sanitized=$(echo "$sanitized" | sed -E 's#\.git$##')
      echo "$sanitized"
      return 0
    fi
  fi
  echo ""; return 1
}

run_gh() {
  need_gh
  local token
  if ! token=$(get_token); then
    echo "Error: GITHUB_PAT not set in env or .env" >&2
    exit 2
  fi
  GH_TOKEN="$token" gh "$@"
}
