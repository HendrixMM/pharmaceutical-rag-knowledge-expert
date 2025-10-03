#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

# Optional filters via env: WORKFLOW, STATUS
repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    echo "Usage: REPO=owner/repo $0" >&2
    exit 2
  fi
fi

args=(run list --repo "$repo")
if [ -n "${WORKFLOW:-}" ]; then args+=(--workflow "$WORKFLOW"); fi
if [ -n "${STATUS:-}" ]; then args+=(--status "$STATUS"); fi

run_gh "${args[@]}"
