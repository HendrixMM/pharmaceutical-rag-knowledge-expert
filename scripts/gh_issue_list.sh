#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    echo "Usage: REPO=owner/repo $0 [state]" >&2
    exit 2
  fi
fi
state="${1:-open}"

run_gh issue list --repo "$repo" --state "$state"
