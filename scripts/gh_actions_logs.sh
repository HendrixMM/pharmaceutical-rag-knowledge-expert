#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

usage(){ echo "Usage: REPO=owner/repo $0 <run-id>"; }
if [ $# -lt 1 ]; then usage; exit 2; fi
run_id="$1"

repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    usage; exit 2
  fi
fi

run_gh run view "$run_id" --repo "$repo" --log
