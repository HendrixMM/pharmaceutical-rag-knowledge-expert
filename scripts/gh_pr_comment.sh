#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

usage() {
  echo "Usage: REPO=owner/repo $0 <pr-number> <comment>"
}

if [ $# -lt 2 ]; then usage; exit 2; fi
pr_number="$1"; shift
comment="$*"

repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    usage; exit 2
  fi
fi

run_gh pr comment "$pr_number" --repo "$repo" --body "$comment"
