#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

# Optional filters via env: STATE (open|fixed|dismissed), SEVERITY (low|medium|high|critical)
repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    echo "Usage: REPO=owner/repo $0" >&2
    exit 2
  fi
fi

owner=$(echo "$repo" | cut -d'/' -f1)
name=$(echo "$repo" | cut -d'/' -f2)

path="/repos/${owner}/${name}/dependabot/alerts"
params=("state=${STATE:-open}")
if [ -n "${SEVERITY:-}" ]; then params+=("severity=$SEVERITY"); fi

run_gh api -H "Accept: application/vnd.github+json" "$path" --paginate $(printf ' -f %s' "${params[@]}")
