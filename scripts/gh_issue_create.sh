#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

usage() {
  cat <<EOF
Usage: REPO=owner/repo $0 -t "Title" -b "Body" [-l labels] [-a assignees]
  -t  Issue title (required)
  -b  Issue body (required)
  -l  Comma-separated labels (optional)
  -a  Comma-separated assignees (optional)
EOF
}

title=""; body=""; labels=""; assignees="";
while getopts ":t:b:l:a:h" opt; do
  case $opt in
    t) title="$OPTARG";;
    b) body="$OPTARG";;
    l) labels="$OPTARG";;
    a) assignees="$OPTARG";;
    h) usage; exit 0;;
    *) usage; exit 2;;
  esac
done

repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    echo "Error: REPO not set and not in a git repo" >&2
    usage
    exit 2
  fi
fi

if [ -z "$title" ] || [ -z "$body" ]; then
  echo "Error: -t and -b are required" >&2
  usage
  exit 2
fi

args=(issue create --repo "$repo" --title "$title" --body "$body")
if [ -n "$labels" ]; then args+=(--label "$labels"); fi
if [ -n "$assignees" ]; then args+=(--assignee "$assignees"); fi

run_gh "${args[@]}"
