#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

usage(){
  cat <<EOF
Usage: REPO=owner/repo $0 -q "query" [-l language] [-f filename] [-p path] [--in <file|path|symbol>]
Examples:
  REPO=owner/repo $0 -q "nemo retriever" -l python
  REPO=owner/repo $0 -q "class VectorDatabase" --in file
EOF
}

query=""; language=""; filename=""; path=""; inqual="";
while [ $# -gt 0 ]; do
  case "$1" in
    -q) shift; query="${1:-}";;
    -l) shift; language="${1:-}";;
    -f) shift; filename="${1:-}";;
    -p) shift; path="${1:-}";;
    --in) shift; inqual="${1:-}";;
    -h|--help) usage; exit 0;;
  esac
  shift || true
done

if [ -z "$query" ]; then usage; exit 2; fi

repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    echo "Error: REPO not set and not in a git repo" >&2
    exit 2
  fi
fi

q="$query repo:$repo"
[ -n "$language" ] && q+=" language:$language"
[ -n "$filename" ] && q+=" filename:$filename"
[ -n "$path" ] && q+=" path:$path"
[ -n "$inqual" ] && q+=" in:$inqual"

run_gh api /search/code -f q="$q" -f per_page=20 -H "Accept: application/vnd.github.text-match+json"
