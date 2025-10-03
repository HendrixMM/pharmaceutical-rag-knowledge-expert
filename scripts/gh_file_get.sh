#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/gh_wrap.sh"

usage(){ echo "Usage: REPO=owner/repo $0 -p <path> [-r <ref>]"; }
ref=""; fpath="";
while getopts ":p:r:h" opt; do
  case $opt in
    p) fpath="$OPTARG";;
    r) ref="$OPTARG";;
    h) usage; exit 0;;
    *) usage; exit 2;;
  esac
done

if [ -z "$fpath" ]; then usage; exit 2; fi

repo="${REPO:-}"
if [ -z "$repo" ]; then
  if ! repo=$(detect_repo); then
    usage; exit 2
  fi
fi

owner=$(echo "$repo" | cut -d'/' -f1)
name=$(echo "$repo" | cut -d'/' -f2)

path="/repos/${owner}/${name}/contents/${fpath}"
if [ -n "$ref" ]; then
  run_gh api -H "Accept: application/vnd.github.raw" "$path" -f ref="$ref"
else
  run_gh api -H "Accept: application/vnd.github.raw" "$path"
fi
