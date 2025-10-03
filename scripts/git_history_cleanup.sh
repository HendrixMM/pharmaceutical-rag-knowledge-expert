#!/usr/bin/env bash
set -euo pipefail

# git_history_cleanup.sh
# Rewrite git history to remove secrets using BFG Repo-Cleaner.

usage() {
  cat <<'USAGE'
Usage: bash scripts/git_history_cleanup.sh [--dry-run] [--skip-backup] [--auto-confirm] [--mirror-dir DIR]

Pre-flight checks, backups, BFG execution, and verification. Requires BFG installed.
USAGE
}

dry_run=false
skip_backup=false
auto_confirm=false
mirror_dir="repo-mirror.git"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift;;
    --skip-backup) skip_backup=true; shift;;
    --auto-confirm) auto_confirm=true; shift;;
    --mirror-dir) mirror_dir="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

fail() { echo "❌ $1" >&2; exit 1; }
info() { echo "$1"; }
ts() { date '+%Y-%m-%d %H:%M:%S'; }

command -v bfg >/dev/null 2>&1 || fail "BFG not installed. Install: brew install bfg or see https://rtyley.github.io/bfg-repo-cleaner/"
git rev-parse --git-dir >/dev/null 2>&1 || fail "Not a git repository"

[[ -f scripts/sensitive-patterns.txt ]] || fail "Missing scripts/sensitive-patterns.txt"

# Ensure clean working tree
if ! $dry_run; then
  if [[ -n "$(git status --porcelain)" ]]; then
    fail "Uncommitted changes present. Commit or stash before cleanup."
  fi
fi

if ! $skip_backup; then
  info "📦 Creating backups..."
  tag_name="backup/week2-start"
  git tag -f "$tag_name" -m "Backup before BFG cleanup $(ts)"
  mkdir -p backups
  pip freeze > "backups/pip-freeze-$(date +%Y%m%d-%H%M%S).txt" || true
  git branch "backup-$(date +%Y%m%d-%H%M%S)" >/dev/null 2>&1 || true
  echo "Backup tag: $tag_name" > "backups/cleanup-backup-manifest-$(date +%Y%m%d-%H%M%S).txt"
fi

info "🧪 Pre-flight summary"
echo "- Dry run: $dry_run"
echo "- Mirror dir: $mirror_dir"
echo "- Patterns file: scripts/sensitive-patterns.txt"

if ! $auto_confirm; then
  echo ""
  echo "This will REWRITE git history using BFG and require a FORCE-PUSH."
  read -rp "Type 'CLEANUP' to proceed: " resp
  [[ "$resp" == "CLEANUP" ]] || fail "User aborted."
fi

if $dry_run; then
  info "🔎 Dry run complete. Review settings above."
  exit 0
fi

repo_url=$(git config --get remote.origin.url || echo "")
[[ -n "$repo_url" ]] || fail "No remote origin URL configured"

rm -rf "$mirror_dir"
info "🔁 Cloning mirror: $repo_url -> $mirror_dir"
git clone --mirror "$repo_url" "$mirror_dir"

pushd "$mirror_dir" >/dev/null
info "🧹 Removing .env files with BFG..."
bfg --delete-files .env --no-blob-protection .

info "🧽 Replacing secret patterns with BFG..."
bfg --replace-text ../scripts/sensitive-patterns.txt .

info "🗑️  Expiring reflog and running aggressive GC..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive
popd >/dev/null

info "🔍 Verification (mirror)"
set +e
git --git-dir="$mirror_dir" log --all --full-history -- .env >/dev/null 2>&1
[[ $? -ne 0 ]] && env_ok=true || env_ok=false

git --git-dir="$mirror_dir" log -p --all | grep -i 'nvapi-' >/dev/null 2>&1; [[ $? -ne 0 ]] && nv_ok=true || nv_ok=false
git --git-dir="$mirror_dir" log -p --all | grep -i 'NVIDIA_API_KEY=' >/dev/null 2>&1; [[ $? -ne 0 ]] && nvk_ok=true || nvk_ok=false
set -e

if $env_ok && $nv_ok && $nvk_ok; then
  info "✅ Mirror verification passed (no .env or NVIDIA keys)."
else
  fail "Verification failed in mirror. Inspect $mirror_dir and try again."
fi

echo ""
info "🚀 Ready to force-push cleaned history"
echo "Run:"
echo "  (cd $mirror_dir && git push --force --all)"
echo "  (cd $mirror_dir && git push --force --tags)"
echo "Notify the team using the template in docs/security/history-redaction.md"

log_file="backups/cleanup-log-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "Cleanup completed at $(ts)"
  echo "Mirror directory: $mirror_dir"
} >> "$log_file"
info "📄 Log saved: $log_file"

exit 0

