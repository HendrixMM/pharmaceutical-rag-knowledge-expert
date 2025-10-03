#!/usr/bin/env bash
set -euo pipefail

# identify_secrets_in_history.sh
# Scan git history for exposed secrets and generate a sanitized patterns file

usage() {
  cat <<'USAGE'
Usage: bash scripts/identify_secrets_in_history.sh [--output FILE] [--report FILE] [--dry-run] [--verbose]

Scans git history for .env and secret-like assignments, extracting safe fingerprints
to scripts/sensitive-patterns.txt (or a custom output).

Options:
  --output FILE   Output patterns file (default: scripts/sensitive-patterns.txt)
  --report FILE   Save detailed findings to this path (e.g., backups/secret-scan-YYYYMMDD.txt)
  --dry-run       Show actions without writing files
  --verbose       Print progress details
USAGE
}

out_file="scripts/sensitive-patterns.txt"
report_file=""
dry_run=false
verbose=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      out_file="$2"; shift 2;;
    --report)
      report_file="$2"; shift 2;;
    --dry-run)
      dry_run=true; shift;;
    --verbose)
      verbose=true; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "❌ Not a git repository" >&2
  exit 2
fi

# Verify .env is ignored
if ! grep -q "^\.env" .gitignore 2>/dev/null; then
  echo "⚠️  .env not listed in .gitignore (please ensure it's ignored)" >&2
fi

if [[ -f .env ]]; then
  echo "❌ .env present in working tree (should not be committed). Aborting." >&2
  exit 2
fi

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { $verbose && echo "$1" || true; }

tmp_report="$(mktemp)"
trap 'rm -f "$tmp_report"' EXIT

echo "==== Secret History Scan ($(ts)) ====\n" >> "$tmp_report"

# Gather commits touching .env
env_commits=$(git log --all --full-history --pretty=format:%H -- .env || true)
count_env_commits=$(echo "$env_commits" | sed '/^$/d' | wc -l | tr -d ' ')
echo "Commits with .env: $count_env_commits" >> "$tmp_report"

# Extract candidate lines from history diffs
candidate_lines=$(git log -p --all -- .env 2>/dev/null | \
  grep -E "+(NVIDIA_API_KEY=|PUBMED_EUTILS_API_KEY=|APIFY_TOKEN=|PUBMED_EMAIL=|OPENALEX_EMAIL=)" | sed 's/^+//')

nv_keys=$(echo "$candidate_lines" | grep -E '^NVIDIA_API_KEY=' | sed 's/NVIDIA_API_KEY=//' | cut -c1-10 | sort -u || true)
pb_keys=$(echo "$candidate_lines" | grep -E '^PUBMED_EUTILS_API_KEY=' | sed 's/PUBMED_EUTILS_API_KEY=//' | cut -c1-10 | sort -u || true)
ap_keys=$(echo "$candidate_lines" | grep -E '^APIFY_TOKEN=' | sed 's/APIFY_TOKEN=//' | cut -c1-10 | sort -u || true)
emails=$(echo "$candidate_lines" | grep -E '^(PUBMED_EMAIL=|OPENALEX_EMAIL=)' | sed 's/^[A-Z_]*=//' | sort -u || true)

{
  echo "# Sensitive Patterns for BFG Repo-Cleaner"
  echo "# Generated: $(ts)"
  echo "# Do not include full secret values; only prefixes/fingerprints."
  echo ""
  echo "# NVIDIA API key prefixes"
  echo "nvapi-"
  echo "NVIDIA_API_KEY=nvapi-"
  echo ""
  echo "# PubMed E-utilities"
  echo "PUBMED_EUTILS_API_KEY="
  echo "PUBMED_EMAIL="
  echo ""
  echo "# Apify tokens"
  echo "APIFY_TOKEN="
  echo ""
  echo "# Generic tokens"
  echo "Bearer "
  echo "Authorization: Bearer "
  echo ""
  echo "# OpenAlex email"
  echo "OPENALEX_EMAIL="
  echo ""
  if [[ -n "$nv_keys" ]]; then
    echo "# Discovered NVIDIA key fingerprints"
    echo "$nv_keys" | sed 's/^/nvapi-/'
  fi
  if [[ -n "$pb_keys" ]]; then
    echo "# Discovered PubMed key fingerprints"
    echo "$pb_keys"
  fi
  if [[ -n "$ap_keys" ]]; then
    echo "# Discovered Apify token fingerprints"
    echo "$ap_keys"
  fi
  if [[ -n "$emails" ]]; then
    echo "# Discovered emails"
    echo "$emails"
  fi
} > "${out_file}.tmp"

if $dry_run; then
  echo "🔎 Dry run: patterns file preview at ${out_file}.tmp"
else
  mv "${out_file}.tmp" "$out_file"
  echo "✅ Patterns written to $out_file"
fi

{
  echo "\nFindings Summary:"
  echo "- Commits with .env: $count_env_commits"
  echo "- NVIDIA key fingerprints: $(echo "$nv_keys" | sed '/^$/d' | wc -l | tr -d ' ')"
  echo "- PubMed key fingerprints: $(echo "$pb_keys" | sed '/^$/d' | wc -l | tr -d ' ')"
  echo "- Apify token fingerprints: $(echo "$ap_keys" | sed '/^$/d' | wc -l | tr -d ' ')"
  echo "- Emails discovered: $(echo "$emails" | sed '/^$/d' | wc -l | tr -d ' ')"
} >> "$tmp_report"

if [[ -n "$report_file" ]]; then
  mkdir -p "$(dirname "$report_file")"
  cp "$tmp_report" "$report_file"
  echo "📄 Report saved to $report_file"
else
  cat "$tmp_report"
fi

# Exit 1 if secrets were found (to draw attention), else 0
total_found=$(( $(echo "$nv_keys" | sed '/^$/d' | wc -l) + $(echo "$pb_keys" | sed '/^$/d' | wc -l) + $(echo "$ap_keys" | sed '/^$/d' | wc -l) ))
if [[ $total_found -gt 0 || $count_env_commits -gt 0 ]]; then
  exit 1
fi
exit 0

