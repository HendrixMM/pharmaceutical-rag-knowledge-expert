set -euo pipefail

# identify_secrets_in_history.sh
# Scan git history for exposed secrets and generate a sanitized patterns file

usage() {
  cat <<'USAGE'
Usage: bash scripts/identify_secrets_in_history.sh [--output FILE] [--report FILE] [--dry-run] [--verbose] [--allow-working-env] [--bfg-map]

Scans git history for .env and secret-like assignments, extracting safe fingerprints
to scripts/sensitive-patterns.txt (or a custom output).

Options:
  --output FILE   Output patterns file (default: scripts/sensitive-patterns.txt)
  --report FILE   Save detailed findings to this path (e.g., backups/secret-scan-YYYYMMDD.txt)
  --dry-run       Show actions without writing files
  --verbose       Print progress details
  --allow-working-env  Proceed even if .env exists in working tree (warn)
  --bfg-map       Emit mapping format lines (fingerprint==>***REMOVED***)
USAGE
}

out_file="scripts/sensitive-patterns.txt"
report_file=""
dry_run=false
verbose=false
allow_working_env=false
bfg_map=false

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
    --allow-working-env)
      allow_working_env=true; shift;;
    --bfg-map)
      bfg_map=true; shift;;
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
  if $allow_working_env; then
    echo "⚠️  .env present in working tree (ignored due to --allow-working-env)" >&2
  else
    echo "❌ .env present in working tree (should not be committed). Aborting." >&2
    exit 2
  fi
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
  grep -E '^\+(NVIDIA_API_KEY=|PUBMED_EUTILS_API_KEY=|APIFY_TOKEN=|PUBMED_EMAIL=|OPENALEX_EMAIL=)' | sed 's/^+//')

nv_keys=$(echo "$candidate_lines" | grep -E '^NVIDIA_API_KEY=' | sed 's/NVIDIA_API_KEY=//' | cut -c1-10 | sort -u || true)
pb_keys=$(echo "$candidate_lines" | grep -E '^PUBMED_EUTILS_API_KEY=' | sed 's/PUBMED_EUTILS_API_KEY=//' | cut -c1-10 | sort -u || true)
ap_keys=$(echo "$candidate_lines" | grep -E '^APIFY_TOKEN=' | sed 's/APIFY_TOKEN=//' | cut -c1-10 | sort -u || true)
emails=$(echo "$candidate_lines" | grep -E '^(PUBMED_EMAIL=|OPENALEX_EMAIL=)' | sed 's/^[A-Z_]*=//' | sort -u || true)

# Filter out known placeholders and examples to avoid no-op replacements
filter_placeholders() {
  awk 'length($0)>5' | \
  grep -E -v '^(your_|test|dummy|placeholder)' | \
  grep -E -v 'example\.com' | \
  grep -E -v ':'
}

nv_keys=$(echo "$nv_keys" | filter_placeholders || true)
pb_keys=$(echo "$pb_keys" | filter_placeholders || true)
ap_keys=$(echo "$ap_keys" | filter_placeholders || true)
emails=$(echo "$emails" | filter_placeholders || true)

{
  echo "# Sensitive Patterns for BFG Repo-Cleaner"
  echo "# Generated: $(ts)"
  echo "# Only include concrete leaked fingerprints; no generic patterns."
  echo "# Do not include full secret values; use safe prefixes/fingerprints."
  echo ""
  emit_line() {
    local val="$1"
    if $bfg_map; then
      printf '%s==>***REMOVED***\n' "$val"
    else
      printf '%s\n' "$val"
    fi
  }
  if [[ -n "$nv_keys" ]]; then
    echo "# Discovered NVIDIA key fingerprints (truncated)"
    while read -r v; do [[ -n "$v" ]] && emit_line "$v"; done <<< "$nv_keys"
  fi
  if [[ -n "$pb_keys" ]]; then
    echo "# Discovered PubMed key fingerprints (truncated)"
    while read -r v; do [[ -n "$v" ]] && emit_line "$v"; done <<< "$pb_keys"
  fi
  if [[ -n "$ap_keys" ]]; then
    echo "# Discovered Apify token fingerprints (truncated)"
    while read -r v; do [[ -n "$v" ]] && emit_line "$v"; done <<< "$ap_keys"
  fi
  if [[ -n "$emails" ]]; then
    echo "# Discovered emails from .env history"
    while read -r v; do [[ -n "$v" ]] && emit_line "$v"; done <<< "$emails"
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
