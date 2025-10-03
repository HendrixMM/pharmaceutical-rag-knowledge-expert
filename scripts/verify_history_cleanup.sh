set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/verify_history_cleanup.sh [--verbose] [--report FILE] [--no-color]

Runs a suite of checks to confirm that git history has been cleaned of secrets.
USAGE
}

verbose=false
report_file="backups/verification-report-$(date +%Y%m%d-%H%M%S).txt"
use_color=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose) verbose=true; shift;;
    --report) report_file="$2"; shift 2;;
    --no-color) use_color=false; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"; NC="\033[0m"
if ! $use_color; then RED=""; GREEN=""; YELLOW=""; NC=""; fi

pass() { echo -e "$1 ${GREEN}✅ PASS${NC}"; }
fail() { echo -e "$1 ${RED}❌ FAIL${NC}"; }
warn() { echo -e "$1 ${YELLOW}⚠️  WARN${NC}"; }

ts() { date '+%Y-%m-%d %H:%M:%S'; }
sha() { git rev-parse --short HEAD 2>/dev/null || echo "unknown"; }

mkdir -p backups
tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT

echo "════════════════════════════════════════════════════════════" | tee -a "$tmp"
echo "Git History Cleanup Verification Report" | tee -a "$tmp"
echo "════════════════════════════════════════════════════════════" | tee -a "$tmp"
echo "Timestamp: $(ts)" | tee -a "$tmp"
echo "Commit SHA: $(sha)" | tee -a "$tmp"
echo "" | tee -a "$tmp"

overall=0

# Check 1: .env in history
if git log --all --full-history -- .env >/dev/null 2>&1; then
  fail "Check 1: .env files in history...................." | tee -a "$tmp"
  overall=1
else
  pass "Check 1: .env files in history...................." | tee -a "$tmp"
fi

# Check 2: nvapi- pattern
if git log -p --all | grep -i 'nvapi-' >/dev/null 2>&1; then
  fail "Check 2: NVIDIA API key patterns.................." | tee -a "$tmp"
  overall=1
else
  pass "Check 2: NVIDIA API key patterns.................." | tee -a "$tmp"
fi

# Check 3: NVIDIA_API_KEY=
if git log -p --all | grep -i 'NVIDIA_API_KEY=' >/dev/null 2>&1; then
  fail "Check 3: NVIDIA_API_KEY assignments..............." | tee -a "$tmp"
  overall=1
else
  pass "Check 3: NVIDIA_API_KEY assignments..............." | tee -a "$tmp"
fi

# Check 4: PUBMED_EUTILS_API_KEY=
if git log -p --all | grep -i 'PUBMED_EUTILS_API_KEY=' >/dev/null 2>&1; then
  fail "Check 4: PubMed API key patterns.................." | tee -a "$tmp"
  overall=1
else
  pass "Check 4: PubMed API key patterns.................." | tee -a "$tmp"
fi

# Check 5: APIFY_TOKEN=
if git log -p --all | grep -i 'APIFY_TOKEN=' >/dev/null 2>&1; then
  fail "Check 5: Apify token patterns....................." | tee -a "$tmp"
  overall=1
else
  pass "Check 5: Apify token patterns....................." | tee -a "$tmp"
fi

# Check 6: detect-secrets baseline scan
if command -v detect-secrets >/dev/null 2>&1; then
  if detect-secrets scan --baseline .secrets.baseline >/dev/null 2>&1; then
    pass "Check 6: detect-secrets scan......................" | tee -a "$tmp"
  else
    fail "Check 6: detect-secrets scan......................" | tee -a "$tmp"
    overall=1
  fi
else
  warn "Check 6: detect-secrets not installed............." | tee -a "$tmp"
fi

# Check 7: .env in .gitignore
if grep -q '^\.env' .gitignore 2>/dev/null; then
  pass "Check 7: .env in .gitignore......................." | tee -a "$tmp"
else
  fail "Check 7: .env in .gitignore......................." | tee -a "$tmp"
  overall=1
fi

# Check 8: .env not in working tree
if [[ ! -f .env ]]; then
  pass "Check 8: .env in working tree....................." | tee -a "$tmp"
else
  warn "Check 8: .env present in working tree............." | tee -a "$tmp"
fi

echo "" | tee -a "$tmp"
echo "════════════════════════════════════════════════════════════" | tee -a "$tmp"
if [[ $overall -eq 0 ]]; then
  echo -e "Overall Status: ${GREEN}✅ VERIFICATION PASSED${NC}" | tee -a "$tmp"
else
  echo -e "Overall Status: ${RED}❌ VERIFICATION FAILED${NC}" | tee -a "$tmp"
fi
echo "════════════════════════════════════════════════════════════" | tee -a "$tmp"

cp "$tmp" "$report_file"
echo ""
echo "📄 Report saved to: $report_file"

exit "$overall"
