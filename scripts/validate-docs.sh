#!/bin/bash
# Documentation Validation Script
set -e
echo "📚 DOCUMENTATION VALIDATION SUITE"
echo "=================================="
date
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ERRORS=0
WARNINGS=0

echo "🔍 Check 1: Required files..."
for doc in docs/API_REFERENCE.md docs/EXAMPLES.md docs/BENCHMARKS.md docs/TROUBLESHOOTING_GUIDE.md mkdocs.yml; do
    [ -f "$doc" ] && echo -e "${GREEN}✅ $doc${NC}" || { echo -e "${RED}❌ $doc${NC}"; ((ERRORS++)); }
done

echo ""
echo "🔍 Check 2: Metadata headers..."
python3 - <<'PY'
import os, sys
count = 0
for root, dirs, files in os.walk('docs'):
    for f in files:
        if f.endswith('.md'):
            path = os.path.join(root, f)
            with open(path) as fp:
                content = fp.read()
                if 'Last Updated:' in content and 'Owner:' in content:
                    count += 1
print(f"✅ {count} files with complete metadata")
PY

echo ""
echo "🔍 Check 3: Internal links..."
python3 - <<'PY'
import re, os, sys
errors = 0
for root, dirs, files in os.walk('docs'):
    if 'technical-history' in root: continue
    for f in files:
        if not f.endswith('.md'): continue
        path = os.path.join(root, f)
        with open(path) as fp:
            for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', fp.read()):
                target = match.group(2)
                if target.startswith(('http:', 'https:', '#')): continue
                target = target.split('#')[0]
                if not target: continue
                if target.startswith('./'):
                    tpath = os.path.join(root, target[2:])
                elif target.startswith('../'):
                    tpath = os.path.normpath(os.path.join(root, target))
                else:
                    tpath = os.path.join(root, target)
                if not os.path.exists(tpath):
                    print(f"❌ Broken: {path} -> {target}")
                    errors += 1
if errors == 0:
    print("✅ All links valid")
else:
    print(f"❌ {errors} broken link(s)")
    sys.exit(1)
PY
[ $? -ne 0 ] && ((ERRORS++))

echo ""
echo "🔍 Check 4: Hardcoded secrets..."
if grep -rE "nvapi-[A-Za-z0-9_-]{40,}" docs/ 2>/dev/null | grep -v "nvapi-YOUR" | grep -q "nvapi-"; then
    echo -e "${RED}❌ Found potential API key${NC}"
    ((ERRORS++))
else
    echo -e "${GREEN}✅ No secrets detected${NC}"
fi

echo ""
echo "🔍 Check 5: Mermaid diagrams..."
COUNT=$(grep -r "\`\`\`mermaid" docs/ 2>/dev/null | wc -l | xargs)
[ "$COUNT" -gt 0 ] && echo -e "${GREEN}✅ Found $COUNT diagram(s)${NC}" || echo -e "${YELLOW}⚠️  No diagrams${NC}"

echo ""
echo "=================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED: $ERRORS error(s)${NC}"
    exit 1
fi
