#!/bin/bash
# ADR Creation Helper Script
# Usage: ./scripts/new-adr.sh "your-decision-title"

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if title provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 \"your-decision-title\"${NC}"
    echo ""
    echo "Example:"
    echo "  $0 \"adopt-cloud-first-strategy\""
    echo "  $0 \"implement-pharmaceutical-guardrails\""
    exit 1
fi

# Get the title and sanitize it
TITLE="$1"
SANITIZED_TITLE=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g')

# Determine next ADR number
ADR_DIR="docs/adr"
if [ ! -d "$ADR_DIR" ]; then
    echo -e "${YELLOW}Error: $ADR_DIR directory not found${NC}"
    echo "Are you in the project root directory?"
    exit 1
fi

# Find the highest existing ADR number
LAST_NUM=$(ls "$ADR_DIR" | grep -E '^[0-9]{4}-' | sort -r | head -1 | awk -F'-' '{print $1}' | sed 's/^0*//')

if [ -z "$LAST_NUM" ]; then
    # No existing ADRs, start with 0001
    NEXT_NUM=1
else
    NEXT_NUM=$((LAST_NUM + 1))
fi

# Format with zero-padding
PADDED_NUM=$(printf "%04d" $NEXT_NUM)

# Create filename
FILENAME="${ADR_DIR}/${PADDED_NUM}-${SANITIZED_TITLE}.md"

# Check if file already exists
if [ -f "$FILENAME" ]; then
    echo -e "${YELLOW}Error: File already exists: $FILENAME${NC}"
    exit 1
fi

# Get current date
CURRENT_DATE=$(date +%Y-%m-%d)

# Copy template and update metadata
echo -e "${BLUE}Creating ADR ${PADDED_NUM}: ${TITLE}${NC}"

# Copy template
cp "${ADR_DIR}/template.md" "$FILENAME"

# Update the title and metadata
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/ADR-NNNN: \[Short Descriptive Title\]/ADR-${PADDED_NUM}: ${TITLE}/g" "$FILENAME"
    sed -i '' "s/Last Updated: YYYY-MM-DD/Last Updated: ${CURRENT_DATE}/g" "$FILENAME"
    sed -i '' "s/Date: YYYY-MM-DD/Date: ${CURRENT_DATE}/g" "$FILENAME"
    sed -i '' "s/Status: \[Proposed | Accepted | Deprecated | Superseded\]/Status: Proposed/g" "$FILENAME"
else
    # Linux
    sed -i "s/ADR-NNNN: \[Short Descriptive Title\]/ADR-${PADDED_NUM}: ${TITLE}/g" "$FILENAME"
    sed -i "s/Last Updated: YYYY-MM-DD/Last Updated: ${CURRENT_DATE}/g" "$FILENAME"
    sed -i "s/Date: YYYY-MM-DD/Date: ${CURRENT_DATE}/g" "$FILENAME"
    sed -i "s/Status: \[Proposed | Accepted | Deprecated | Superseded\]/Status: Proposed/g" "$FILENAME"
fi

echo -e "${GREEN}✅ Created: $FILENAME${NC}"
echo ""
echo "Next steps:"
echo "1. Edit the ADR and fill in all sections"
echo "2. Update docs/adr/README.md to add this ADR to the index"
echo "3. Link from related documentation"
echo "4. Commit with message: \"docs: add ADR-${PADDED_NUM} ${TITLE}\""
echo ""
echo -e "${BLUE}Opening ADR in default editor...${NC}"

# Try to open in editor
if command -v $EDITOR &> /dev/null; then
    $EDITOR "$FILENAME"
elif command -v code &> /dev/null; then
    code "$FILENAME"
elif command -v vim &> /dev/null; then
    vim "$FILENAME"
else
    echo -e "${YELLOW}No editor found. Please open manually: $FILENAME${NC}"
fi
