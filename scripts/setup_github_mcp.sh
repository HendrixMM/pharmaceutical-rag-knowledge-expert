#!/usr/bin/env bash
set -euo pipefail

# Setup GitHub MCP server for Claude Code CLI using project scope.
# Does not store your token in Git. Requires GITHUB_PAT env var or .env with GITHUB_PAT.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if ! command -v claude >/dev/null 2>&1; then
  echo "Error: 'claude' CLI not found. Install Claude Code CLI first." >&2
  exit 1
fi

# Resolve token without sourcing .env (which may contain unquoted values)
TOKEN="${GITHUB_PAT:-}"
if [ -z "$TOKEN" ] && [ -f .env ]; then
  TOKEN=$(grep -E "^GITHUB_PAT=" .env | tail -n1 | sed -e 's/^GITHUB_PAT=//' -e 's/^"//' -e 's/"$//')
fi
if [ -z "$TOKEN" ]; then
  echo "Error: GITHUB_PAT not set in environment or .env."
  echo "- Create a GitHub Personal Access Token (PAT) with minimal scopes"
  echo "- Add 'GITHUB_PAT=your_token' to .env (never commit real tokens)"
  exit 2
fi

echo "Adding GitHub MCP server at project scope…"
claude mcp remove -s project github >/dev/null 2>&1 || true
claude mcp add -s project -t http \
  github \
  https://api.githubcopilot.com/mcp/ \
  -H "Authorization: Bearer ${TOKEN}"

echo "Verifying configuration…"
claude mcp get github || { echo "Failed to configure GitHub MCP server."; exit 3; }

echo "✅ GitHub MCP server configured for this project (.mcp.json created)."
echo "   Note: .mcp.json is git-ignored to avoid committing secrets."
