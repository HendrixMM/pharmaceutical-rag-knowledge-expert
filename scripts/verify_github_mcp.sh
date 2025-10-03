#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  echo "Error: 'claude' CLI not found." >&2
  exit 1
fi

echo "Listing MCP servers…"
claude mcp list || true

echo "Details for 'github' server…"
claude mcp get github || {
  echo "GitHub MCP server not found in current scope. Try: make mcp-github-add" >&2
  exit 2
}

echo "✅ Verification complete."

