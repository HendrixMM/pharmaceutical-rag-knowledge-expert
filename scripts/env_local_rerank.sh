#!/usr/bin/env bash
set -euo pipefail

echo "NEMO_RERANKING_ENDPOINT=http://localhost:8502/v1/rerank" > .env.local
echo "Wrote .env.local with NEMO_RERANKING_ENDPOINT"
