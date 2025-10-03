#!/usr/bin/env bash
set -euo pipefail

# Load NVIDIA_API_KEY from .env if present
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "WARNING: NVIDIA_API_KEY is not set. The reranker container may not start without credentials." >&2
fi

# Detect docker compose command
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "ERROR: Neither 'docker compose' nor 'docker-compose' is available." >&2
  exit 1
fi

echo "Starting Reranker NIM via docker compose..."
"${COMPOSE_CMD[@]}" --profile rerank up -d reranker

echo "Waiting for reranker to become healthy on http://localhost:8502/health ..."
ATTEMPTS=60
for i in $(seq 1 "$ATTEMPTS"); do
  if curl -sf http://localhost:8502/health >/dev/null 2>&1; then
  echo "Reranker is healthy."
  # Persist local endpoint for convenience
  if [[ -x scripts/env_local_rerank.sh ]]; then
    ./scripts/env_local_rerank.sh || true
  else
    echo "NEMO_RERANKING_ENDPOINT=http://localhost:8502/v1/rerank" > .env.local
    echo "Wrote .env.local with NEMO_RERANKING_ENDPOINT"
  fi
  exit 0
fi
  sleep 2
done

echo "ERROR: Reranker did not become healthy in time. Recent container logs:" >&2
"${COMPOSE_CMD[@]}" logs --tail=100 reranker || true
exit 2
