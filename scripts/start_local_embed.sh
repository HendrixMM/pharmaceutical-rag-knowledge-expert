#!/usr/bin/env bash
set -euo pipefail

# Load env
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "WARNING: NVIDIA_API_KEY is not set. The embedder container may not start without credentials." >&2
fi

# Detect compose
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "ERROR: Neither 'docker compose' nor 'docker-compose' is available." >&2
  exit 1
fi

echo "Starting Embedder NIM via docker compose..."
"${COMPOSE_CMD[@]}" up -d embedder

echo "Waiting for embedder to become healthy on http://localhost:8501/health ..."
ATTEMPTS=60
for i in $(seq 1 "$ATTEMPTS"); do
  if curl -sf http://localhost:8501/health >/dev/null 2>&1; then
    echo "Embedder is healthy."
    echo "NEMO_EMBEDDING_ENDPOINT=http://localhost:8501/v1/embed" >> .env.local
    echo "Wrote .env.local with NEMO_EMBEDDING_ENDPOINT"
    exit 0
  fi
  sleep 2
done

echo "ERROR: Embedder did not become healthy in time. Recent container logs:" >&2
"${COMPOSE_CMD[@]}" logs --tail=100 embedder || true
exit 2

