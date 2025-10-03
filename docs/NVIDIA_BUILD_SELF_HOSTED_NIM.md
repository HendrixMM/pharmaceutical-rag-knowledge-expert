---
Last Updated: 2025-10-03
Owner: Infrastructure Team
Review Cadence: Monthly
---

# NVIDIA Build Self‑Hosted NIM Deployment (Docker + Google Colab GPU)

Overview

- Deploy self‑hosted NeMo Retriever NIM services for Embedding and Reranking using Docker locally or on Google Colab GPU. Integrate with this repo by pointing the client to your local endpoints via environment variables.

Models

- Embedding: nv-embedqa-e5-v5 (primary for pharmaceutical Q&A)
- Reranking: llama-3_2-nemoretriever-500m-rerank-v2 (latest reranker)

Prerequisites

- NVIDIA Build API key with permissions to pull/run these NIM images
- Docker installed (local); Colab account with GPU runtime (for demo)
- Optional: kubectl for Kubernetes deployments

Common Setup

1. Create a secret env file for Docker
   nvidia_build_secret.env
   NVIDIA_BUILD_API_KEY=your_api_key_here

2. Authenticate and pull images
   nvidia-build login --api-key $NVIDIA_BUILD_API_KEY
   nvidia-build pull nvidia/nv-embedqa-e5-v5:latest
   nvidia-build pull nvidia/llama-3_2-nemoretriever-500m-rerank-v2:latest

Local Docker Deployment
Embedding service
docker run -d \
 --name nv-embed \
 --env-file nvidia_build_secret.env \
 -e EMBED_MODEL=nv-embedqa-e5-v5 \
 -p 8501:8501 \
 nvidia/nv-embedqa-e5-v5:latest

Health
curl -H "Authorization: Bearer $NVIDIA_BUILD_API_KEY" \
 http://localhost:8501/health

Usage
curl -X POST http://localhost:8501/v1/embed \
 -H "Content-Type: application/json" \
 -d '{"text":["example chunk"]}'

Reranking service
docker run -d \
 --name llama-rerank \
 --env-file nvidia_build_secret.env \
 -e RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2 \
 -p 8502:8502 \
 nvidia/llama-3_2-nemoretriever-500m-rerank-v2:latest

Health
curl -H "Authorization: Bearer $NVIDIA_BUILD_API_KEY" \
 http://localhost:8502/health

Usage
curl -X POST http://localhost:8502/v1/rerank \
 -H "Content-Type: application/json" \
 -d '{"query":"drug interactions","candidates":["text1","text2","text3"]}'

Google Colab GPU (experimental)

1. Runtime → Change runtime type → GPU
2. Install Docker + NVIDIA Build CLI
   !apt-get update && apt-get install -y docker.io
   !pip install nvidia-build-cli
3. Authenticate + pull
   !echo "NVIDIA_BUILD_API_KEY=your_api_key_here" > secret.env
   !nvidia-build login --api-key your_api_key_here
   !nvidia-build pull nvidia/nv-embedqa-e5-v5:latest
   !nvidia-build pull nvidia/llama-3_2-nemoretriever-500m-rerank-v2:latest
4. Run containers (ports exposed locally inside Colab VM)
   !docker run -d --name nv-embed \
    --env-file secret.env -e EMBED_MODEL=nv-embedqa-e5-v5 \
    -p 8501:8501 nvidia/nv-embedqa-e5-v5:latest
   !docker run -d --name llama-rerank \
    --env-file secret.env -e RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2 \
    -p 8502:8502 nvidia/llama-3_2-nemoretriever-500m-rerank-v2:latest
5. Test
   !curl -H "Authorization: Bearer $NVIDIA_BUILD_API_KEY" http://localhost:8501/health
   !curl -X POST http://localhost:8501/v1/embed -H "Content-Type: application/json" -d '{"text":["test"]}'

Note: Colab often restricts Docker; local Docker or GPU VMs (Runpod/AWS/GCP) are more reliable for self‑hosting.

App Integration (this repo)

- Point the client to your self‑hosted endpoints by setting the following in .env:
  NEMO_EMBEDDING_ENDPOINT=http://localhost:8501/v1/embed
  NEMO_RERANKING_ENDPOINT=http://localhost:8502/v1/rerank

  # Optional if you later host extraction too

  NEMO_EXTRACTION_ENDPOINT=http://localhost:8503/v1/extract

- Optional model overrides for test harness:
  EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
  RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2

Troubleshooting

- 401 Unauthorized: verify your NVIDIA Build API key in the container env; ensure headers are set when calling /health or APIs.
- 404 Not Found: confirm route paths (/v1/embed, /v1/rerank) and exposed ports (-p mapping).
- Docker permissions: run with sudo as needed; ensure Docker daemon is running.
- Colab limitations: Docker may not be available in some Colab sessions; use local Docker or a GPU VM for reliability.

Security

- Never commit real API keys. Use env files and secrets.
- Limit exposed ports; prefer tunneling for demos (ngrok/Cloudflared) and IP whitelisting when possible.
