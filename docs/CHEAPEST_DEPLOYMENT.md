---
Last Updated: 2025-10-03
Owner: Cost Optimization Team
Review Cadence: Monthly
---

# Cheapest Deployment Plan for NeMo Retriever Pipeline (NVIDIA Build + Colab)

Goal

- Run the NeMo Retriever pipeline end‑to‑end at zero infrastructure cost by using NVIDIA Build’s free developer tier for hosted NIM endpoints and Google Colab (free GPU) for orchestration. Operates in PubMed mode (no PDFs) and preserves the pharmaceutical overlay.

What you use for free

- NVIDIA Build (cloud‑hosted NIMs)
  - Extraction (NV‑Ingest), Embedding (nvidia/nv-embedqa-e5-v5), Reranking (meta/llama-3_2-nemoretriever-500m-rerank-v2)
  - Free developer credits; monitor usage in NVIDIA Build console
- Google Colab
  - Free GPU runtime for running the client and overlay logic (no model hosting)

Repository env

- Set these in .env locally or in Colab via os.environ/write .env:
  - NVIDIA_API_KEY=your_build_api_key
  - PHARMA_DOMAIN_OVERLAY=true
  - ENABLE_NEMO_EXTRACTION=true
  - NEMO_EXTRACTION_STRATEGY=nemo
  - NEMO_EXTRACTION_STRICT=true
  - APP_ENV=production
  - NVIDIA_BUILD_FREE_TIER=true (optional credit logging)
  - Optional model overrides:
    - EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
    - RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2
  - Optional endpoint overrides (if your tenant exposes different URLs):
    - NEMO_EMBEDDING_ENDPOINT=
    - NEMO_RERANKING_ENDPOINT=
    - NEMO_EXTRACTION_ENDPOINT=

Colab quickstart

1. Open a new Colab notebook → Runtime → Change runtime type → GPU
2. Minimal dependencies
   !pip install -q python-dotenv aiohttp aiofiles langchain-core langchain langchain-community langchain-nvidia-ai-endpoints
3. Write .env (or set via os.environ)
   %%bash
   cat > .env << 'ENV'
   NVIDIA_API_KEY=YOUR_BUILD_KEY
   PHARMA_DOMAIN_OVERLAY=true
   ENABLE_NEMO_EXTRACTION=true
   NEMO_EXTRACTION_STRATEGY=nemo
   NEMO_EXTRACTION_STRICT=true
   APP_ENV=production
   NVIDIA_BUILD_FREE_TIER=true
   EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
   RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2
   ENV
4. Clone repo and run PubMed mode
   !git clone YOUR_REPO_URL repo && cd repo
   !python scripts/nim_native_test.py --pubmed "metformin pharmacokinetics"

What the harness does

- Health gate: validates Embedding and Reranking NIMs (prints endpoint + latency)
- PubMed mode: fetches titles/abstracts, embeds and reranks via NIMs
- Overlay: applies pharmaceutical overlay locally (no credits)
- Summary: selected models, durations, approximate credits (if enabled)

Monitoring credits

- NVIDIA_BUILD_FREE_TIER=true enables approximate credit logging; authoritative numbers are in the NVIDIA Build console.
- You can also inspect response headers directly when calling endpoints (x‑nvidia‑credits\* headers when available).

Troubleshooting

- 401 Unauthorized: Key not provisioned for model/service; fix in NVIDIA Build console
- 404/422: Endpoint path or model name mismatch; verify tenant endpoints and model IDs
- PubMed rate limits: Set PUBMED_EUTILS_API_KEY and PUBMED_EMAIL in .env for higher limits

Why this is cheapest

- All heavy compute is NVIDIA managed (covered by free credits)
- Colab runs only orchestration + overlay logic
- No cloud VM, no Docker hosting, no serverless costs
