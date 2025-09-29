Troubleshooting Guide — NVIDIA Build + NeMo Retriever (Grouped by Phase)

Phase 0–1: Provisioning & Configuration Validation
- 401 Unauthorized
  - Cause: API key lacks access to the model/service.
  - Fix: In the NVIDIA Build console, enable the Embedding (nvidia/nv-embedqa-e5-v5) and Reranking (meta/llama-3_2-nemoretriever-500m-rerank-v2) NIMs for your key. Regenerate the key if needed.
  - Verify:
    curl -H "Authorization: Bearer $NVIDIA_API_KEY" https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings/health

- 404 Not Found
  - Cause: Wrong endpoint path or tenant‑specific URL.
  - Fix: Confirm your tenant endpoints. If different, set env overrides: NEMO_EMBEDDING_ENDPOINT, NEMO_RERANKING_ENDPOINT.
  - Verify:
    curl -H "Authorization: Bearer $NVIDIA_API_KEY" $NEMO_EMBEDDING_ENDPOINT/health

- 422 Unprocessable Entity
  - Cause: Payload shape mismatch or model ID mismatch.
  - Fix: Ensure JSON keys match the endpoint schema and model name is correct.
  - Verify (embedding example):
    curl -X POST $NEMO_EMBEDDING_ENDPOINT -H 'Authorization: Bearer $NVIDIA_API_KEY' -H 'Content-Type: application/json' -d '{"input":["test"]}'

Phase 2: PubMed Fetch
- Empty or malformed payload
  - Cause: E‑utilities schema changes or network issues.
  - Fix: Ensure PUBMED_EUTILS_API_KEY and PUBMED_EMAIL are set; check network.
  - Verify: Rerun with a simple query ("metformin pharmacokinetics"); inspect logs for “payload malformed” or “no usable texts”.

Phase 3: Health & Runtime
- High latency warnings (⚠️)
  - Cause: Transient load or regional distance.
  - Fix: Increase --health-latency-ms or retry; keep batch sizes small.

Phase 4: Credits
- Low credits warning
  - Cause: Free‑tier nearing quota.
  - Fix: Monitor in Build console; reduce batch size/top‑k; request more credits.

Useful Checks
- Endpoints summary and models are printed at run start — verify they match your tenant and desired models.
- Use --skip-health for ad‑hoc overlay demos; restore strict runs when ready.

Security Notes
- Never commit real keys; keep them in .env or Colab cell variables.
- Masked keys are printed in logs by default; do not log full env.

