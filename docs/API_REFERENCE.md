# API Reference and Configuration (Enhanced)

---

Last Updated: 2025-10-03
Owner: API Team
Review Cadence: Bi-weekly

---

<!-- TOC -->

- [Overview](#overview)
- [Environment Variables Reference](#environment-variables-reference)
  - [Core Configuration](#core-configuration)
  - [NVIDIA API Configuration](#nvidia-api-configuration)
  - [Model Selection](#model-selection)
  - [PubMed Integration](#pubmed-integration)
  - [Pharmaceutical Features](#pharmaceutical-features)
  - [Vector Database](#vector-database)
  - [Rate Limiting & Performance](#rate-limiting--performance)
  - [Monitoring & Cost Tracking](#monitoring--cost-tracking)
- [Python API Examples](#python-api-examples)
  - [Basic Embedding Client](#basic-embedding-client)
  - [RAG Agent Usage](#rag-agent-usage)
  - [PubMed Integration](#pubmed-integration-examples)
- [cURL API Examples](#curl-api-examples)
  - [Embedding Request](#embedding-request)
  - [Reranking Request](#reranking-request)
- [Edge Cases & Error Handling](#edge-cases--error-handling)
  - [Missing API Key](#edge-case-1-missing-api-key)
  - [Invalid API Key](#edge-case-2-invalid-api-key)
  - [Rate Limiting (HTTP 429)](#edge-case-3-rate-limiting-http-429)
  - [Model Unavailability](#edge-case-4-model-unavailability)
  - [Embedding Dimension Mismatch](#edge-case-5-embedding-dimension-mismatch)
  - [PubMed Rate Limiting](#edge-case-6-pubmed-rate-limiting)
  - [Cache Expiration](#edge-case-7-cache-expiration)
  - [Free Tier Exhaustion](#edge-case-8-free-tier-exhaustion)
- [CLI Command Reference](#cli-command-reference)
- [API Response Formats](#api-response-formats)
- [Rate Limits & Quotas](#rate-limits--quotas)
- [Troubleshooting Quick Reference](#troubleshooting-quick-reference)
- [Cross-References](#cross-references)

<!-- /TOC -->

## Overview

This comprehensive API reference provides detailed configuration options, code examples, and edge case handling for the NVIDIA NeMoRetriever RAG Template. The system is built with a cloud-first architecture using NVIDIA Build platform with NGC deprecation immunity.

**Key Features:**
- NGC-independent architecture (immune to March 2026 deprecation)
- Free tier: 10,000 requests/month
- OpenAI SDK compatibility
- Comprehensive pharmaceutical domain support
- Self-hosted fallback options

## Environment Variables Reference

### Core Configuration

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `NVIDIA_API_KEY` | string | - | NVIDIA Build API key (format: `nvapi-...`) | ✅ Yes |
| `APP_ENV` | string | `development` | Application environment (`development` \| `production`) | No |
| `DOCS_FOLDER` | string | `Data/Docs` | Directory containing documents to index | No |
| `VECTOR_DB_PATH` | string | `./vector_db` | Path to vector database storage | No |
| `CHUNK_SIZE` | integer | `1000` | Size of document chunks in characters | No |
| `CHUNK_OVERLAP` | integer | `200` | Overlap between chunks in characters | No |

**Example:**
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
export APP_ENV="production"
export DOCS_FOLDER="/path/to/pharmaceutical/docs"
```

### NVIDIA API Configuration

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `NVIDIA_BUILD_BASE_URL` | string | `https://integrate.api.nvidia.com/v1` | NVIDIA Build API endpoint | No |
| `EMBEDDING_BASE_URL` | string | (uses NVIDIA_BUILD_BASE_URL) | Override for embedding endpoint | No |
| `NEMO_RERANKING_ENDPOINT` | string | - | Self-hosted reranking endpoint | No |
| `NEMO_EXTRACTION_ENDPOINT` | string | - | Self-hosted extraction endpoint | No |

**Cloud-First Architecture:**
```bash
# Default: NVIDIA Build (cloud, NGC-independent)
NVIDIA_BUILD_BASE_URL=https://integrate.api.nvidia.com/v1

# Optional: Self-hosted fallback
NEMO_EMBEDDING_ENDPOINT=http://localhost:8000/v1
NEMO_RERANKING_ENDPOINT=http://localhost:8001/v1
```

### Model Selection

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `EMBEDDING_MODEL` | string | `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` | Primary embedding model | No |
| `EMBEDDING_FALLBACK_MODEL` | string | `nvidia/nv-embed-v1` | Fallback if primary unavailable | No |
| `RERANK_MODEL` | string | `llama-3_2-nemoretriever-500m-rerank-v2` | Reranking model | No |
| `LLM_MODEL_NAME` | string | `meta/llama-3.1-8b-instruct` | LLM for generation | No |

**Model Dimensions:**
- `llama-3.2-nemoretriever-1b-vlm-embed-v1`: **1024 dimensions**
- `nvidia/nv-embed-v1`: **4096 dimensions**

**Critical:** Switching models requires reindexing due to dimension differences.

### PubMed Integration

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `PUBMED_EMAIL` | string | - | Contact email (NCBI identification) | Recommended |
| `PUBMED_EUTILS_API_KEY` | string | - | NCBI API key for increased rate limits | No |
| `PUBMED_CACHE_DIR` | string | `./pubmed_cache` | PubMed response cache directory | No |
| `NCBI_CACHE_TTL_HOURS` | integer | `24` | Cache TTL (NCBI requires ≥24 hours) | No |
| `DEFAULT_MAX_ITEMS` | integer | `30` | Default max results per query | No |
| `ENABLE_DEDUPLICATION` | boolean | `true` | Deduplicate by DOI/PMID | No |

**Rate Limits:**
- Without API key: **3 requests/second**
- With API key: **10 requests/second**

### Pharmaceutical Features

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `PHARMACEUTICAL_RESEARCH_MODE` | boolean | `false` | Enable pharmaceutical features | No |
| `PHARMA_DOMAIN_OVERLAY` | boolean | `true` | Enable drug interaction detection | No |
| `ENABLE_MEDICAL_GUARDRAILS` | boolean | `true` | Enable safety guardrails | No |
| `NEMO_PHARMACEUTICAL_ANALYSIS` | boolean | `true` | Extract pharmaceutical metadata | No |
| `ENABLE_PHARMA_QUERY_ENHANCEMENT` | boolean | `true` | Expand queries with pharma terms | No |
| `PHARMA_MAX_TERMS` | integer | `8` | Max enhancement terms | No |

### Vector Database

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `VECTOR_DB_PER_MODEL` | boolean | `false` | Separate indexes per model | No |
| `FAISS_INDEX_TYPE` | string | `Flat` | FAISS index type | No |
| `ENABLE_VECTOR_DB_STATS` | boolean | `true` | Track vector DB statistics | No |

**VECTOR_DB_PER_MODEL Explanation:**
When `true`, creates separate vector stores for each embedding model to prevent dimension mismatches:
```
vector_db/
├── llama-3.2-nemoretriever-1b-vlm-embed-v1/  # 1024-dim
└── nv-embed-v1/                                # 4096-dim
```

### Rate Limiting & Performance

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `ENABLE_RATE_LIMITING` | boolean | `true` | Enable rate limiting | No |
| `MAX_REQUESTS_PER_SECOND` | float | `3.0` | Max requests/second | No |
| `DAILY_REQUEST_LIMIT` | integer | `10000` | Daily request cap (free tier) | No |
| `EMBEDDING_BATCH_SIZE` | integer | `10` | Batch size for embeddings | No |
| `ENABLE_ADVANCED_CACHING` | boolean | `true` | Enable query caching | No |

### Monitoring & Cost Tracking

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `NVIDIA_CREDITS_MONITORING` | boolean | `true` | Track NVIDIA Build usage | No |
| `PHARMA_BUDGET_LIMIT_USD` | float | - | Monthly budget limit | No |
| `ENABLE_DEBUG_LOGGING` | boolean | `false` | Enable debug logs | No |

## Python API Examples

### Basic Embedding Client

**Example 1: Initialize and Embed Documents**

```python
from src.nvidia_embeddings import NVIDIAEmbeddings

# Initialize with API key
embeddings = NVIDIAEmbeddings(
    api_key="nvapi-your-key-here",  # Or use NVIDIA_API_KEY env var
    embedding_model_name="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    batch_size=10,
    max_retries=3
)

# Embed multiple documents
documents = [
    "Warfarin is an anticoagulant medication.",
    "Aspirin has antiplatelet effects.",
    "Drug interactions can be serious."
]

# Returns: List[List[float]] with shape (3, 1024)
doc_embeddings = embeddings.embed_documents(documents)
print(f"Embedded {len(doc_embeddings)} documents")
print(f"Embedding dimension: {len(doc_embeddings[0])}")
# Expected output:
# Embedded 3 documents
# Embedding dimension: 1024

# Embed a query
query = "What are drug interactions?"
query_embedding = embeddings.embed_query(query)
print(f"Query embedding dimension: {len(query_embedding)}")
# Expected output:
# Query embedding dimension: 1024
```

**Example 2: Error Handling and Fallback**

```python
from src.nvidia_embeddings import NVIDIAEmbeddings
import logging

# Enable logging to see fallback behavior
logging.basicConfig(level=logging.INFO)

try:
    # This will attempt llama-3.2, fallback to nv-embed-v1 if unavailable
    embeddings = NVIDIAEmbeddings(
        embedding_model_name="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        fallback_model_name="nvidia/nv-embed-v1",
        probe_on_init=True  # Test model availability on startup
    )

    result = embeddings.embed_query("test query")
    print(f"Model used: {embeddings.model_name}")
    print(f"Selection reason: {embeddings.model_selection_reason}")

except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Runtime error: {e}")
```

### RAG Agent Usage

**Example 3: Complete RAG Workflow**

```python
from src.enhanced_rag_agent import EnhancedRAGAgent
from src.document_loader import DocumentLoader
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase

# 1. Load documents
loader = DocumentLoader(docs_folder="Data/Docs")
documents = loader.load_documents()
print(f"Loaded {len(documents)} documents")

# 2. Create embeddings
embeddings = NVIDIAEmbeddings()

# 3. Build vector database
vector_db = VectorDatabase(
    embeddings=embeddings,
    persist_directory="./vector_db"
)
vector_db.add_documents(documents)
vector_db.save()

# 4. Initialize RAG agent
agent = EnhancedRAGAgent(
    enable_guardrails=True,
    pharmaceutical_mode=True
)

# 5. Query with pharmaceutical context
response = agent.ask(
    "What are the drug interactions between warfarin and aspirin?",
    species_filter="human",  # Filter for human studies
    top_k=5
)

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)} documents")
print(f"Disclaimer: {response.disclaimer}")
```

### PubMed Integration Examples

**Example 4: Basic PubMed Scraping**

```python
from src.pubmed_scraper import PubMedScraper

# Initialize with caching
scraper = PubMedScraper(
    email="researcher@example.com",  # Required by NCBI
    api_key="your-ncbi-api-key",     # Optional, increases rate limits
    cache_dir="./pubmed_cache"
)

# Search PubMed
results = scraper.search(
    query="drug interactions warfarin aspirin",
    max_results=30
)

for article in results:
    print(f"PMID: {article['pmid']}")
    print(f"Title: {article['title']}")
    print(f"Year: {article['year']}")
    print(f"---")

# Metadata is cached in .pubmed.json sidecar files
```

**Example 5: PubMed with Ranking**

```python
# Enable study ranking for clinical relevance
results = scraper.search(
    query="pharmacokinetics study phase 3",
    max_results=50,
    rank=True,  # Enable study quality ranking
    enable_study_ranking=True
)

# Results are ordered by quality score
for article in results[:5]:  # Top 5 results
    print(f"Score: {article.get('quality_score', 'N/A')}")
    print(f"Title: {article['title']}")
```

## cURL API Examples

### Embedding Request

**Basic Embedding Request:**

```bash
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Authorization: Bearer nvapi-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Text to embed"],
    "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    "input_type": "query",
    "truncate": "END"
  }'
```

**Expected Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ..., 0.789]
    }
  ],
  "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### Reranking Request

**Rerank Retrieved Documents:**

```bash
curl -X POST https://integrate.api.nvidia.com/v1/ranking \
  -H "Authorization: Bearer nvapi-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3_2-nemoretriever-500m-rerank-v2",
    "query": {"text": "drug interactions warfarin"},
    "passages": [
      {"text": "Warfarin interacts with aspirin..."},
      {"text": "Aspirin is an NSAID medication..."}
    ]
  }'
```

**Expected Response:**
```json
{
  "rankings": [
    {
      "index": 0,
      "logit": 8.5,
      "score": 0.95
    },
    {
      "index": 1,
      "logit": 3.2,
      "score": 0.61
    }
  ]
}
```

## Edge Cases & Error Handling

### Edge Case 1: Missing API Key

**Problem:** NVIDIA_API_KEY environment variable not set

**Python Error:**
```python
ValueError: NVIDIA API key is required. Set NVIDIA_API_KEY environment variable.
```

**Code Location:** [src/nvidia_embeddings.py:112](../src/nvidia_embeddings.py#L112)

**Solution:**
```bash
# Set environment variable
export NVIDIA_API_KEY="nvapi-your-key-here"

# Or pass directly to constructor
embeddings = NVIDIAEmbeddings(api_key="nvapi-your-key-here")
```

**cURL Equivalent:**
```bash
# HTTP 401 Unauthorized
curl -X POST https://integrate.api.nvidia.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["test"], "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"}'

# Response: {"error": {"message": "Unauthorized", "type": "invalid_request_error"}}
```

### Edge Case 2: Invalid API Key

**Problem:** API key is invalid or expired

**HTTP Response:** `401 Unauthorized` or `403 Forbidden`

**Code Location:** [src/nvidia_embeddings.py:428-430](../src/nvidia_embeddings.py#L428)

**Error Handling:**
```python
# Automatic retry with exponential backoff
try:
    embeddings = NVIDIAEmbeddings()
    result = embeddings.embed_query("test")
except requests.exceptions.HTTPError as e:
    if e.response.status_code in [401, 403]:
        print("Invalid API key. Verify at https://build.nvidia.com")
```

**Solution:**
1. Verify API key at https://build.nvidia.com
2. Regenerate if expired
3. Check key format: must start with `nvapi-`

### Edge Case 3: Rate Limiting (HTTP 429)

**Problem:** Exceeded rate limits

**HTTP Response:** `429 Too Many Requests`

**Code Location:** [src/nvidia_embeddings.py:432-434](../src/nvidia_embeddings.py#L432)

**Automatic Handling:**
```python
# Built-in exponential backoff with jitter
# Retries: wait 1s, 2s, 4s, 8s...
embeddings = NVIDIAEmbeddings(
    max_retries=5,
    retry_delay=1.0
)
```

**Manual Rate Limiting:**
```bash
# Enable rate limiting in .env
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_SECOND=3.0
DAILY_REQUEST_LIMIT=10000
```

**Solution:**
1. Enable `ENABLE_RATE_LIMITING=true`
2. Reduce `EMBEDDING_BATCH_SIZE`
3. Implement request queuing
4. Monitor usage with credit tracker

### Edge Case 4: Model Unavailability

**Problem:** Requested model not available on NVIDIA Build

**Error Indicators:**
- `"unknown model"`
- `"model not found"`
- `"not available"`

**Code Location:** [src/nvidia_embeddings.py:337-400](../src/nvidia_embeddings.py#L337)

**Automatic Fallback:**
```python
# Automatically falls back to nv-embed-v1
embeddings = NVIDIAEmbeddings(
    embedding_model_name="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    fallback_model_name="nvidia/nv-embed-v1"
)

# Check which model is being used
print(f"Active model: {embeddings.model_name}")
print(f"Reason: {embeddings.model_selection_reason}")
```

**Configuration:**
```bash
# Configure fallback in .env
EMBEDDING_MODEL=nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
EMBEDDING_FALLBACK_MODEL=nvidia/nv-embed-v1
```

### Edge Case 5: Embedding Dimension Mismatch

**Problem:** Switching models with different dimensions causes vector DB errors

**Models:**
- `llama-3.2-nemoretriever`: **1024 dimensions**
- `nv-embed-v1`: **4096 dimensions**

**Error:**
```
RuntimeError: Vector dimension mismatch: expected 1024, got 4096
```

**Solution 1: Per-Model Vector Stores**
```bash
# Enable separate indexes per model
VECTOR_DB_PER_MODEL=true
```

**Solution 2: Reindex Corpus**
```python
# Clear existing index and reindex with new model
from src.vector_database import VectorDatabase

vector_db = VectorDatabase(persist_directory="./vector_db")
vector_db.clear()  # WARNING: Deletes all indexed documents

# Reindex with new model
embeddings = NVIDIAEmbeddings(embedding_model_name="nvidia/nv-embed-v1")
vector_db = VectorDatabase(embeddings=embeddings)
vector_db.add_documents(documents)
```

### Edge Case 6: PubMed Rate Limiting

**Problem:** NCBI E-utilities rate limits exceeded

**Rate Limits:**
- Without API key: **3 requests/second**
- With API key: **10 requests/second**

**Error:**
```
HTTP 429: Too Many Requests (NCBI E-utilities)
```

**Solution:**
```bash
# Set contact email (recommended by NCBI)
PUBMED_EMAIL=researcher@example.com

# Get API key for higher limits
PUBMED_EUTILS_API_KEY=your-ncbi-api-key

# Enable rate limiting
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_SECOND=2.5  # Stay under 3 req/s limit
```

### Edge Case 7: Cache Expiration

**Problem:** NCBI requires 24-hour minimum cache TTL

**NCBI Policy:** https://www.ncbi.nlm.nih.gov/books/NBK25497/

**Configuration:**
```bash
# Minimum 24 hours (NCBI compliance)
NCBI_CACHE_TTL_HOURS=24

# Grace period for stale cache
CACHE_ALLOW_STALE_WITHIN_GRACE=true
CACHE_GRACE_PERIOD_HOURS=2
```

**Handling Stale Cache:**
```python
from src.pubmed_scraper import PubMedScraper

scraper = PubMedScraper(
    cache_dir="./pubmed_cache",
    cache_ttl_hours=24,
    allow_stale_grace=True
)

# Uses cached data if within grace period
# Refreshes if cache is expired beyond grace
results = scraper.search("drug interactions")
```

### Edge Case 8: Free Tier Exhaustion

**Problem:** Exceeded NVIDIA Build free tier (10,000 requests/month)

**Monitoring:**
```bash
# Enable credit tracking
NVIDIA_CREDITS_MONITORING=true

# Set budget alert
PHARMA_BUDGET_LIMIT_USD=10.00
```

**Check Usage:**
```python
from src.monitoring.credit_tracker import CreditTracker

tracker = CreditTracker()
usage = tracker.get_usage_summary()

print(f"Requests used: {usage['total_requests']}")
print(f"Remaining: {usage['remaining_requests']}")
print(f"Reset date: {usage['reset_date']}")
```

**Optimization Strategies:**
1. **Enable Batch Processing:**
   ```bash
   EMBEDDING_BATCH_SIZE=20  # Increase batch size
   ```

2. **Enable Caching:**
   ```bash
   ENABLE_ADVANCED_CACHING=true
   CACHE_QUERY_RESULTS=true
   ```

3. **Reduce Query Frequency:**
   ```bash
   DAILY_REQUEST_LIMIT=300  # ~10k/month ÷ 30 days
   ```

4. **Switch to Self-Hosted:**
   ```bash
   # Use local NIMs (no API costs)
   NEMO_EMBEDDING_ENDPOINT=http://localhost:8000/v1
   ```

## CLI Command Reference

**Test API Connectivity:**
```bash
python scripts/nvidia_build_api_test.py
```

**Run Pharmaceutical Benchmarks:**
```bash
# All categories
python scripts/run_pharmaceutical_benchmarks.py

# Specific category
python scripts/run_pharmaceutical_benchmarks.py --category drug_interactions

# With concurrency
python scripts/run_pharmaceutical_benchmarks.py --concurrency 4

# Simulation mode (no API calls)
python scripts/run_pharmaceutical_benchmarks.py --simulate
```

**Validate Environment:**
```bash
python scripts/validate_env.py
```

**Launch CLI Interface:**
```bash
python main.py --mode cli
```

**Launch Web Interface:**
```bash
streamlit run streamlit_app.py
```

**Monitor Performance:**
```bash
python scripts/performance_monitor.py
```

## API Response Formats

**Embedding Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ..., 0.789]  // 1024 or 4096 floats
    }
  ],
  "model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

**Reranking Response:**
```json
{
  "rankings": [
    {
      "index": 0,
      "logit": 8.5,
      "score": 0.95
    }
  ],
  "model": "llama-3_2-nemoretriever-500m-rerank-v2",
  "usage": {
    "query_tokens": 5,
    "passage_tokens": 100,
    "total_tokens": 105
  }
}
```

**RAG Response (with Pharmaceutical Metadata):**
```python
{
    "answer": "Warfarin and aspirin can interact...",
    "sources": [
        {
            "content": "Document text...",
            "metadata": {
                "source": "pubmed_12345678",
                "title": "Study title",
                "year": 2023,
                "drug_names": ["warfarin", "aspirin"],
                "species": "human",
                "study_type": "clinical_trial"
            }
        }
    ],
    "disclaimer": "This information is for research purposes only. Consult healthcare professionals for medical advice.",
    "confidence_score": 0.87
}
```

## Rate Limits & Quotas

### NVIDIA Build Platform

| Tier | Requests/Month | Cost |
|------|----------------|------|
| Free | 10,000 | $0 |
| Paid | Custom | Variable |

**Daily Breakdown (Free Tier):**
- ~333 requests/day
- ~14 requests/hour
- Monitor with: `NVIDIA_CREDITS_MONITORING=true`

### PubMed E-utilities

| Configuration | Requests/Second |
|---------------|-----------------|
| No API key | 3 |
| With API key | 10 |

**Get API Key:** https://www.ncbi.nlm.nih.gov/account/

### Apify (PubMed Scraping)

**Cost:** $0.25 per 1000 compute units

**Budget Tracking:**
```bash
MONTHLY_BUDGET_LIMIT=19.99
```

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ValueError: NVIDIA API key is required` | Missing API key | Set `NVIDIA_API_KEY` |
| HTTP 401/403 | Invalid API key | Verify at build.nvidia.com |
| HTTP 429 | Rate limiting | Enable `ENABLE_RATE_LIMITING=true` |
| `unknown model` error | Model unavailable | Check fallback configuration |
| Vector dimension mismatch | Model switch | Enable `VECTOR_DB_PER_MODEL=true` |
| Slow PubMed queries | No API key | Set `PUBMED_EUTILS_API_KEY` |
| Cache errors | Expired cache | Check `NCBI_CACHE_TTL_HOURS` |
| Free tier exhausted | Over 10k requests/month | Enable caching or upgrade |

**For detailed troubleshooting:** [docs/TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

## Cross-References

### Architecture & Design
- [ADR-0001: Adopt NVIDIA NeMo Retriever](adr/0001-use-nemo-retriever.md) - Decision rationale
- [Architecture Documentation](ARCHITECTURE.md) - System architecture
- [NGC Deprecation Immunity](NGC_DEPRECATION_IMMUNITY.md) - NGC independence

### Integration Guides
- [API Integration Guide](API_INTEGRATION_GUIDE.md) - Advanced patterns
- [Examples](EXAMPLES.md) - Code examples
- [Features](FEATURES.md) - Feature documentation

### Operations
- [Free Tier Maximization](FREE_TIER_MAXIMIZATION.md) - Cost optimization
- [Cheapest Deployment](CHEAPEST_DEPLOYMENT.md) - Budget deployment
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Detailed diagnostics

### Pharmaceutical Domain
- [Pharmaceutical Best Practices](PHARMACEUTICAL_BEST_PRACTICES.md) - Domain guidelines
- [Benchmarks](BENCHMARKS.md) - Performance benchmarks

---

**Last Verified:** 2025-10-03
**API Version:** NVIDIA Build v1
**Free Tier Status:** Active (10,000 requests/month)
