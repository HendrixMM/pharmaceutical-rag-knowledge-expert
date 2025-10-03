---
Last Updated: 2025-10-03
Owner: API Team
Review Cadence: Bi-weekly
---

# API Reference and Configuration

<!-- TOC -->

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [Core Configuration](#core-configuration)
- [Pharmaceutical Configuration](#pharmaceutical-configuration)
- [Medical Safety Features](#medical-safety-features)
- [Advanced Configuration](#advanced-configuration)
- [PubMed Integration](#pubmed-integration)
- [Programmatic API](#programmatic-api)
- [API Response Formats](#api-response-formats)
- [Rate Limits & Quotas](#rate-limits--quotas)
- [Cross-References](#cross-references)
<!-- /TOC -->

This document centralizes configuration and programmatic API usage for the RAG Template for NVIDIA NeMoRetriever. Content is consolidated from the original README (configuration, flags, PubMed workflow, and pharma options) and aligned with advanced integration patterns.

## Overview

- Environment variables configure runtime behavior (models, endpoints, pharma features, caching, vector DB, PubMed).
- Programmatic APIs expose key classes and helpers for building RAG workflows.
- NVIDIA Build (cloud-first) is the default; self-hosted endpoints are supported as overrides.

## Environment Variables

The following categories summarize environment configuration. See examples and defaults in `.env.example`.

- Core configuration

  - `NVIDIA_API_KEY`: Required. NVIDIA Build API key.
  - `APP_ENV`: `development` | `production` (enforces strict behaviors in production).
  - `DOCS_FOLDER`, `VECTOR_DB_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`.

- Models and endpoints

  - `EMBEDDING_MODEL`, `RERANK_MODEL`.
  - `NEMO_EMBEDDING_ENDPOINT`, `NEMO_RERANKING_ENDPOINT`, `NEMO_EXTRACTION_ENDPOINT` (self-hosted overrides).

- Rerank behavior

  - `ENABLE_CLOUD_FIRST_RERANK` (ordering), `ENABLE_NVB_RERANK` (legacy compatibility).
  - Retry/backoff tuning: `RERANK_RETRY_BACKOFF_BASE`, `RERANK_RETRY_MAX_ATTEMPTS`, `RERANK_RETRY_JITTER`.

- PubMed integration

  - `PUBMED_EUTILS_API_KEY` (optional), `PUBMED_EMAIL` (recommended).
  - `PUBMED_CACHE_DIR`, ranking/dedup flags.

- Vector database

  - `VECTOR_DB_PER_MODEL=true|false` (per-model index isolation). Dimension checks enforce safety.

- Pharmaceutical configuration
  - `PHARMACEUTICAL_RESEARCH_MODE`, pharmaceutical feature flags, budgeting/cost tracking toggles.
  - Compliance, disclaimers, and QA toggles.

## Core Configuration

Focus on NVIDIA Build defaults with optional self-hosted fallbacks via `NEMO_*_ENDPOINT` overrides. Production mode restricts non-NVIDIA fallbacks for extraction (strict parsing).

## Pharmaceutical Configuration

Enable domain features for drug queries, clinical filtering, safety overlays, species preference, and budgeting. These flags propagate through clients and workflows.

## Medical Safety Features

Safety-first outputs with disclaimers and validation. In regulated contexts, enable compliance flags and medical terminology checks.

## Advanced Configuration

- Batch processing, caching, and memoization strategies to optimize cost and latency.
- Backoff/jitter and retry limits to improve resilience.

## PubMed Integration

- E-utilities integration with caching and sidecar metadata (`.pubmed.json`).
- CLI helpers support scraping, deduplication, and ranking.

## Programmatic API

High-level patterns and key components used across examples and integrations.

- RAG Agent

  - `RAGAgent` and helpers for retrieval, filtering, and answer generation.

- PubMed Scraper

  - `PubMedScraper` and CLI entry points for article fetching and sidecar generation.

- Configuration classes
  - Read from env and construct runtime configuration with sensible defaults.

## API Response Formats

- Embeddings: arrays of float vectors with metadata.
- Rerank results: items with relevance scores and source metadata.
- Document objects: text, source, and enriched pharma metadata (when enabled).

## Rate Limits & Quotas

- NVIDIA Build free tier: track requests; enable budgeting flags to monitor consumption.
- PubMed: respect E-utilities rate limits; set contact email.

## Cross-References

- Advanced integration patterns: [docs/API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md)
- Pharmaceutical guidance: [docs/PHARMACEUTICAL_BEST_PRACTICES.md](PHARMACEUTICAL_BEST_PRACTICES.md)
- Usage examples: [docs/EXAMPLES.md](EXAMPLES.md)
- Troubleshooting: [docs/TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
