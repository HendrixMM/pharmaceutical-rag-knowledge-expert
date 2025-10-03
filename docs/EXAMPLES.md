---
Last Updated: 2025-10-03
Owner: Developer Experience Team
Review Cadence: Weekly
---

# Examples

<!-- TOC -->

- [Quick Start Examples](#quick-start-examples)
- [Basic Usage](#basic-usage)
- [Pharmaceutical Examples](#pharmaceutical-examples)
- [PubMed Integration](#pubmed-integration)
- [Advanced RAG Patterns](#advanced-rag-patterns)
- [MCP Integration](#mcp-integration)
- [Monitoring & Analytics](#monitoring--analytics)
- [Production Patterns](#production-patterns)
- [Testing Examples](#testing-examples)
- [Complete Workflows](#complete-workflows)
- [Cross-References](#cross-references)
<!-- /TOC -->

A curated set of runnable examples that cover quick start, pharmaceutical use cases, PubMed workflows, advanced RAG patterns, and MCP integration.

## Quick Start Examples

- CLI: minimal usage to index and query documents.
- Web interface: launch Streamlit and ask first questions.
- Programmatic: create a basic RAG pipeline in a few lines.

## Basic Usage

- Load documents, build embeddings, vector search, rerank, and generate answers.

## Pharmaceutical Examples

- Drug interaction queries, clinical study filtering, species-specific search, and safety alerts.

## PubMed Integration

- Scraping with sidecar files, metadata extraction, deduplication, rank/preserve ordering variations.

## Advanced RAG Patterns

- Custom embeddings, reranking strategies, batch processing, caching.

## MCP Integration

See `examples/usage_example.py` for 8 examples covering:

- Basic client usage and prompt generation
- Troubleshooting patterns
- Enhanced agent and migration workflows
- Custom configuration and performance monitoring

## Monitoring & Analytics

- Performance tracking, cost monitoring, and pharma-specific metrics.

## Production Patterns

- Error handling, retry logic, health checks, graceful degradation.

## Testing Examples

- Unit and integration patterns, pharmaceutical benchmarks, async tests and markers.

## Complete Workflows

- End-to-end flows combining ingestion, search, pharma overlays, and generation.

## Cross-References

- Configuration details: [docs/API_REFERENCE.md](API_REFERENCE.md)
- Feature explanations: [docs/FEATURES.md](FEATURES.md)
- Development and testing: [docs/DEVELOPMENT.md](DEVELOPMENT.md)
- Advanced integration: [docs/API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md)
