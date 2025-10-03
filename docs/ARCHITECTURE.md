---
Last Updated: 2025-10-03
Owner: Architecture Team
Review Cadence: Monthly
---

# Architecture

<!-- TOC -->
- [Overview](#overview)
- [Diagram Placeholder](#diagram-placeholder)
- [Key Components](#key-components)
- [Design Principles](#design-principles)
- [Technology Stack](#technology-stack)
- [Data Flow](#data-flow)
- [Deployment Architecture](#deployment-architecture)
- [Future Enhancements](#future-enhancements)
- [Cross-References](#cross-references)
<!-- /TOC -->

High-level description of the RAG system architecture to be expanded with diagrams and detailed component interactions in Phase 2.1.

## Overview

The system implements a hybrid NeMo 2.0 RAG architecture with cloud-first NVIDIA Build endpoints and optional self-hosted NIM fallbacks. Pharmaceutical features overlay the core pipeline.

## Diagram Placeholder

[Architecture diagram to be added — see Phase 2.1 plan]

## Key Components

- RAG Pipeline: document loading, embedding, vector storage, retrieval, generation
- NVIDIA Integration: embedding, reranking, extraction services
- Pharmaceutical Layer: guardrails, safety validation, compliance
- Data Layer: vector database, caching, PubMed integration
- Monitoring & Analytics

## Design Principles

- Modularity and separation of concerns
- Backward compatibility for existing imports and features
- Cloud-first with NGC deprecation immunity

## Technology Stack

Python, LangChain, FAISS, Streamlit, NVIDIA NeMo/NIM, and related SDKs

## Data Flow

Ingestion → Embedding → Vector Search → Rerank → Generation → Guardrail Validation → Response

## Deployment Architecture

Default: NVIDIA Build (cloud-first). Fallback: self-hosted NIMs with equivalent APIs.

## Future Enhancements

- C4 diagrams (L1 system context), mermaid data flows
- Guardrail component tracing
- Deployment architecture illustrations

## Cross-References

- Immunity rationale: [docs/NGC_DEPRECATION_IMMUNITY.md](NGC_DEPRECATION_IMMUNITY.md)
- Feature mapping: [docs/FEATURES.md](FEATURES.md)
- Deployment: [docs/DEPLOYMENT.md](DEPLOYMENT.md)
- Integration patterns: [docs/API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md)
