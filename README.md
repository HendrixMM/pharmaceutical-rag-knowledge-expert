# 🏛️ RAG Template for NVIDIA NeMoRetriever

<!-- TOC -->
- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Quality & Testing](#quality--testing)
- [NGC Deprecation Immunity](#ngc-deprecation-immunity)
- [Support & Community](#support--community)
- [License & Academic Context](#license--academic-context)
<!-- /TOC -->

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Build-green)](https://build.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NGC Immune](https://img.shields.io/badge/NGC_Deprecation-IMMUNE-success?logo=nvidia)](docs/NGC_DEPRECATION_IMMUNITY.md)
[![Docs Checks](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/actions/workflows/docs.yml/badge.svg)](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/actions/workflows/docs.yml)
[![Docs External Links](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/actions/workflows/docs-nightly.yml/badge.svg)](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/actions/workflows/docs-nightly.yml)

Production-ready Retrieval-Augmented Generation (RAG) template powered by NVIDIA NeMo Retriever (NIM), with a pharmaceutical-focused feature set, modern architecture, and NGC-deprecation immunity.

## Overview

This template helps you build document Q&A systems with:

- A clean RAG pipeline (ingestion → embeddings → vector search → rerank → generation)
- Cloud‑first NVIDIA Build integration with self‑hosted fallback
- A Streamlit web UI and CLI for quick iteration
- Guardrails and domain features for pharmaceutical research

Who’s it for: engineers, researchers, and teams building reliable, domain-aware RAG apps.

## Key Features

- 🔎 Core RAG: ingestion, vector search, reranking, citations
- 🤖 NeMo 2.0 Hybrid: multi‑model intelligence, backward compatible
- 💊 Pharma overlay: drug interactions, clinical filtering, safety-first outputs
- 🧠 PubMed: scraping, metadata sidecars, deduplication
- 🧰 Developer experience: tests, quality gates, Makefile, examples
- 🛡️ NGC‑immune: cloud‑first strategy with self‑hosted fallback
- 💸 Cost‑aware: free‑tier optimization, batching, caching
- 📈 Monitoring: performance metrics and health checks

Complete details: see docs/FEATURES.md

## Quick Start

Prerequisites

- Python 3.8+
- NVIDIA API key

Install and run

1. Clone
   - `git clone https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever.git`
   - `cd RAG-Template-for-NVIDIA-nemoretriever`
2. Install
   - Base: `pip install -r requirements.txt`
   - Optional GPU/NeMo: `pip install -r requirements-nemo.txt`
   - Optional medical: `pip install -r requirements-medical.txt`
3. Configure
   - `cp .env.example .env`
   - Add `NVIDIA_API_KEY` to `.env`
4. Run
   - Web: `streamlit run streamlit_app.py`
   - CLI: `python main.py`

More details: QUICK_START.md and SETUP_GUIDE.md

## Architecture Overview

Hybrid cloud‑first RAG with NVIDIA Build (primary) and optional self‑hosted NIMs.

Placeholder: [Architecture diagram will be added. See docs/ARCHITECTURE.md]

Key decisions

- Cloud‑first for reliability and speed
- NGC deprecation immunity (no migration required)
- Modular design with backward compatibility

Details: docs/ARCHITECTURE.md

## Project Structure

```
RAG-Template-for-NVIDIA-nemoretriever/
├── src/         # Core code
├── docs/        # Documentation
├── tests/       # Test suite
├── examples/    # Usage examples
├── scripts/     # Utilities (quality, docs, benchmarks)
└── guardrails/  # Safety modules
```

See docs/DEVELOPMENT.md for a deeper breakdown.

## Documentation

- Getting Started: QUICK_START.md, SETUP_GUIDE.md
- Core Docs: docs/FEATURES.md, docs/API_REFERENCE.md, docs/EXAMPLES.md, docs/DEVELOPMENT.md, docs/DEPLOYMENT.md
- Security: docs/security/history-redaction.md, docs/security/key-rotation-tracker.md
- Specialized Guides: docs/API_INTEGRATION_GUIDE.md, docs/PHARMACEUTICAL_BEST_PRACTICES.md, docs/NEMO_PIPELINE_GUIDE.md,
  docs/NVIDIA_MODEL_ACCESS_GUIDE.md, docs/NVIDIA_BUILD_SELF_HOSTED_NIM.md, docs/FREE_TIER_MAXIMIZATION.md, docs/CHEAPEST_DEPLOYMENT.md
- Reference: docs/TROUBLESHOOTING_GUIDE.md, SECURITY.md, CONTRIBUTING.md, CHANGELOG.md
- Full map: docs/\_shared/toc.md

## Quality & Testing

- Quality gates: `make quality`
- Run tests: `make test` (coverage: `make coverage`)
- Pre‑commit: `pre-commit install`
  See docs/DEVELOPMENT.md for tools and standards.

## NGC Deprecation Immunity

This template is designed to be immune to the March 2026 NGC API deprecation:

- Cloud‑first (NVIDIA Build) and self‑hosted fallback
- No migration needed; already independent
- Audit script verifies no NGC dependencies

Learn more: docs/NGC_DEPRECATION_IMMUNITY.md

## Support & Community

- Issues: GitHub Issues
- Security: SECURITY.md
- Contributing: CONTRIBUTING.md

## License & Academic Context

- MIT License — see LICENSE
- Built for research and engineering workflows (Concordia University context)

— If this helps, please star the repo. Built with excellence. ✨
