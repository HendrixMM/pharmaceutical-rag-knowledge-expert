---
Last Updated: 2025-10-03
Owner: Documentation Team
Review Cadence: Monthly
Purpose: Canonical navigation structure for all project documentation
---

# Documentation Navigation (Canonical TOC)

<!-- TOC -->

- [Getting Started](#getting-started)
- [Core Documentation](#core-documentation)
- [Specialized Guides](#specialized-guides)
- [Development](#development)
- [Deployment](#deployment)
- [Architecture & Decisions](#architecture--decisions)
- [Reference](#reference)
<!-- /TOC -->

This is the canonical navigation blueprint for the repository. It defines the logical structure of all documentation and should be kept in sync as docs evolve.

## Getting Started

- [README.md](../../README.md) — Project overview, quick start, and links
- [QUICK_START.md](../technical-history/QUICK_START.md) — Step-by-step quick start
- [SETUP_GUIDE.md](../technical-history/SETUP_GUIDE.md) — Detailed environment and setup guide

## Core Documentation

- [docs/FEATURES.md](../FEATURES.md) — Full feature list and capabilities
- [docs/API_REFERENCE.md](../API_REFERENCE.md) — Environment variables, configuration, and API details
- [docs/EXAMPLES.md](../EXAMPLES.md) — Usage examples and runnable snippets

## Specialized Guides

- [docs/API_INTEGRATION_GUIDE.md](../API_INTEGRATION_GUIDE.md) — Advanced integration patterns and workflows
- [docs/PHARMACEUTICAL_BEST_PRACTICES.md](../PHARMACEUTICAL_BEST_PRACTICES.md) — Pharmaceutical domain guidance and safety
- [docs/NEMO_PIPELINE_GUIDE.md](../NEMO_PIPELINE_GUIDE.md) — NeMo Retriever extraction, embedding, reranking pipeline
- [docs/NVIDIA_MODEL_ACCESS_GUIDE.md](../NVIDIA_MODEL_ACCESS_GUIDE.md) — Model access tiers and endpoints
- [docs/NVIDIA_BUILD_SELF_HOSTED_NIM.md](../NVIDIA_BUILD_SELF_HOSTED_NIM.md) — Self-hosted NIM setup
- [docs/FREE_TIER_MAXIMIZATION.md](../FREE_TIER_MAXIMIZATION.md) — Cost optimization with NVIDIA free tier
- [docs/CHEAPEST_DEPLOYMENT.md](../CHEAPEST_DEPLOYMENT.md) — Zero-cost, minimal deployment blueprint

## Development

- [docs/DEVELOPMENT.md](../DEVELOPMENT.md) — Developer setup, testing, quality, and workflows
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — Contribution guidelines
- [docs/MAINTAINERS.md](../MAINTAINERS.md) — Ownership, SLAs, and review expectations

## Deployment

- [docs/DEPLOYMENT.md](../DEPLOYMENT.md) — Consolidated deployment guide (cloud-first and self-hosted)

## Architecture & Decisions

- [docs/ARCHITECTURE.md](../ARCHITECTURE.md) — System architecture overview and components
- [docs/NGC_DEPRECATION_IMMUNITY.md](../NGC_DEPRECATION_IMMUNITY.md) — NGC API deprecation immunity rationale

## Reference

- [docs/TROUBLESHOOTING_GUIDE.md](../TROUBLESHOOTING_GUIDE.md) — Troubleshooting and maintenance
- [docs/TROUBLESHOOTING_NVIDIA_BUILD.md](../TROUBLESHOOTING_NVIDIA_BUILD.md) — NVIDIA Build-specific troubleshooting
- [docs/security/history-redaction.md](../security/history-redaction.md) — Git history cleanup runbook
- [docs/security/key-rotation-tracker.md](../security/key-rotation-tracker.md) — API key rotation audit log
- [SECURITY.md](../../SECURITY.md) — Security policy and practices
- [CHANGELOG.md](../../CHANGELOG.md) — Release notes and changes
- [docs/\_shared/metadata_template.md](./metadata_template.md) — Template for required documentation metadata

---

Maintenance Notes

- Update this file when adding, renaming, or removing docs.
- Keep section ordering consistent to help users discover content quickly.
