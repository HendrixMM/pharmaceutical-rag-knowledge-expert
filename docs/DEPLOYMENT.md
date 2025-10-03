---
Last Updated: 2025-10-03
Owner: DevOps Team
Review Cadence: Monthly
---

# Deployment Guide (Consolidated)

<!-- TOC -->
- [Overview and Philosophy](#overview-and-philosophy)
- [Quick Deployment Options](#quick-deployment-options)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Configuration Management](#configuration-management)
- [Cloud-First Deployment (NVIDIA Build)](#cloud-first-deployment-nvidia-build)
- [Self-Hosted Fallback Setup (Expanded)](#self-hosted-fallback-setup-expanded)
- [Local Services Management](#local-services-management)
- [Docker Deployment (Expanded)](#docker-deployment-expanded)
- [Cloud Platform Deployment (Expanded)](#cloud-platform-deployment-expanded)
- [Monitoring and Alerts](#monitoring-and-alerts)
- [Security Hardening](#security-hardening)
- [Performance Optimization](#performance-optimization)
- [Backup and Recovery](#backup-and-recovery)
- [Production Checklist and Deployment](#production-checklist-and-deployment)
- [Post-Deployment](#post-deployment)
- [Cross-References](#cross-references)
<!-- /TOC -->

Consolidates and expands the prior deployment guidance with quick options, cloud-first strategy, and self-hosted fallbacks.

## Overview and Philosophy

Cloud-first with NVIDIA Build for speed and reliability, with optional self-hosted NIMs as fallbacks or for air-gapped needs.

## Quick Deployment Options

- Local development: run Streamlit and CLI with a local `.env`.
- Docker: containerize the app and run locally.
- Cloud platforms: deploy to Streamlit Cloud, Heroku, or container services on AWS/GCP/Azure.

## Prerequisites

- NVIDIA API key, configured environment variables, and optional local service endpoints.

## Environment Setup

- Copy `.env.example` and set values; production environments should rely on secrets managers.

## Configuration Management

- Centralize config in environment variables; prefer cloud-first endpoints with strict fallbacks in production.

## Cloud-First Deployment (NVIDIA Build)

- Default mode uses NVIDIA Build endpoints for embeddings, reranking, and (optionally) extraction.
- Monitor credits and rate limits.

## Self-Hosted Fallback Setup (Expanded)

- Run local NIM services for reranker, embedder, and extraction.
- Configure `NEMO_*_ENDPOINT` env vars to point to local services.

## Local Services Management

- Start/stop services via Make:
  - `make start-rerank` / `make stop-rerank`
  - `make start-embed` / `make stop-embed`
  - `make start-extraction` / `make stop-extraction`
  - `make start-local-stack` / `make stop-local-stack`
  - `make start-local-all` / `make stop-local-all`
  - Cleanup: `make down-all`

## Docker Deployment (Expanded)

- Build and run the container; pass env via `.env` or platform secrets.

## Cloud Platform Deployment (Expanded)

- Guidance for Streamlit Cloud, Heroku, and container platforms (ECS, Cloud Run, AKS, etc.).

## Monitoring and Alerts

- Health checks, performance tracking, and pharmaceutical insights.

## Security Hardening

- Secrets management, HTTPS, minimal logs, safe defaults.

## Performance Optimization

- Batch processing, caching, vector store settings.

## Backup and Recovery

- Backup vector DB and configuration; document restore procedures.

## Production Checklist and Deployment

- Final checks for production promotion.

## Post-Deployment

- Ongoing monitoring, capacity planning, and incident readiness.

## Cross-References

- Development setup: [docs/DEVELOPMENT.md](DEVELOPMENT.md)
- Configuration: [docs/API_REFERENCE.md](API_REFERENCE.md)
- Troubleshooting: [docs/TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- Architecture rationale: [docs/NGC_DEPRECATION_IMMUNITY.md](NGC_DEPRECATION_IMMUNITY.md)
- Security: [../SECURITY.md](../SECURITY.md)
