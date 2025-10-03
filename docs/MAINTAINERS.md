---
Last Updated: 2025-10-03
Owner: Maintainers Team
Review Cadence: Quarterly
---

# Maintainers Guide

<!-- TOC -->
- [Ownership & Review Routing](#ownership--review-routing)
- [Review SLAs (Targets)](#review-slas-targets)
- [PR Acceptance Criteria](#pr-acceptance-criteria)
- [Release & Changelog](#release--changelog)
- [Issue Triage](#issue-triage)
- [Security & Compliance](#security--compliance)
- [Decision Records](#decision-records)
- [Contacts](#contacts)
<!-- /TOC -->

This guide outlines ownership, review expectations, and lightweight processes that keep velocity high without adding bottlenecks.

## Ownership & Review Routing

- Source of truth for file ownership: .github/CODEOWNERS
- Docs automation and CI: .github/workflows/
- Documentation standards and structure: docs/\_shared/toc.md, CONTRIBUTING.md
- Security: SECURITY.md

## Review SLAs (Targets)

- Documentation-only PRs: review within 2 business days
- Minor code/infra PRs: review within 2 business days
- Critical fixes (P0): acknowledge within 2 hours, merge as soon as checks pass

## PR Acceptance Criteria

- CI green (Docs Checks, unit tests if applicable)
- Scope minimal and well-described
- For docs PRs:
  - TOCs updated if headings changed (make docs)
  - Internal links validated (CI covers; optional local: make docs-linkcheck)
  - Metadata present for new docs
  - Appropriate labels applied (auto-labeler helps)

## Release & Changelog

- Maintain CHANGELOG.md (Keep a Changelog)
- For doc-only changes, log under Unreleased → Documentation
- Tag releases after significant milestones; keep commits small and traceable

## Issue Triage

- Label new issues within 24h (bug, feature, documentation, ci)
- Priorities: P0 (critical), P1 (high), P2 (normal)
- Link related PRs and close issues on merge

## Security & Compliance

- Follow SECURITY.md for vulnerability handling and reporting
- Pre-commit hooks are required locally (security and quality gates)
- Never commit secrets; use .env and GitHub Secrets

## Decision Records

- Architectural rationale: docs/ARCHITECTURE.md and docs/NGC_DEPRECATION_IMMUNITY.md
- Notable changes: CHANGELOG.md

## Contacts

- General: community@yourorg.com
- Security: security@yourorg.com
- Emergency: on-call channel (internal)
