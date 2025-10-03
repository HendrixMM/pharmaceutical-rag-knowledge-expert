---
Last Updated: 2025-10-03
Owner: Engineering Team
Review Cadence: Bi-weekly
---

# Development Guide

<!-- TOC -->
- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Project Architecture](#project-architecture)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Makefile Targets](#makefile-targets)
- [CI/CD Pipeline](#cicd-pipeline)
- [Common Development Tasks](#common-development-tasks)
- [Troubleshooting Development](#troubleshooting-development)
- [Contributing](#contributing)
- [Cross-References](#cross-references)
<!-- /TOC -->

Comprehensive guide for setting up the development environment, running tests, maintaining quality, and contributing changes.

## Getting Started

- Prerequisites: Python, make, virtualenv.
- Clone repo, create venv, install deps: core/dev/medical/nemo.
- Copy `.env.example` to `.env` and configure keys.

## Environment Setup

- Virtual environment best practices and IDE tips.
- Install pre-commit hooks: `pre-commit install`.
- Manage environment variables via `.env` and OS env.

## Project Architecture

- Directory structure (src/, docs/, tests/, examples/, scripts/, guardrails/).
- Separation of concerns, modularity, backward compatibility.

## Code Quality Standards

- Formatting: Black, isort
- Linting: Flake8
- Types: MyPy
- Security: Bandit
- Dependencies: Safety, pip-audit
- Docs: Docstrings and ownership metadata for markdown files

## Testing

- Unit, integration, and pharma benchmarks.
- Run tests: `make test`, coverage: `make coverage`.
- Async tests and markers; pharma disclaimers testing recommendations.

## Makefile Targets

- Install: `install`, `install-dev`, `install-all`
- Quality: `quality`, `format`, `lint`, `security`, `check`
- Testing: `test`, `test-unit`, `test-integration`, `coverage`, `test-rerank`, `pharma-bench`
- Docs: `docs`, `docs-linkcheck`, `docs-validate`, `docs-all` (see Makefile)
- Local services: `start-rerank`, `start-embed`, `start-extraction`, `start-local-stack`
- Cleanup: `clean`, `down-all`

## CI/CD Pipeline

- GitHub Actions (quality gates, multi-version tests, coverage, security scanning).

## Common Development Tasks

- Adding features, updating tests, docs updates, running benchmarks, debugging tips.

## Troubleshooting Development

- Common setup issues, import errors, test failures, pre-commit problems, env var pitfalls.

## Contributing

- See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.
- Commit message conventions and review process.

## Cross-References

- API configuration: [docs/API_REFERENCE.md](API_REFERENCE.md)
- Examples: [docs/EXAMPLES.md](EXAMPLES.md)
- Runtime troubleshooting: [docs/TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- Security: [../SECURITY.md](../SECURITY.md)
- Make commands: `Makefile`
