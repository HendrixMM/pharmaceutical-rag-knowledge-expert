# Changelog

---

Last Updated: 2025-10-03
Owner: Release Management Team
Review Cadence: With each release

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- CRITICAL: Completed git history cleanup to remove exposed API keys
  - Scrubbed NVIDIA API keys (nvapi-*) from all commits
  - Scrubbed PubMed E-utilities API keys from all commits
  - Scrubbed Apify tokens from all commits
  - Removed all `.env` files from git history using BFG Repo-Cleaner
  - Force-pushed cleaned history on [DATE]
  - All exposed keys rotated immediately after cleanup
- Added `detect-secrets` baseline for automated secret prevention
- Enhanced pre-commit hooks with `detect-secrets` integration
- Created comprehensive security documentation:
  - `docs/security/history-redaction.md` - Complete cleanup process documentation
  - `scripts/git_history_cleanup.sh` - Automated cleanup script
  - `scripts/identify_secrets_in_history.sh` - Secret scanning script
  - `scripts/verify_history_cleanup.sh` - Verification script
- Established backup procedures:
  - Created `backup/week2-start` tag before cleanup
  - Exported dependency snapshot to `backups/pip-freeze-[date].txt`
  - Documented all audit artifacts in `docs/security/history-redaction.md`

### Changed

- BREAKING: Refactored README.md from 1559 lines to <250 lines for improved discoverability
- Reorganized documentation into specialized files:
  - Created `docs/API_REFERENCE.md` for comprehensive API and configuration reference
  - Created `docs/EXAMPLES.md` for usage examples and code snippets
  - Created `docs/FEATURES.md` for detailed feature descriptions
  - Created `docs/DEVELOPMENT.md` for development setup and guidelines
  - Consolidated `docs/DEPLOYMENT_GUIDE.md` into `docs/DEPLOYMENT.md` with enhanced content
- Added `docs/_shared/toc.md` as canonical documentation navigation hierarchy
- Added ownership metadata (Last Updated, Owner, Review Cadence) to all documentation files

### Added

- Documentation tooling and automation:
  - `scripts/docs_linkcheck.py` for validating documentation links
  - `scripts/docs_toc_generator.py` for automatic table of contents generation
  - `scripts/docs_metadata_validator.py` for metadata validation
  - Makefile targets: `docs`, `docs-linkcheck`, `docs-validate`, `docs-all`
  - Pre-commit hooks for documentation validation
- Created `docs/ARCHITECTURE.md` (basic structure, to be expanded in Phase 2.1)

### Documentation

- All documentation files now include ownership metadata for accountability
- Improved cross-referencing between documentation files
- Enhanced discoverability with clear navigation structure
- README now serves as a concise landing page with links to detailed documentation

### Added

- Production-grade security automation with pre-commit hooks
- Placeholder detection in configuration validator
- Startup environment validation script (`scripts/validate_env.py`)
- Comprehensive contributor documentation (CONTRIBUTING.md)
- Security policy and responsible disclosure process (SECURITY.md)
- 5-minute quick start guide (QUICK_START.md)
- Standard repository documentation (CHANGELOG.md, CODE_OF_CONDUCT.md)

### Changed

- Migrated pre-commit config to latest format
- Enhanced `.gitignore` for better security (excludes `.env` files)
- Updated `README.md` with security best practices section
- Replaced exposed API keys with placeholders in `.env`

### Security

- **CRITICAL**: Removed exposed NVIDIA API keys from `.env` file
- **CRITICAL**: Removed exposed PubMed API keys from `.env` file
- Added automated secret scanning via pre-commit hooks
- Implemented 16 automated code quality and security gates
- Created `.env.example` as secure configuration template

## [2.1.0] - 2025-09-30

### Added

- Comprehensive pharmaceutical benchmarking system
- 13 new test files for pharmaceutical validation
- Test fixtures for benchmark validation
- NGC deprecation immunity architecture
- NVIDIA Build platform integration (cloud-first strategy)

### Changed

- Migrated from NGC API to NVIDIA Build platform
- Updated embedding models to latest versions
- Enhanced pharmaceutical metadata extraction

### Removed

- `compose-ngc.yaml` (deprecated, NGC-independent now)

## [2.0.0] - 2025-09-20

### Added

- NeMo 2.0 hybrid architecture
- Pharmaceutical domain excellence features
- Drug interaction analysis
- Clinical study filtering
- Species-specific search capabilities

### Changed

- Major architectural refactoring
- Improved modular design
- Enhanced medical guardrails

## [1.0.0] - 2025-08-15

### Added

- Initial release
- Basic RAG template with NVIDIA embeddings
- LangChain integration
- Streamlit web interface
- FAISS vector database
- PDF document processing

---

## Version History Summary

- **v2.1.x**: Pharmaceutical benchmarking + NGC deprecation immunity
- **v2.0.x**: NeMo 2.0 + pharmaceutical domain features
- **v1.x**: Initial RAG template release

## Migration Guides

### Migrating to v2.1+ (NGC Deprecation Immunity)

**IMPORTANT**: v2.1+ is NGC-independent. No action required for existing users.

1. Your existing NVIDIA API key continues to work
2. No NGC account required
3. Self-hosted fallback options available

See [NGC_DEPRECATION_IMMUNITY.md](docs/NGC_DEPRECATION_IMMUNITY.md) for details.

### Migrating from v1.x to v2.0+

**Breaking Changes:**

- New pharmaceutical metadata schema
- Updated embedding model dimensions
- Changed vector database structure

**Migration Steps:**

1. Backup your existing vector database
2. Update dependencies: `pip install -r requirements.txt`
3. Rebuild vector database: `python scripts/rebuild_vectordb.py`
4. Update `.env` configuration (see `.env.example`)

## Deprecation Notices

### Deprecated in v2.1

- **NGC API access**: Will be removed March 2026 (already migrated)
- **Old embedding models**: Migrated to E5-v5, Mistral7B-v2

### Removed in v2.1

- `compose-ngc.yaml`: Replaced with NGC-independent `docker-compose.yml`
- Legacy NVIDIA API endpoint fallbacks

## Security Updates

### v2.1.1 (Pending)

- **CRITICAL**: Git history cleanup (remove historical API keys)
- Added automated security scanning in CI/CD
- Enhanced pre-commit hooks for secret prevention

### v2.1.0

- Implemented proper API key management
- Created secure `.env.example` template
- Added security documentation

## Support

- **Bug Reports**: [GitHub Issues](https://github.com/your-org/RAG-Template-for-NVIDIA-nemoretriever/issues)
- **Security Issues**: See [SECURITY.md](SECURITY.md)
- **Questions**: [GitHub Discussions](https://github.com/your-org/RAG-Template-for-NVIDIA-nemoretriever/discussions)

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/) principles. Each version documents Added, Changed, Deprecated, Removed, Fixed, and Security updates.
