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
  - Scrubbed NVIDIA API keys (nvapi-\*) from all commits
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

- Enhanced `docs/ARCHITECTURE.md` from placeholder to comprehensive architecture documentation (Phase 2.1)
- All GitHub templates reference existing `CONTRIBUTING.md` and `SECURITY.md`
- Templates include pharmaceutical-specific fields and considerations
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

- **GitHub Templates** (Phase 2.1):
  - Created `.github/ISSUE_TEMPLATE/bug_report.yml` - Structured bug report template with pharmaceutical context
  - Created `.github/ISSUE_TEMPLATE/feature_request.yml` - Feature request template with use case classification
  - Created `.github/ISSUE_TEMPLATE/security_disclosure.yml` - Security vulnerability disclosure template
  - Created `.github/ISSUE_TEMPLATE/config.yml` - Issue template configuration with contact links
  - Created `.github/PULL_REQUEST_TEMPLATE.md` - Comprehensive PR template with security, docs, and data-protection checkboxes
  - Created `.github/TEMPLATE_VALIDATION.md` - Template validation guide and testing procedures
- **Code of Conduct**:
  - Created `CODE_OF_CONDUCT.md` using Contributor Covenant v2.1
  - Added pharmaceutical domain-specific standards
  - Included enforcement guidelines and contact information
- **Enhanced Architecture Documentation**:
  - Added C4 Level 1 system context diagram (Mermaid) to `docs/ARCHITECTURE.md`
  - Added detailed data flow visualization showing complete RAG pipeline
  - Added comprehensive guardrail component tracing:
    - Input rails validation flow
    - Retrieval rails source validation
    - Output rails safety checks and disclaimer injection
    - Pharmaceutical safety modules documentation
  - Documented all guardrails components with file references
  - Added deployment architecture details
  - Expanded technology stack and design principles
- Documentation tooling and automation:
  - `scripts/docs_linkcheck.py` for validating documentation links
  - `scripts/docs_toc_generator.py` for automatic table of contents generation
  - `scripts/docs_metadata_validator.py` for metadata validation
  - Makefile targets: `docs`, `docs-linkcheck`, `docs-validate`, `docs-all`
  - Pre-commit hooks for documentation validation

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

### Added (Phase 2.2: Documentation Enhancements)

- **Architecture Decision Records (ADRs)**:
  - Created `docs/adr/` directory structure
  - Added `docs/adr/0001-use-nemo-retriever.md` - NeMo Retriever technology selection rationale
  - Added `docs/adr/template.md` - ADR template for future decisions
  - Added `docs/adr/README.md` - ADR index and usage guidelines
  - Established bidirectional traceability between ADRs and implementation docs

- **Enhanced API Documentation**:
  - Expanded `docs/API_REFERENCE.md` from 423 to 835 lines
  - Added comprehensive edge case handling (8 scenarios)
  - Added cURL examples for API testing
  - Added rate limiting configuration reference
  - Added pharmaceutical-specific API patterns

- **Examples Documentation**:
  - Enhanced `docs/EXAMPLES.md` from 97 to 1,285 lines
  - Added 12 complete code examples with pharmaceutical focus
  - Added drug interaction analysis workflows
  - Added clinical trial processing examples
  - Added batch processing patterns
  - Added error handling best practices

- **Benchmarks Documentation**:
  - Created `docs/BENCHMARKS.md` (759 lines)
  - Added pharmaceutical research benchmarks
  - Added performance metrics (latency, throughput, cost per query)
  - Added accuracy metrics for drug safety queries
  - Added free tier optimization strategies
  - Added comparison tables for model selection

- **Troubleshooting Enhancements**:
  - Enhanced `docs/TROUBLESHOOTING_GUIDE.md` from 312 to 1,804 lines
  - Added Mermaid diagnostic decision tree
  - Added 15+ categorized troubleshooting scenarios
  - Added NGC deprecation troubleshooting
  - Added pharmaceutical-specific debugging
  - Added performance optimization guides

- **Documentation Infrastructure**:
  - Created `mkdocs.yml` with Material theme configuration
  - Added navigation structure for 30+ documentation files
  - Configured search, syntax highlighting, and Mermaid support
  - Added extra CSS/JS for MathJax and custom styling
  - Set up strict mode validation for link integrity

- **Missing Navigation Files**:
  - Created `docs/index.md` - Documentation home page (8.4KB)
  - Created `docs/INSTALLATION.md` - Installation guide (11KB)
  - Created `docs/QUICK_START.md` - Quick start guide (9.4KB)
  - Created `docs/CONFIGURATION.md` - Configuration reference (5.3KB)
  - Created `docs/MONITORING.md` - Monitoring guide (9.3KB)
  - Created `docs/TESTING.md` - Testing framework docs (10KB)
  - Created `docs/ROADMAP.md` - Project roadmap (6.9KB)
  - Created `docs/SUPPORT.md` - Support resources (9.5KB)
  - Symlinked `docs/CONTRIBUTING.md`, `docs/CHANGELOG.md`, `docs/SECURITY.md`, `docs/CODE_OF_CONDUCT.md`

### Changed (Phase 2.2)

- Updated `docs/ARCHITECTURE.md` to include ADR references and bidirectional traceability
- Updated `docs/NGC_DEPRECATION_IMMUNITY.md` to reference ADR-0001
- Fixed 46 broken documentation links for `mkdocs build --strict` compliance
- Standardized all documentation with metadata (Last Updated, Owner, Review Cadence)
- Updated dependency list: added `mkdocs-material`, `mkdocs-minify-plugin`

### Documentation (Phase 2.2)

- All documentation files now build successfully with `mkdocs build --strict`
- Established ADR workflow for architectural decisions
- Improved documentation discoverability with centralized navigation
- Added MkDocs local preview capability (`mkdocs serve`)

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

See [NGC_DEPRECATION_IMMUNITY.md](NGC_DEPRECATION_IMMUNITY.md) for details.

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

### Post-History-Cleanup Actions (v2.1.1+)

IMPORTANT: If you cloned this repository before 2025-10-03, you must re-clone.

The git history was rewritten on 2025-10-03 to remove exposed API keys. This was a force-push operation.

Action Required:

1. Backup your local changes: `git stash` or commit to a branch
2. Delete your local clone: `rm -rf RAG-Template-for-NVIDIA-nemoretriever`
3. Re-clone the repository: `git clone [repo-url]`
4. Restore your changes: apply stashed changes or cherry-pick commits
5. Update your `.env`: copy `.env.example` to `.env` and add new API keys

For Fork Owners:
If you maintain a fork, see `docs/security/history-redaction.md` for instructions on applying the cleanup to your fork.

## Deprecation Notices

### Deprecated in v2.1

- **NGC API access**: Will be removed March 2026 (already migrated)
- **Old embedding models**: Migrated to E5-v5, Mistral7B-v2

### Removed in v2.1

- `compose-ngc.yaml`: Replaced with NGC-independent `docker-compose.yml`
- Legacy NVIDIA API endpoint fallbacks

## Security Updates

### v2.1.1 (2025-10-03)

- **CRITICAL**: Git history cleanup completed (removed historical API keys)
- All exposed NVIDIA, PubMed, and Apify keys removed from history
- Force-push completed on 2025-10-03
- All keys rotated and verified
- Added `detect-secrets` baseline for automated secret prevention
- Enhanced pre-commit hooks with industry-standard secret detection
- Comprehensive security documentation and runbooks created

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
