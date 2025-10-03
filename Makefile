# Pharmaceutical RAG System - Development Commands
# Run 'make help' to see available commands

.PHONY: help install install-dev install-all clean test test-unit test-integration
.PHONY: lint format check security coverage quality fix pre-commit setup-dev
.PHONY: test-rerank
.PHONY: start-rerank stop-rerank down-all
.PHONY: bench-ci
.PHONY: docs docs-serve docs-linkcheck docs-linkcheck-all docs-validate docs-all docs-check-toc
.PHONY: mcp-github-add mcp-github-verify mcp-github-remove

# Default target
help:
	@echo "🏥 Pharmaceutical RAG System - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "📦 Installation:"
	@echo "  install          Install core dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-all      Install everything (core + dev + medical + nemo)"
	@echo ""
	@echo "🧹 Code Quality:"
	@echo "  quality          Run all quality checks"
	@echo "  format           Auto-fix formatting (black + isort)"
	@echo "  lint             Run linting (flake8)"
	@echo "  security         Run security scan (bandit)"
	@echo "  check            Check without fixing"
	@echo ""
	@echo "🔒 Security:"
	@echo "  secret-scan      Scan git history for exposed secrets"
	@echo "  history-cleanup  Clean git history (requires confirmation)"
	@echo "  verify-cleanup   Verify git history cleanup was successful"
	@echo "  rotate-keys      Display key rotation checklist"
	@echo "  secrets-baseline Update detect-secrets baseline (manual review)"
	@echo "🧪 Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests (requires API keys)"
	@echo "  test-rerank      Run NVIDIA Build rerank test (requires NVIDIA_API_KEY)"
	@echo "  pharma-bench     Run pharma capability checks (requires NVIDIA_API_KEY)"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs             Generate/update table of contents for all docs"
	@echo "  docs-serve       Serve documentation locally (placeholder)"
	@echo "  docs-linkcheck   Validate documentation links (internal only)"
	@echo "  docs-validate    Validate documentation metadata"
	@echo "  docs-check-toc   Check TOCs (no writes); fails if drift"
	@echo "  docs-all         Run all documentation checks (toc, linkcheck, validate)"
	@echo "  start-rerank     Start local reranker NIM with Docker Compose (8502)"
	@echo "  start-embed      Start local embedder NIM with Docker Compose (8501)"
	@echo "  start-extraction Start local extraction NIM with Docker Compose (8503)"
	@echo "  start-local-stack Start embedder + reranker locally (writes .env.local)"
	@echo "  stop-local-stack  Stop embedder + reranker"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "🚀 Benchmarks:"
	@echo "  bench-ci         Run orchestrated preflight→run locally (CI-like)"
	@echo "⚙️  Development Setup:"
	@echo "  setup-dev        Complete development setup"
	@echo "  pre-commit       Install pre-commit hooks"
	@echo "  clean            Clean temporary files"
	@echo ""
	@echo "🧩 MCP (Claude) Integrations:"
	@echo "  mcp-github-add    Configure GitHub MCP server (project-scoped)"
	@echo "  mcp-github-verify Verify GitHub MCP server configuration"
	@echo "  mcp-github-remove Remove GitHub MCP server from project scope"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-all:
	pip install -r requirements.txt -r requirements-nemo.txt -r requirements-medical.txt -r requirements-dev.txt

# Code quality targets
quality:
	python scripts/quality_check.py

format:
	python scripts/quality_check.py --fix

check:
	python scripts/quality_check.py --no-tests

lint:
	flake8 src tests scripts examples

security: audit secret-scan
	bandit -r src

.PHONY: audit
audit:
	@echo "🔎 Running secret audit (pre-commit hook)..." && \
	if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit run secret-audit --all-files || true; \
	else \
		echo "pre-commit not installed; skipping secret-audit"; \
	fi

# Testing targets
test:
	pytest tests

test-unit:
	pytest tests -k "not integration"

test-integration:
	pytest tests -k "integration"

coverage:
	pytest tests --cov=src --cov-report=html --cov-report=term-missing

# NVIDIA Build rerank quick test
test-rerank:
	@if [ "${SKIP_RERANK}" = "true" ]; then \
	  echo "Skipping rerank test (SKIP_RERANK=true)"; \
	else \
	  echo "Running NVIDIA Build rerank test…" && python scripts/nvidia_build_rerank_test.py; \
	fi

# Pharma capability benchmark
.PHONY: pharma-bench pharma-bench-nofallback
pharma-bench:
	@echo "Running pharma capability checks (fallback enabled)…" && \
	python scripts/pharma_benchmarks_cli.py --pretty --fallback=true

pharma-bench-nofallback:
	@echo "Running pharma capability checks (no fallback)…" && \
	python scripts/pharma_benchmarks_cli.py --pretty --fallback=false

.PHONY: start-rerank stop-rerank
start-rerank:
	@bash scripts/start_local_rerank.sh

stop-rerank:
	@echo "Stopping local reranker…" && \
	bash -c 'if docker compose version >/dev/null 2>&1; then docker compose stop reranker || true; docker compose rm -f reranker || true; else docker-compose stop reranker || true; docker-compose rm -f reranker || true; fi'

.PHONY: start-embed stop-embed start-extraction stop-extraction
start-embed:
	@bash scripts/start_local_embed.sh

stop-embed:
	@echo "Stopping local embedder…" && \
	bash -c 'if docker compose version >/dev/null 2>&1; then docker compose stop embedder || true; docker compose rm -f embedder || true; else docker-compose stop embedder || true; docker-compose rm -f embedder || true; fi'

start-extraction:
	@bash scripts/start_local_extraction.sh

stop-extraction:
	@echo "Stopping local extraction…" && \
	bash -c 'if docker compose version >/dev/null 2>&1; then docker compose stop extraction || true; docker compose rm -f extraction || true; else docker-compose stop extraction || true; docker-compose rm -f extraction || true; fi'

.PHONY: start-local-stack stop-local-stack start-local-all stop-local-all
start-local-stack:
	@echo "Starting local embedder and reranker…" && \
	bash scripts/start_local_embed.sh && \
	bash scripts/start_local_rerank.sh && \
	echo "Local stack ready. .env.local updated with endpoints."

stop-local-stack:
	@echo "Stopping local embedder and reranker…" && \
	$(MAKE) -s stop-embed && \
	$(MAKE) -s stop-rerank

start-local-all:
	@echo "Starting local embedder, reranker, and extraction…" && \
	$(MAKE) -s start-local-stack && \
	bash scripts/start_local_extraction.sh

stop-local-all:
	@echo "Stopping local embedder, reranker, and extraction…" && \
	$(MAKE) -s stop-embed && \
	$(MAKE) -s stop-rerank && \
	$(MAKE) -s stop-extraction

down-all:
	@echo "Stopping all local services…" && \
	bash -c 'if docker compose version >/dev/null 2>&1; then docker compose down || true; else docker-compose down || true; fi'

# Development setup
setup-dev: install-dev pre-commit
	@echo "✅ Development environment setup complete!"

pre-commit:
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf bandit-report.json

# Quick development cycle
dev: format lint test-unit
	@echo "🎉 Development cycle complete!"

# CI-like bench orchestration (requires NVIDIA_API_KEY)
bench-ci:
	@echo "Running orchestrated benchmarks (preflight→run)…" && \
	python scripts/orchestrate_benchmarks.py \
	  --mode $${MODE:-both} \
	  --preset $${PRESET:-cloud_first_adaptive} \
	  --output $${OUT:-results/benchmark_runs/orchestrated_local} \
	  --summary-output $${OUT:-results/benchmark_runs/orchestrated_local}/summary.json \
	  --auto-concurrency --skip-classifier-validation \

# MCP (Claude) — GitHub server wiring
mcp-github-add:
	bash scripts/setup_github_mcp.sh

mcp-github-verify:
	bash scripts/verify_github_mcp.sh

mcp-github-remove:
	-claude mcp remove -s project github || true
	@echo "✅ Removed project-scoped 'github' MCP server (if present)."
	  --preflight-sample-count $${PF_SAMPLES:-1} \
	  --preflight-min-concurrency $${PF_MIN_CONC:-2} \
	  --fail-on-preflight \
	  --fail-on-regressions \
	  --min-cloud-score $${MIN_SCORE:-0.30} \
	  --max-cloud-latency-ms $${MAX_P95_MS:-12000} \
	  --max-queries $${MAX_Q:-1}

# Documentation targets
docs:
	@echo "Generating table of contents for documentation..."
	@python scripts/docs_toc_generator.py README.md --in-place || true
	@find docs -name '*.md' -exec python scripts/docs_toc_generator.py {} --in-place \;
	@echo "✅ Documentation TOCs updated"

docs-serve:
	@echo "Documentation serving not yet configured"
	@echo "This target will be implemented in Week 3 with mkdocs"
	@echo "For now, view markdown files directly or use a markdown preview tool"

docs-linkcheck:
	@echo "Checking documentation links..."
	@python scripts/docs_linkcheck.py --skip-external --exclude tests/ --exclude easyapi-pubmed-optimization-guide.md --exclude easyapi-pubmed-integration.md
	@echo "✅ Link check complete"

docs-linkcheck-all:
	@echo "Checking all documentation links (including external)..."
	@python scripts/docs_linkcheck.py
	@echo "✅ Full link check complete"

docs-validate:
	@echo "Validating documentation metadata..."
	@python scripts/docs_metadata_validator.py --path docs/ --exclude docs/_shared/metadata_template.md
	@echo "✅ Metadata validation complete"

docs-all: docs docs-linkcheck docs-validate
	@echo "✅ All documentation checks passed"

# Non-writing TOC drift check (CI-friendly)
docs-check-toc:
	@echo "Checking TOC drift (non-writing)..."
	@python scripts/docs_toc_generator.py README.md --check --quiet || exit 1
	@find docs -name '*.md' -exec python scripts/docs_toc_generator.py {} --check --quiet \; || exit 1
	@echo "✅ TOCs are up to date"

# Security: Git history cleanup targets
.PHONY: secret-scan history-cleanup verify-cleanup rotate-keys

secret-scan:
	@echo "🔍 Scanning git history for exposed secrets..." && \
	bash scripts/identify_secrets_in_history.sh --report backups/secret-scan-$(shell date +%Y%m%d-%H%M%S).txt || true && \
	echo "✅ Secret scan complete. Review backups/secret-scan-*.txt"

history-cleanup:
	@echo "⚠️  WARNING: This will rewrite git history (force-push required)" && \
	echo "📖 Review docs/security/history-redaction.md before proceeding" && \
	echo "" && \
	read -p "Have you reviewed the documentation and coordinated with the team? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		bash scripts/git_history_cleanup.sh; \
	else \
		echo "❌ Cleanup aborted. Review documentation first."; \
		exit 1; \
	fi

verify-cleanup:
	@echo "✅ Verifying git history cleanup..." && \
	bash scripts/verify_history_cleanup.sh && \
	echo "📊 Verification complete. Check backups/verification-report-*.txt"

rotate-keys:
	@echo "🔑 API Key Rotation Checklist" && \
	echo "═══════════════════════════════════════════════════════════" && \
	echo "" && \
	echo "1. NVIDIA API Key:" && \
	echo "   - Visit: https://build.nvidia.com" && \
	echo "   - Revoke old key" && \
	echo "   - Generate new key" && \
	echo "   - Test: python scripts/nvidia_build_api_test.py" && \
	echo "" && \
	echo "2. PubMed API Key:" && \
	echo "   - Visit: https://www.ncbi.nlm.nih.gov/account/" && \
	echo "   - Delete old key" && \
	echo "   - Create new key" && \
	echo "   - Update PUBMED_EUTILS_API_KEY in .env" && \
	echo "" && \
	echo "3. Apify Token:" && \
	echo "   - Visit: https://console.apify.com" && \
	echo "   - Revoke old token" && \
	echo "   - Generate new token" && \
	echo "   - Update APIFY_TOKEN in .env" && \
	echo "" && \
	echo "4. Update CI/CD Secrets:" && \
	echo "   - GitHub: Settings → Secrets → Actions" && \
	echo "   - Update all relevant secrets" && \
	echo "" && \
	echo "5. Verify:" && \
	echo "   - Run: make test" && \
	echo "   - Check usage logs for unauthorized access" && \
	echo "" && \
	echo "6. Document:" && \
	echo "   - Update: docs/security/key-rotation-tracker.md" && \
	echo "   - Add rotation date and new key IDs" && \
	echo "" && \
	echo "═══════════════════════════════════════════════════════════" && \
	echo "📖 Full documentation: docs/security/key-rotation-tracker.md"

secrets-baseline:
	@echo "🔐 Updating detect-secrets baseline..." && \
	detect-secrets scan --baseline .secrets.baseline --update && \
	echo "✅ Baseline updated. Review and commit .secrets.baseline"
