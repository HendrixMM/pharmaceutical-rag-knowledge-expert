# Pharmaceutical RAG System - Development Commands
# Run 'make help' to see available commands

.PHONY: help install install-dev install-all clean test test-unit test-integration
.PHONY: lint format check security coverage quality fix pre-commit setup-dev
.PHONY: test-rerank
.PHONY: start-rerank stop-rerank down-all

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
	@echo "🧪 Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests (requires API keys)"
	@echo "  test-rerank      Run NVIDIA Build rerank test (requires NVIDIA_API_KEY)"
	@echo "  pharma-bench     Run pharma capability checks (requires NVIDIA_API_KEY)"
	@echo "  start-rerank     Start local reranker NIM with Docker Compose (8502)"
	@echo "  start-embed      Start local embedder NIM with Docker Compose (8501)"
	@echo "  start-extraction Start local extraction NIM with Docker Compose (8503)"
	@echo "  start-local-stack Start embedder + reranker locally (writes .env.local)"
	@echo "  stop-local-stack  Stop embedder + reranker"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "⚙️  Development Setup:"
	@echo "  setup-dev        Complete development setup"
	@echo "  pre-commit       Install pre-commit hooks"
	@echo "  clean            Clean temporary files"

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

security:
	bandit -r src

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
