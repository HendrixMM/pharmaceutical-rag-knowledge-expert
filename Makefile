# Pharmaceutical RAG System - Development Commands
# Run 'make help' to see available commands

.PHONY: help install install-dev install-all clean test test-unit test-integration
.PHONY: lint format check security coverage quality fix pre-commit setup-dev

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