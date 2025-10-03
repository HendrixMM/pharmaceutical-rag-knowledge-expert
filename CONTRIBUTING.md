# Contributing to RAG Template for NVIDIA NemoRetriever

---

Last Updated: 2025-10-03
Owner: Community Team
Review Cadence: Monthly

---

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🚀 Quick Start for Contributors

### 1. Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/RAG-Template-for-NVIDIA-nemoretriever.git
cd RAG-Template-for-NVIDIA-nemoretriever

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks (CRITICAL for security)
pre-commit install

# Configure environment
cp .env.example .env
# Edit .env and add your real API keys (see README.md Security section)

# Validate setup
python scripts/validate_env.py
```

### 2. Running Tests

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/test_pharmaceutical_benchmarks.py
pytest -m integration  # Integration tests only
pytest -m unit  # Unit tests only

# Run with coverage
make coverage
```

### 3. Code Quality Checks

```bash
# Run all quality checks
make quality

# Individual checks
make lint     # Linting with flake8
make format   # Auto-format with black
make security # Security audit with bandit
make type-check  # Type checking with mypy
```

## 📋 Contribution Workflow

### 1. Before You Start

- **Check existing issues**: Avoid duplicate work
- **Create an issue**: Discuss your idea before implementing
- **Get feedback**: Wait for maintainer approval on significant changes

### 2. Development Process

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**

   - Write clean, readable code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test thoroughly**

   ```bash
   # Run tests
   make test

   # Check code quality
   make quality

   # Validate environment
   python scripts/validate_env.py
   ```

4. **Commit your changes**
   ```bash
   # Pre-commit hooks will run automatically
   git add .
   git commit -m "feat: add new feature description"
   ```

### 3. Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `security`: Security improvements

**Examples:**

```bash
feat(embeddings): add support for new NVIDIA embedding model
fix(auth): resolve API key validation bug
docs(security): update API key management guidelines
security: implement placeholder detection in config validator
```

### 4. Pull Request Process

1. **Push your branch**

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**

   - Use a descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Add screenshots/demos if applicable

3. **Code Review**

   - Address reviewer feedback promptly
   - Keep discussions constructive
   - Update tests if requested

4. **Merge**
   - Maintainer will merge once approved
   - Your branch will be deleted automatically

## 🔒 Security Guidelines

### CRITICAL: Never Commit Secrets

- **Pre-commit hooks** will block commits with API keys
- Always use `.env` for local configuration (gitignored)
- Use `.env.example` for templates (tracked, has placeholders)
- See [SECURITY.md](SECURITY.md) for details

### If You Accidentally Commit Secrets

1. **Immediately notify maintainers** via security@yourorg.com
2. **Rotate the exposed keys** at their respective portals
3. **DO NOT** try to hide it - we need to address it properly

## 📝 Code Style Guidelines

### Python Style

- **PEP 8** compliance (enforced by flake8)
- **Black** formatting (line length: 100)
- **Google-style docstrings**
- **Type hints** for public APIs

**Example:**

```python
def process_query(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    max_results: int = 10
) -> List[Document]:
    """
    Process a pharmaceutical query and return relevant documents.

    Args:
        query: The user's search query
        filters: Optional metadata filters to apply
        max_results: Maximum number of results to return

    Returns:
        List of relevant documents sorted by relevance

    Raises:
        ValueError: If query is empty or max_results is invalid
    """
    # Implementation
    pass
```

### Pharmaceutical Domain Guidelines

- Use standardized drug nomenclature (generic names preferred)
- Include medical disclaimers where appropriate
- Follow FDA/regulatory best practices
- Document clinical data sources

## 🧪 Testing Guidelines

### Test Coverage Requirements

- **Minimum coverage**: 80% for new code
- **Critical paths**: 100% coverage required
- **Edge cases**: Always test error scenarios

### Test Organization

```
tests/
├── unit/                    # Unit tests (fast, isolated)
├── integration/             # Integration tests (slower)
├── fixtures/               # Test data and fixtures
└── test_*.py               # Test files
```

### Writing Tests

```python
import pytest
from src.your_module import your_function

class TestYourFeature:
    """Test suite for your feature."""

    def test_basic_functionality(self):
        """Test basic happy path."""
        result = your_function("input")
        assert result == "expected_output"

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            your_function(invalid_input)

    @pytest.mark.integration
    def test_integration(self):
        """Test integration with external service."""
        # Integration test code
        pass
```

## 📚 Documentation

### When to Update Docs

- **New features**: Always document public APIs
- **Breaking changes**: Update migration guides
- **Bug fixes**: Update if behavior changes
- **Performance**: Document optimization impacts

### Documentation Locations

- `README.md`: Overview and quick start
- `docs/DEVELOPMENT.md`: Development setup and guidelines
- `docs/API_REFERENCE.md`: Configuration and API reference
- `docs/EXAMPLES.md`: Usage examples and runnable snippets
- `docs/_shared/toc.md`: Canonical documentation map
- Code docstrings: API documentation
- `CHANGELOG.md`: Version history (maintained by maintainers)

### Documentation Standards

- Include ownership metadata at the top of new docs:
  ```
  ---
  Last Updated: YYYY-MM-DD
  Owner: Team Name
  Review Cadence: Weekly|Bi-weekly|Monthly|Quarterly|Annually
  ---
  ```
- Run `make docs` to refresh tables of contents.
- Run `make docs-linkcheck` to validate internal links.
- See `docs/_shared/toc.md` for organization and placement guidance.

### Documentation Maintenance

- When changing headings in `README.md` or files under `docs/`, run `make docs` to refresh TOC blocks.
- Pre-commit checks validate only changed files:
  - `docs-metadata-check` enforces metadata on modified docs.
  - `docs-linkcheck` validates internal links for modified `README.md` and `docs/` files (excludes `tests/` and `easyapi-*`).
- CI on pull requests runs quick, non-network checks:
  - TOC drift check (non-writing) to catch stale TOCs.
  - Internal link validation and metadata validation.
- External links are validated weekly in a scheduled workflow to avoid PR bottlenecks.
- To verify locally without modifying files, run `make docs-check-toc`; for a full pass use `make docs-all`.

## 🐛 Bug Reports

### Good Bug Reports Include

1. **Clear description** of the issue
2. **Steps to reproduce**
3. **Expected vs actual behavior**
4. **Environment details**:
   - OS and version
   - Python version
   - Package versions (`pip freeze`)
5. **Error messages/logs**
6. **Screenshots** if applicable

### Bug Report Template

```markdown
**Description**
Brief description of the bug

**To Reproduce**

1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**

- OS: macOS 13.0
- Python: 3.9.7
- Package versions: (paste pip freeze output)

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Before Requesting

- Search existing issues
- Check roadmap
- Consider if it fits project scope

### Feature Request Template

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How would you solve it?

**Alternatives**
What alternatives have you considered?

**Additional Context**
Any other relevant information
```

## 🤝 Code of Conduct

### Our Standards

- **Be respectful**: Treat everyone with respect
- **Be inclusive**: Welcome diverse perspectives
- **Be constructive**: Provide helpful feedback
- **Be professional**: Keep discussions on-topic

### Unacceptable Behavior

- Harassment or discrimination
- Personal attacks
- Trolling or insulting comments
- Publishing private information

### Enforcement

- **First offense**: Warning
- **Second offense**: Temporary ban
- **Repeated offenses**: Permanent ban

Report violations to: conduct@yourorg.com

## 📞 Getting Help

- **Questions**: Use GitHub Discussions
- **Bugs**: Create GitHub Issue
- **Security**: Email security@yourorg.com
- **General**: community@yourorg.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

## Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project README

Thank you for contributing! 🎉
