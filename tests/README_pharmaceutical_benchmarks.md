# Pharmaceutical Benchmarking Test Suite

Comprehensive test suite for the pharmaceutical benchmarking system, validating the 3 verification comment fixes.

## 📊 Test Suite Overview

| Metric | Value |
|--------|-------|
| **Total Tests** | 61 tests across 4 files |
| **Test Files** | 4 test files + fixtures |
| **Execution Time** | ~3.5 seconds (fast unit/integration/e2e tests) |
| **Pass Rate** | 100% (all tests passing) |
| **Code Coverage** | 76.08% (run_pharmaceutical_benchmarks.py)<br>32.73% (pharmaceutical_benchmark_report.py) |

## 🗂️ Test Organization

### File Structure

```
tests/
├── conftest.py                              # 10+ pharmaceutical test fixtures
├── fixtures/                                 # Test data directory
│   ├── benchmark_sample_v1.json             # 5-query test benchmark
│   ├── baseline_metadata_valid.json         # Valid baseline structure
│   ├── baseline_metadata_invalid.json       # Missing fields example
│   ├── client_response_openai_format.json   # OpenAI-style response
│   └── client_response_custom_format.json   # Custom response format
├── test_baseline_metadata_validation.py     # 16 tests for Comment 3
├── test_regression_detection.py             # 21 tests for Comment 2
├── test_benchmark_runner_integration.py     # 16 tests for Comment 1
└── test_pharmaceutical_e2e.py               # 8 E2E tests (Mini-Phase B)
```

### Test Files by Verification Comment

#### 1. test_baseline_metadata_validation.py (16 tests)
**Validates**: Comment 3 - Baseline metadata in benchmark JSON files

**Test Classes** (6):
- `TestBaselineMetadataPresence` - Validates baselines exist
- `TestCloudBaselineStructure` - Cloud baseline fields & types
- `TestSelfHostedBaselineStructure` - Self-hosted baseline fields
- `TestRegressionThresholds` - Threshold structure validation
- `TestCloudVsSelfHostedComparison` - Relationship validation
- `TestBaselineMetadataCompleteness` - Overall completeness

**What It Tests**:
- ✅ All 5 benchmark categories have baseline metadata
- ✅ Cloud and self_hosted sections have required fields
- ✅ Regression thresholds (5%, 20%, 50%) are correct
- ✅ Cloud cost > 0, self_hosted cost = 0
- ✅ Cloud latency < self_hosted latency
- ✅ Notes fields populated with context

**Markers**: `@pytest.mark.pharmaceutical`, `@pytest.mark.baseline_validation`, `@pytest.mark.unit`, `@pytest.mark.fast`

#### 2. test_regression_detection.py (21 tests)
**Validates**: Comment 2 - Regression detection math fix

**Test Classes** (6):
- `TestAccuracyRegressionDetection` - Accuracy threshold tests
- `TestCostRegressionDetection` - Cost threshold tests
- `TestLatencyRegressionDetection` - Latency threshold tests
- `TestMultipleRegressionsDetection` - Multiple regression handling
- `TestEdgeCases` - Zero baselines, boundary conditions
- `TestPercentageCalculationAccuracy` - Mathematical correctness

**What It Tests**:
- ✅ Fixed calculation: `((current - baseline) / baseline) * 100`
- ✅ Original bug prevented: `cost_change / max(0.01, abs(cost_change - cost_change))`
- ✅ 5% accuracy drop triggers regression
- ✅ 20% cost increase triggers regression
- ✅ 50% latency increase triggers regression
- ✅ Zero baseline edge cases handled
- ✅ Multiple simultaneous regressions detected

**Markers**: `@pytest.mark.pharmaceutical`, `@pytest.mark.regression_detection`, `@pytest.mark.unit`, `@pytest.mark.fast`

**Coverage**: 24.09% of `scripts/pharmaceutical_benchmark_report.py`

#### 3. test_benchmark_runner_integration.py (16 tests)
**Validates**: Comment 1 - Real client integration

**Test Classes** (7):
- `TestRunnerInitialization` - Client initialization paths
- `TestExecuteQueryFullPipeline` - Classify → execute → track pipeline
- `TestResponseExtraction` - Response format parsing
- `TestCreditsEstimation` - Pharmaceutical multipliers
- `TestDomainMapping` - Domain → QueryType mapping
- `TestModeSelection` - Cloud/self_hosted/both modes
- `TestCLIArguments` - CLI argument handling

**What It Tests**:
- ✅ BenchmarkRunner uses real EnhancedNeMoClient
- ✅ PharmaceuticalQueryClassifier integration
- ✅ PharmaceuticalCostAnalyzer tracking
- ✅ Mode selection (cloud/self_hosted/both)
- ✅ Response extraction (OpenAI/custom formats)
- ✅ Safety query 1.5x credit multiplier
- ✅ Error handling and graceful fallbacks

**Markers**: `@pytest.mark.pharmaceutical`, `@pytest.mark.client_integration`, `@pytest.mark.integration`, `@pytest.mark.fast`, `@pytest.mark.cost_optimization`

**Coverage**: Contributes to 76.08% coverage of `scripts/run_pharmaceutical_benchmarks.py`

#### 4. test_pharmaceutical_e2e.py (8 tests) **[NEW - Mini-Phase B]**
**Validates**: End-to-end workflows across all verification comments

**Test Classes** (3):
- `TestBenchmarkExecutionE2E` - Complete benchmark execution workflow (lines 400-488)
- `TestCLIExecutionE2E` - CLI interface testing (lines 506-586)
- `TestRegressionDetectionWorkflowE2E` - Regression detection workflow (lines 186-231)

**What It Tests**:
- ✅ Full benchmark run: load → execute → evaluate → aggregate
- ✅ Query failure handling and error recovery
- ✅ CLI argument parsing and execution
- ✅ --simulate, --mode, --save-results options
- ✅ Multiple benchmark execution (all categories)
- ✅ Complete regression detection workflow with baseline comparison
- ✅ Missing category handling
- ✅ Threshold boundary conditions

**Markers**: `@pytest.mark.pharmaceutical`, `@pytest.mark.e2e`, `@pytest.mark.fast`

**Coverage**: +30.23% improvement to `run_pharmaceutical_benchmarks.py` (45.85% → 76.08%)<br>+8.64% improvement to `pharmaceutical_benchmark_report.py` (24.09% → 32.73%)

## 🎯 Pytest Markers

Tests are organized with pytest markers for selective execution:

### Verification Comment Markers
```bash
# Comment 3: Baseline metadata validation
pytest -m baseline_validation -v

# Comment 2: Regression detection logic
pytest -m regression_detection -v

# Comment 1: Client integration paths
pytest -m client_integration -v
```

### Standard Markers
```bash
# All pharmaceutical benchmark tests
pytest -m pharmaceutical -v

# Unit tests only
pytest -m "unit and pharmaceutical" -v

# Integration tests only
pytest -m "integration and pharmaceutical" -v

# E2E tests only (Mini-Phase B)
pytest -m e2e -v

# Fast tests (< 1s each)
pytest -m fast -v

# Cost optimization tests
pytest -m cost_optimization -v
```

### Combined Markers
```bash
# All new pharmaceutical tests
pytest -m "baseline_validation or regression_detection or client_integration" -v

# Unit tests for Comments 2 and 3
pytest -m "(baseline_validation or regression_detection) and unit" -v
```

## 🧪 Test Fixtures

Comprehensive fixtures defined in `tests/conftest.py`:

### Mock Fixtures
```python
mock_enhanced_nemo_client         # Mock EnhancedNeMoClient for API testing
mock_pharmaceutical_classifier    # Mock PharmaceuticalQueryClassifier
mock_cost_analyzer               # Mock PharmaceuticalCostAnalyzer
```

### Data Fixtures
```python
sample_benchmark_data            # Complete benchmark JSON structure
sample_baseline_metadata         # Baseline metadata structure
sample_client_response_openai_format    # OpenAI-style response
sample_client_response_custom_format    # Custom response format
```

### Path Fixtures
```python
benchmarks_directory            # Path to benchmarks/
test_fixtures_directory         # Path to tests/fixtures/
```

### Fixture Usage Example
```python
def test_example(mock_enhanced_nemo_client, sample_benchmark_data):
    """Example test using fixtures."""
    # mock_enhanced_nemo_client is pre-configured
    # sample_benchmark_data contains valid benchmark structure
    pass
```

## 🚀 Running Tests

### Quick Start
```bash
# Run all pharmaceutical benchmark tests
pytest tests/test_baseline_metadata_validation.py \
       tests/test_regression_detection.py \
       tests/test_benchmark_runner_integration.py -v

# Run with markers (faster)
pytest -m pharmaceutical -v
```

### Coverage Analysis
```bash
# Coverage for regression detection
pytest tests/test_regression_detection.py \
    --cov=scripts.pharmaceutical_benchmark_report \
    --cov-report=html

# Coverage for runner integration
pytest tests/test_benchmark_runner_integration.py \
    --cov=scripts.run_pharmaceutical_benchmarks \
    --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

### Selective Execution
```bash
# Only baseline validation tests (Comment 3)
pytest -m baseline_validation -v

# Only regression detection tests (Comment 2)
pytest -m regression_detection -v

# Only client integration tests (Comment 1)
pytest -m client_integration -v

# Fast tests only
pytest -m fast -v

# Specific test class
pytest tests/test_regression_detection.py::TestAccuracyRegressionDetection -v

# Specific test method
pytest tests/test_baseline_metadata_validation.py::TestCloudBaselineStructure::test_cloud_baseline_required_fields -v
```

### CI/CD Integration
```bash
# Fail fast on first error
pytest -m pharmaceutical -x

# Run in parallel (if pytest-xdist installed)
pytest -m pharmaceutical -n auto

# Generate JUnit XML for CI
pytest -m pharmaceutical --junitxml=junit.xml

# Quiet mode with summary
pytest -m pharmaceutical -q --tb=line
```

## 📈 Coverage Targets

| File | Current Coverage | Target | Status |
|------|-----------------|--------|--------|
| `scripts/run_pharmaceutical_benchmarks.py` | **76.08%** | 80%+ | 🟢 Nearly complete (+30.23% from Phase A/B) |
| `scripts/pharmaceutical_benchmark_report.py` | **32.73%** | 80%+ | 🟡 Needs improvement (+8.64% from Phase A/B) |
| `src/monitoring/pharmaceutical_benchmark_tracker.py` | 0% | 70%+ | 🔴 Not directly tested |

### Coverage Gaps

**run_pharmaceutical_benchmarks.py** (Missing ~24%):
- ✅ ~~Simulation mode execution paths (lines 300-488)~~ **NOW COVERED**
- ✅ ~~CLI argument parsing and main() (lines 492-584)~~ **NOW COVERED**
- Some initialization edge cases (lines 35-41, 61-62, 104-127)
- File I/O operations (lines 492-503)
- Error handling edge cases in execution flow

**pharmaceutical_benchmark_report.py** (Missing ~67%):
- ✅ ~~Comparison report generation (lines 186-231)~~ **NOW COVERED**
- Report generation methods (lines 41-176)
- Markdown/HTML formatting (lines 183-231)
- File loading utilities (lines 281-385)

### Improving Coverage

To reach 80%+ coverage targets:

1. **Add E2E tests** - Test full workflow including report generation
2. **Test simulation mode** - Cover execute_query_simulated() path
3. **Test CLI parsing** - Add argparse tests
4. **Test report formats** - Test markdown/HTML/JSON generation
5. **Add parametrized tests** - Test multiple input variations

## 🏗️ Test Design Patterns

### AAA Pattern
All tests follow Arrange-Act-Assert:
```python
def test_example(self):
    # Arrange: Setup test data
    baseline_accuracy = 0.85
    current_accuracy = 0.80

    # Act: Execute the code under test
    flags = ComparisonReportGenerator._check_regressions(
        baseline_accuracy=baseline_accuracy,
        current_accuracy=current_accuracy,
        # ... other params
    )

    # Assert: Verify expected behavior
    assert "accuracy_regression" in flags
```

### Mocking Strategy
- Mock at the module boundary (EnhancedNeMoClient, etc.)
- Use `unittest.mock.Mock` for flexible mocks
- Use `@patch` decorator for clean dependency injection
- Mock responses are realistic (match production formats)

### Test Data Isolation
- All test data in `tests/fixtures/` directory
- No dependencies on production data
- Fixtures provide consistent test data
- Can modify fixtures without affecting tests

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Total Execution Time** | 1.49-3.06 seconds |
| **Average per Test** | ~0.03-0.06 seconds |
| **Slowest Test** | 0.15s (TestBaselineMetadataPresence setup) |
| **Fastest Tests** | < 0.005s (most unit tests) |

All tests are marked `@pytest.mark.fast` as they complete in < 1 second.

## 🐛 Debugging Tests

### Run Single Test with Verbose Output
```bash
pytest tests/test_regression_detection.py::TestAccuracyRegressionDetection::test_accuracy_regression_5_percent_drop -vv
```

### Show Print Statements
```bash
pytest tests/test_baseline_metadata_validation.py -s
```

### Drop into Debugger on Failure
```bash
pytest tests/test_regression_detection.py --pdb
```

### Show Locals on Failure
```bash
pytest tests/test_benchmark_runner_integration.py -l
```

## ✅ Best Practices

1. **Descriptive Test Names** - Test names explain what they validate
2. **Single Assertion Focus** - Each test validates one behavior
3. **No Test Dependencies** - Tests run independently in any order
4. **Fast Execution** - All tests complete in < 3 seconds
5. **Comprehensive Edge Cases** - Zero values, boundaries, errors covered
6. **Proper Mocking** - External dependencies mocked at module boundary
7. **Fixtures for Reuse** - Common test data in reusable fixtures
8. **Markers for Organization** - Tests tagged for selective execution

## 📝 Adding New Tests

### Template for New Test Class
```python
@pytest.mark.pharmaceutical
@pytest.mark.your_marker
@pytest.mark.unit  # or integration
@pytest.mark.fast
class TestYourFeature:
    """Test description."""

    def test_your_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        input_data = "test"

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_value
```

### Checklist for New Tests
- [ ] Descriptive test class and method names
- [ ] Appropriate pytest markers applied
- [ ] Uses existing fixtures where possible
- [ ] Follows AAA pattern (Arrange-Act-Assert)
- [ ] Single, clear assertion per test
- [ ] Includes docstring explaining test purpose
- [ ] Edge cases covered (zero, None, empty, etc.)
- [ ] Runs in < 1 second (marked `@pytest.mark.fast`)

## 🔗 Related Documentation

- [Pharmaceutical Benchmark System](../README.md)
- [Verification Comments](../VERIFICATION_COMMENTS.md)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)

## 📞 Support

For questions or issues with tests:
1. Check test docstrings for expected behavior
2. Run with `-vv` flag for verbose output
3. Review fixture definitions in `conftest.py`
4. Check pytest markers in `pytest.ini`

---

**Last Updated**: 2025-10-01
**Test Suite Version**: 1.1 (Phase A + Mini-Phase B)
**Total Tests**: 61 (53 Phase A + 8 Mini-Phase B)
**Pass Rate**: 100%
**Coverage Improvement**: +30.23% (run_pharmaceutical_benchmarks.py), +8.64% (pharmaceutical_benchmark_report.py)
