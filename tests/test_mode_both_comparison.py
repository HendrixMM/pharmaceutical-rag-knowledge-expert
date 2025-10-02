"""
Test Mode 'Both' Comparison Feature

Tests for Comment 2 verification: cloud vs self-hosted comparison feature.
Tests the new mode="both" functionality that executes queries against both
cloud and self-hosted endpoints and compares results.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pharmaceutical_benchmarks import BenchmarkRunner, BenchmarkConfig


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.unit
@pytest.mark.fast
class TestExecuteQueryWithEndpoint:
    """Test execute_query_with_endpoint() method for forced endpoint execution."""

    def test_execute_query_with_cloud_endpoint_simulated(self):
        """Test query execution forced to cloud endpoint in simulation mode."""
        config = BenchmarkConfig()
        from scripts.run_pharmaceutical_benchmarks import EndpointType

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')
        query = "What are the side effects of metformin?"

        # Execute with cloud endpoint (simulated)
        response, latency, credits = runner.execute_query_with_endpoint(
            query, timeout=30, endpoint=EndpointType.CLOUD
        )

        # Verify response structure (simulated responses)
        assert isinstance(response, str)
        assert len(response) > 0  # Simulation returns non-empty response
        assert isinstance(latency, float)
        assert latency >= 0
        assert isinstance(credits, int)
        assert credits >= 0

    def test_execute_query_with_self_hosted_endpoint_simulated(self):
        """Test query execution forced to self-hosted endpoint in simulation mode."""
        config = BenchmarkConfig()
        from scripts.run_pharmaceutical_benchmarks import EndpointType

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')
        query = "What are the drug interactions with aspirin?"

        # Execute with self-hosted endpoint (simulated)
        response, latency, credits = runner.execute_query_with_endpoint(
            query, timeout=30, endpoint=EndpointType.SELF_HOSTED
        )

        # Verify response structure
        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(latency, float)
        assert latency >= 0
        assert isinstance(credits, int)
        assert credits >= 0


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.unit
@pytest.mark.fast
class TestExecuteQueryBoth:
    """Test execute_query_both() method for dual endpoint execution."""

    def test_execute_query_both_success_simulated(self):
        """Test execute_query_both() when both endpoints succeed in simulation mode."""
        config = BenchmarkConfig()

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')
        query = "What are the contraindications for warfarin?"

        # Execute against both endpoints (simulated)
        result = runner.execute_query_both(query, timeout=30)

        # Verify result structure
        assert isinstance(result, dict)
        assert "cloud" in result
        assert "self_hosted" in result
        assert "comparison" in result

        # Verify cloud section
        assert "response" in result["cloud"]
        assert "latency_ms" in result["cloud"]
        assert "credits_used" in result["cloud"]
        assert "succeeded" in result["cloud"]
        assert result["cloud"]["succeeded"] is True

        # Verify self_hosted section
        assert "response" in result["self_hosted"]
        assert "latency_ms" in result["self_hosted"]
        assert "credits_used" in result["self_hosted"]
        assert "succeeded" in result["self_hosted"]
        assert result["self_hosted"]["succeeded"] is True

        # Verify comparison section
        assert "latency_diff_ms" in result["comparison"]
        assert "latency_ratio" in result["comparison"]
        assert "cost_diff" in result["comparison"]
        assert "both_succeeded" in result["comparison"]
        assert "cloud_faster" in result["comparison"]
        assert "self_hosted_cheaper" in result["comparison"]
        assert result["comparison"]["both_succeeded"] is True

    def test_execute_query_both_latency_comparison_simulated(self):
        """Test execute_query_both() latency comparison calculations in simulation mode."""
        config = BenchmarkConfig()

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')
        query = "Test query"

        result = runner.execute_query_both(query, timeout=30)

        # Verify latency calculations
        cloud_latency = result["cloud"]["latency_ms"]
        sh_latency = result["self_hosted"]["latency_ms"]
        latency_diff = result["comparison"]["latency_diff_ms"]
        latency_ratio = result["comparison"]["latency_ratio"]

        # Verify calculations are correct
        assert latency_diff == sh_latency - cloud_latency
        if cloud_latency > 0:
            assert latency_ratio == sh_latency / cloud_latency

        # Verify cloud_faster flag
        assert result["comparison"]["cloud_faster"] == (cloud_latency < sh_latency)


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.unit
@pytest.mark.fast
class TestRunBenchmarkModeBoth:
    """Test run_benchmark() with mode='both'."""

    def test_run_benchmark_mode_both_structure_simulated(self, sample_benchmark_data):
        """Test run_benchmark() with mode='both' produces correct result structure in simulation mode."""
        config = BenchmarkConfig()

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')

        # Mock the loader
        with patch.object(runner.loader, 'load_benchmark', return_value=sample_benchmark_data):
            result = runner.run_benchmark("drug_interactions", version=1)

        # Verify metadata
        assert result["metadata"]["mode"] == "both"
        assert "cloud_successful_queries" in result["metadata"]
        assert "self_hosted_successful_queries" in result["metadata"]

        # Verify metrics structure
        assert "metrics" in result
        assert "cloud" in result["metrics"]
        assert "self_hosted" in result["metrics"]
        assert "comparison" in result["metrics"]

        # Verify cloud metrics
        cloud_metrics = result["metrics"]["cloud"]
        assert "average_accuracy" in cloud_metrics
        assert "average_overall_score" in cloud_metrics
        assert "average_latency_ms" in cloud_metrics
        assert "average_credits_per_query" in cloud_metrics
        assert "total_credits" in cloud_metrics

        # Verify self_hosted metrics
        sh_metrics = result["metrics"]["self_hosted"]
        assert "average_accuracy" in sh_metrics
        assert "average_overall_score" in sh_metrics
        assert "average_latency_ms" in sh_metrics
        assert "average_credits_per_query" in sh_metrics
        assert "total_credits" in sh_metrics

        # Verify comparison metrics
        comparison = result["metrics"]["comparison"]
        assert "accuracy_diff" in comparison
        assert "latency_diff_ms" in comparison
        assert "cost_diff" in comparison
        assert "cloud_faster" in comparison
        assert "self_hosted_cheaper" in comparison

        # Verify query results
        assert "query_results" in result
        for query_result in result["query_results"]:
            assert query_result["mode"] == "both"
            assert "cloud" in query_result
            assert "self_hosted" in query_result
            assert "comparison" in query_result


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestCompareAgainstBaselinesModeBoth:
    """Test compare_against_baselines() for mode='both'."""

    def test_compare_against_baselines_both_modes(self):
        """Test baseline comparison for mode='both' result."""
        config = BenchmarkConfig()

        from scripts.run_pharmaceutical_benchmarks import BenchmarkRunner

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')

        # Create mock benchmark result for mode='both'
        benchmark_result = {
            "metadata": {
                "category": "drug_interactions",
                "version": 1,
                "mode": "both"
            },
            "metrics": {
                "cloud": {
                    "average_accuracy": 0.80,  # 5.9% drop from baseline
                    "average_credits_per_query": 15.0,  # 20% increase from baseline
                    "average_latency_ms": 500.0  # 11% increase from baseline
                },
                "self_hosted": {
                    "average_accuracy": 0.78,  # 4.9% drop from baseline (no regression)
                    "average_credits_per_query": 0.0,  # Same as baseline
                    "average_latency_ms": 1300.0  # 53% increase from baseline (regression)
                }
            }
        }

        # Baselines from drug_interactions_v1.json
        baselines = {
            "cloud": {
                "average_accuracy": 0.85,
                "average_cost_per_query": 12.5,
                "average_latency_ms": 450.0
            },
            "self_hosted": {
                "average_accuracy": 0.82,
                "average_cost_per_query": 0.0,
                "average_latency_ms": 850.0
            },
            "regression_thresholds": {
                "accuracy_drop_percent": 5,
                "cost_increase_percent": 20,
                "latency_increase_percent": 50
            }
        }

        # Compare against baselines
        comparison = runner.compare_against_baselines(benchmark_result, baselines)

        # Verify structure
        assert comparison["mode"] == "both"
        assert "cloud" in comparison
        assert "self_hosted" in comparison
        assert "overall_has_regressions" in comparison

        # Verify cloud regressions (should have accuracy regression)
        cloud = comparison["cloud"]
        assert "regressions" in cloud
        assert "has_regressions" in cloud
        assert cloud["has_regressions"] is True
        assert "accuracy_regression" in cloud["regressions"]

        # Verify self_hosted regressions (should have latency regression)
        sh = comparison["self_hosted"]
        assert "regressions" in sh
        assert "has_regressions" in sh
        assert sh["has_regressions"] is True
        assert "latency_regression" in sh["regressions"]

        # Overall should have regressions
        assert comparison["overall_has_regressions"] is True

    def test_compare_against_baselines_no_regressions(self):
        """Test baseline comparison when no regressions detected."""
        config = BenchmarkConfig()

        from scripts.run_pharmaceutical_benchmarks import BenchmarkRunner

        runner = BenchmarkRunner(config, use_real_clients=False, mode='both')

        # Create mock benchmark result with improvements
        benchmark_result = {
            "metadata": {
                "category": "drug_interactions",
                "version": 1,
                "mode": "both"
            },
            "metrics": {
                "cloud": {
                    "average_accuracy": 0.87,  # Improved
                    "average_credits_per_query": 12.0,  # Improved
                    "average_latency_ms": 400.0  # Improved
                },
                "self_hosted": {
                    "average_accuracy": 0.84,  # Improved
                    "average_credits_per_query": 0.0,  # Same
                    "average_latency_ms": 800.0  # Improved
                }
            }
        }

        baselines = {
            "cloud": {
                "average_accuracy": 0.85,
                "average_cost_per_query": 12.5,
                "average_latency_ms": 450.0
            },
            "self_hosted": {
                "average_accuracy": 0.82,
                "average_cost_per_query": 0.0,
                "average_latency_ms": 850.0
            }
        }

        comparison = runner.compare_against_baselines(benchmark_result, baselines)

        # No regressions should be detected
        assert comparison["cloud"]["has_regressions"] is False
        assert len(comparison["cloud"]["regressions"]) == 0
        assert comparison["self_hosted"]["has_regressions"] is False
        assert len(comparison["self_hosted"]["regressions"]) == 0
        assert comparison["overall_has_regressions"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
