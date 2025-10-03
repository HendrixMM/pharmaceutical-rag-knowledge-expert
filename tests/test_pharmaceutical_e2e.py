"""
Test Pharmaceutical E2E Workflows

End-to-end tests for complete benchmark execution workflows.
Mini-Phase B implementation targeting high-value coverage gaps.
"""
import sys
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pharmaceutical_benchmark_report import ComparisonReportGenerator
from scripts.run_pharmaceutical_benchmarks import BenchmarkConfig, BenchmarkRunner


@pytest.mark.pharmaceutical
@pytest.mark.e2e
@pytest.mark.fast
class TestBenchmarkExecutionE2E:
    """Test complete benchmark execution workflow (lines 400-488)."""

    def test_full_benchmark_run_simulated(self, sample_benchmark_data):
        """Test full benchmark run: load → execute → evaluate → aggregate."""
        # Arrange: Setup runner with simulated execution
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        # Mock the loader to return fixture data
        with patch.object(runner.loader, "load_benchmark", return_value=sample_benchmark_data):
            # Act: Run complete benchmark
            result = runner.run_benchmark("drug_interactions", version=1)

        # Assert: Verify result structure
        assert result is not None, "Benchmark result should not be None"

        # Check metadata
        assert "metadata" in result
        assert result["metadata"]["category"] == "drug_interactions"
        assert result["metadata"]["version"] == 1
        assert "run_timestamp" in result["metadata"]
        assert "total_queries" in result["metadata"]
        assert "successful_queries" in result["metadata"]
        assert "failed_queries" in result["metadata"]

        # Verify query count matches fixture
        expected_queries = len(sample_benchmark_data.get("queries", []))
        assert result["metadata"]["total_queries"] == expected_queries, f"Should process {expected_queries} queries"

        # Check metrics
        assert "metrics" in result
        assert "average_accuracy" in result["metrics"]
        assert "average_overall_score" in result["metrics"]
        assert "average_latency_ms" in result["metrics"]
        assert "average_credits_per_query" in result["metrics"]
        assert "total_credits" in result["metrics"]

        # Verify metric ranges
        assert 0 <= result["metrics"]["average_accuracy"] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= result["metrics"]["average_overall_score"] <= 1, "Overall score should be between 0 and 1"
        assert result["metrics"]["average_latency_ms"] >= 0, "Latency should be non-negative"
        assert result["metrics"]["average_credits_per_query"] >= 0, "Credits should be non-negative"

        # Check query results
        assert "query_results" in result
        assert len(result["query_results"]) == expected_queries, f"Should have {expected_queries} query results"

        # Verify each query result has required fields
        for query_result in result["query_results"]:
            assert "query_id" in query_result
            assert "query" in query_result

            # Should have either scores (success) or error (failure)
            has_scores = "scores" in query_result
            has_error = "error" in query_result
            assert has_scores or has_error, "Query result should have either scores or error"

            if has_scores:
                assert "response" in query_result
                assert "latency_ms" in query_result
                assert "credits_used" in query_result
                assert "timestamp" in query_result

                # Verify score structure
                scores = query_result["scores"]
                assert "accuracy" in scores
                assert "overall" in scores

        # Verify result was stored in runner
        assert len(runner.results) > 0, "Runner should store results"
        assert runner.results[-1] == result, "Last result should match returned result"

    def test_full_benchmark_run_handles_query_failures(self):
        """Test that benchmark run handles individual query failures gracefully."""
        # Arrange: Setup runner with simulated execution
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        # Create benchmark data with multiple queries
        multi_query_data = {
            "metadata": {
                "category": "drug_interactions",
                "version": 1,
                "baselines": {"cloud": {}, "self_hosted": {}, "regression_thresholds": {}},
            },
            "queries": [
                {
                    "id": "test_001",
                    "query": "Test query 1",
                    "expected_content": ["content1"],
                    "evaluation_criteria": {},
                },
                {
                    "id": "test_002",
                    "query": "Test query 2",
                    "expected_content": ["content2"],
                    "evaluation_criteria": {},
                },
            ],
        }

        # Mock execute_query to fail on second query
        original_execute = runner.execute_query
        call_count = [0]

        def execute_with_failure(query, timeout=30):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated API failure")
            return original_execute(query, timeout)

        runner.execute_query = execute_with_failure

        with patch.object(runner.loader, "load_benchmark", return_value=multi_query_data):
            # Act: Run benchmark
            result = runner.run_benchmark("drug_interactions", version=1)

        # Assert: Should have partial success
        assert result["metadata"]["failed_queries"] == 1, "Should have exactly one failed query"
        assert result["metadata"]["successful_queries"] == 1, "Should have exactly one successful query"
        assert result["metadata"]["total_queries"] == 2, "Should have 2 total queries"

        # Check that error is recorded
        error_results = [r for r in result["query_results"] if "error" in r]
        assert len(error_results) == 1, "Should have 1 error result"
        assert "Simulated API failure" in error_results[0]["error"]


@pytest.mark.pharmaceutical
@pytest.mark.e2e
@pytest.mark.fast
class TestCLIExecutionE2E:
    """Test CLI interface workflow (lines 506-586)."""

    def test_cli_execution_simulation_single_category(self, capsys):
        """Test CLI execution: argparse → runner init → execution → output."""
        # Arrange: Mock sys.argv for CLI arguments
        test_args = [
            "run_pharmaceutical_benchmarks.py",
            "--category",
            "drug_interactions",
            "--version",
            "1",
            "--simulate",
            "--mode",
            "cloud",
        ]

        with patch("sys.argv", test_args), patch("scripts.run_pharmaceutical_benchmarks.BenchmarkRunner") as MockRunner:
            # Setup mock runner
            mock_runner_instance = Mock()
            mock_result = {
                "metadata": {"category": "drug_interactions", "version": 1},
                "metrics": {
                    "average_overall_score": 0.85,
                    "average_latency_ms": 450.0,
                    "average_credits_per_query": 3.5,
                },
            }
            mock_runner_instance.run_benchmark.return_value = mock_result
            mock_runner_instance.loader.list_available_benchmarks.return_value = []
            MockRunner.return_value = mock_runner_instance

            # Act: Import and run main
            from scripts.run_pharmaceutical_benchmarks import main

            exit_code = main()

            # Assert: Verify successful return code
            assert exit_code == 0, "Should return 0 for success"

            # Verify runner was initialized correctly
            MockRunner.assert_called_once()
            call_kwargs = MockRunner.call_args[1]
            assert call_kwargs["use_real_clients"] is False, "Should use simulation mode"
            assert call_kwargs["mode"] == "cloud", "Should use cloud mode"

            # Verify run_benchmark was called with correct arguments
            mock_runner_instance.run_benchmark.assert_called_once_with("drug_interactions", 1)

            # Verify output was printed (use pytest's capsys fixture)
            captured = capsys.readouterr()
            assert "drug_interactions" in captured.out
            assert "0.850" in captured.out or "0.85" in captured.out
            assert "450" in captured.out
            assert "3.5" in captured.out or "3.50" in captured.out

    def test_cli_execution_all_benchmarks(self):
        """Test CLI execution with no category specified (runs all)."""
        # Arrange: Mock sys.argv without category
        test_args = ["run_pharmaceutical_benchmarks.py", "--simulate", "--mode", "cloud"]

        with patch("sys.argv", test_args), patch("scripts.run_pharmaceutical_benchmarks.BenchmarkRunner") as MockRunner:
            # Setup mock runner
            mock_runner_instance = Mock()
            mock_runner_instance.loader.list_available_benchmarks.return_value = [
                ("drug_interactions", 1),
                ("pharmacokinetics", 1),
            ]
            mock_runner_instance.run_benchmark.return_value = {
                "metadata": {"category": "test", "version": 1},
                "metrics": {"average_overall_score": 0.8},
            }
            MockRunner.return_value = mock_runner_instance

            # Act: Run main
            from scripts.run_pharmaceutical_benchmarks import main

            exit_code = main()

            # Assert: Should return success
            assert exit_code == 0, "Should return 0 for success"

            # Assert: Should run both benchmarks
            assert mock_runner_instance.run_benchmark.call_count == 2, "Should run 2 benchmarks"

            # Verify both categories were run
            calls = mock_runner_instance.run_benchmark.call_args_list
            categories = [call[0][0] for call in calls]
            assert "drug_interactions" in categories
            assert "pharmacokinetics" in categories

    def test_cli_save_results_option(self, tmp_path, capsys):
        """Test CLI --save-results option."""
        # Arrange: Mock sys.argv with save option
        test_args = [
            "run_pharmaceutical_benchmarks.py",
            "--category",
            "drug_interactions",
            "--simulate",
            "--save-results",
            "--output",
            str(tmp_path),
        ]

        with patch("sys.argv", test_args), patch("scripts.run_pharmaceutical_benchmarks.BenchmarkRunner") as MockRunner:
            # Setup mock runner with complete result structure
            mock_runner_instance = Mock()
            mock_runner_instance.run_benchmark.return_value = {
                "metadata": {"category": "drug_interactions", "version": 1},
                "metrics": {
                    "average_overall_score": 0.85,
                    "average_latency_ms": 450.0,
                    "average_credits_per_query": 3.5,
                },
            }
            mock_runner_instance.save_results.return_value = str(tmp_path / "results.json")
            MockRunner.return_value = mock_runner_instance

            # Act: Run main
            from scripts.run_pharmaceutical_benchmarks import main

            exit_code = main()

            # Assert: Should return success
            assert exit_code == 0, "Should return 0 for success"

            # Assert: Should call save_results
            mock_runner_instance.save_results.assert_called_once_with(str(tmp_path))


@pytest.mark.pharmaceutical
@pytest.mark.e2e
@pytest.mark.fast
class TestRegressionDetectionWorkflowE2E:
    """Test regression detection workflow (lines 186-231)."""

    def test_regression_detection_workflow_with_regressions(self):
        """Test complete regression detection: baseline → current → comparison → flags."""
        # Arrange: Create baseline and current results with intentional regressions
        baseline_results = [
            {
                "metadata": {"category": "drug_interactions"},
                "metrics": {"average_accuracy": 0.85, "average_credits_per_query": 10.0, "average_latency_ms": 500.0},
            },
            {
                "metadata": {"category": "pharmacokinetics"},
                "metrics": {"average_accuracy": 0.80, "average_credits_per_query": 8.0, "average_latency_ms": 400.0},
            },
        ]

        current_results = [
            {
                "metadata": {"category": "drug_interactions"},
                "metrics": {
                    "average_accuracy": 0.78,  # 8.2% drop - triggers accuracy regression
                    "average_credits_per_query": 13.0,  # 30% increase - triggers cost regression
                    "average_latency_ms": 800.0,  # 60% increase - triggers latency regression
                },
            },
            {
                "metadata": {"category": "pharmacokinetics"},
                "metrics": {
                    "average_accuracy": 0.82,  # Improved
                    "average_credits_per_query": 7.5,  # Improved
                    "average_latency_ms": 380.0,  # Improved
                },
            },
        ]

        # Act: Generate comparison
        generator = ComparisonReportGenerator(baseline_results, current_results)
        comparison = generator.generate_comparison()

        # Assert: Verify comparison structure
        assert isinstance(comparison, dict)
        assert "drug_interactions" in comparison
        assert "pharmacokinetics" in comparison

        # Check drug_interactions (should have regressions)
        di_comparison = comparison["drug_interactions"]

        assert "baseline" in di_comparison
        assert di_comparison["baseline"]["accuracy"] == 0.85
        assert di_comparison["baseline"]["cost"] == 10.0
        assert di_comparison["baseline"]["latency"] == 500.0

        assert "current" in di_comparison
        assert di_comparison["current"]["accuracy"] == 0.78
        assert di_comparison["current"]["cost"] == 13.0
        assert di_comparison["current"]["latency"] == 800.0

        assert "changes" in di_comparison
        assert di_comparison["changes"]["accuracy_change"] == pytest.approx(-0.07, abs=0.01)
        assert di_comparison["changes"]["cost_change"] == pytest.approx(3.0, abs=0.01)
        assert di_comparison["changes"]["latency_change"] == pytest.approx(300.0, abs=0.1)

        assert "regression_flags" in di_comparison
        flags = di_comparison["regression_flags"]
        assert "accuracy_regression" in flags, "8.2% accuracy drop should trigger regression"
        assert "cost_regression" in flags, "30% cost increase should trigger regression"
        assert "latency_regression" in flags, "60% latency increase should trigger regression"
        assert len(flags) == 3, "Should detect all 3 regressions"

        # Check pharmacokinetics (should have no regressions)
        pk_comparison = comparison["pharmacokinetics"]

        assert pk_comparison["current"]["accuracy"] > pk_comparison["baseline"]["accuracy"], "Accuracy should improve"
        assert pk_comparison["current"]["cost"] < pk_comparison["baseline"]["cost"], "Cost should improve"
        assert pk_comparison["current"]["latency"] < pk_comparison["baseline"]["latency"], "Latency should improve"

        pk_flags = pk_comparison["regression_flags"]
        assert len(pk_flags) == 0, f"Should have no regressions for improved metrics, got {pk_flags}"

    def test_regression_detection_workflow_missing_categories(self):
        """Test regression detection handles missing categories gracefully."""
        # Arrange: Baseline has category not in current
        baseline_results = [
            {
                "metadata": {"category": "drug_interactions"},
                "metrics": {"average_accuracy": 0.85, "average_credits_per_query": 10.0, "average_latency_ms": 500.0},
            },
            {
                "metadata": {"category": "old_category"},
                "metrics": {"average_accuracy": 0.75, "average_credits_per_query": 5.0, "average_latency_ms": 300.0},
            },
        ]

        current_results = [
            {
                "metadata": {"category": "drug_interactions"},
                "metrics": {"average_accuracy": 0.85, "average_credits_per_query": 10.0, "average_latency_ms": 500.0},
            },
            {
                "metadata": {"category": "new_category"},
                "metrics": {"average_accuracy": 0.80, "average_credits_per_query": 8.0, "average_latency_ms": 400.0},
            },
        ]

        # Act: Generate comparison
        generator = ComparisonReportGenerator(baseline_results, current_results)
        comparison = generator.generate_comparison()

        # Assert: Should handle gracefully
        assert "drug_interactions" in comparison, "Should compare matching category"

        # Missing categories should not crash or appear in comparison
        # (they're skipped if metrics missing)
        assert isinstance(comparison, dict)

    def test_regression_detection_workflow_boundary_thresholds(self):
        """Test regression detection at threshold boundaries."""
        # Arrange: Create results that exceed regression thresholds
        # Note: Implementation uses strict inequality (> not >=), so we need values ABOVE thresholds
        baseline_results = [
            {
                "metadata": {"category": "test_category"},
                "metrics": {"average_accuracy": 1.0, "average_credits_per_query": 10.0, "average_latency_ms": 500.0},
            }
        ]

        current_results = [
            {
                "metadata": {"category": "test_category"},
                "metrics": {
                    "average_accuracy": 0.94,  # 6% drop (exceeds 5% threshold)
                    "average_credits_per_query": 12.5,  # 25% increase (exceeds 20% threshold)
                    "average_latency_ms": 800.0,  # 60% increase (exceeds 50% threshold)
                },
            }
        ]

        # Act: Generate comparison
        generator = ComparisonReportGenerator(baseline_results, current_results)
        comparison = generator.generate_comparison()

        # Assert: Values exceeding thresholds should trigger regressions
        flags = comparison["test_category"]["regression_flags"]
        assert "accuracy_regression" in flags, "6% drop should trigger (threshold < -5)"
        assert "cost_regression" in flags, "25% increase should trigger (threshold > 20)"
        assert "latency_regression" in flags, "60% increase should trigger (threshold > 50)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
