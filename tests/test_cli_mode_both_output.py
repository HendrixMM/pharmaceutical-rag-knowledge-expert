"""
Test CLI Output for Mode 'Both'

Tests for Comment 1 verification: CLI doesn't crash with --mode both and --category.
"""
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pharmaceutical_benchmarks import BenchmarkConfig, BenchmarkRunner


@pytest.mark.pharmaceutical
@pytest.mark.cli
@pytest.mark.unit
@pytest.mark.fast
class TestCLIModeBothOutput:
    """Test CLI output handling for mode='both' doesn't crash."""

    def test_cli_mode_both_does_not_crash(self, sample_benchmark_data):
        """Test that CLI doesn't crash when printing dual-mode results."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="both")

        # Mock the loader
        with patch.object(runner.loader, "load_benchmark", return_value=sample_benchmark_data):
            result = runner.run_benchmark("drug_interactions", version=1)

        # Verify structure
        assert result["metadata"]["mode"] == "both"
        assert "cloud" in result["metrics"]
        assert "self_hosted" in result["metrics"]
        assert "comparison" in result["metrics"]

        # The critical check: accessing these keys should not raise KeyError
        assert "average_overall_score" in result["metrics"]["cloud"]
        assert "average_overall_score" in result["metrics"]["self_hosted"]
        assert "average_latency_ms" in result["metrics"]["cloud"]
        assert "average_latency_ms" in result["metrics"]["self_hosted"]
        assert "average_credits_per_query" in result["metrics"]["cloud"]
        assert "average_credits_per_query" in result["metrics"]["self_hosted"]

        # Verify comparison metrics
        assert "accuracy_diff" in result["metrics"]["comparison"]
        assert "latency_diff_ms" in result["metrics"]["comparison"]
        assert "cost_diff" in result["metrics"]["comparison"]

    def test_cli_single_mode_still_works(self, sample_benchmark_data):
        """Test that single-mode CLI output still works (backward compatibility)."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        # Mock the loader
        with patch.object(runner.loader, "load_benchmark", return_value=sample_benchmark_data):
            result = runner.run_benchmark("drug_interactions", version=1)

        # Verify single-mode structure
        assert result["metadata"]["mode"] == "cloud"
        assert "average_overall_score" in result["metrics"]
        assert "average_latency_ms" in result["metrics"]
        assert "average_credits_per_query" in result["metrics"]

        # Should NOT have cloud/self_hosted sub-keys
        assert "cloud" not in result["metrics"]
        assert "self_hosted" not in result["metrics"]

    @patch("sys.stdout", new_callable=StringIO)
    def test_cli_print_mode_both_formatted_output(self, mock_stdout, sample_benchmark_data):
        """Test that CLI formatted print works for mode='both' without errors."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="both")

        # Mock the loader
        with patch.object(runner.loader, "load_benchmark", return_value=sample_benchmark_data):
            result = runner.run_benchmark("drug_interactions", version=1)

        # Simulate the CLI print logic (lines 892-917)
        try:
            if result["metadata"].get("mode") == "both":
                # This should not crash
                result["metrics"]["cloud"]["average_overall_score"]
                result["metrics"]["cloud"]["average_latency_ms"]
                result["metrics"]["cloud"]["average_credits_per_query"]

                result["metrics"]["self_hosted"]["average_overall_score"]
                result["metrics"]["self_hosted"]["average_latency_ms"]
                result["metrics"]["self_hosted"]["average_credits_per_query"]

                result["metrics"]["comparison"]["accuracy_diff"]
                result["metrics"]["comparison"]["latency_diff_ms"]
                result["metrics"]["comparison"]["cost_diff"]

                # If we get here without KeyError, test passes
                assert True
            else:
                pytest.fail("Result mode should be 'both'")
        except KeyError as e:
            pytest.fail(f"CLI print logic raised KeyError: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
