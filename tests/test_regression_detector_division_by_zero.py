"""
Test Regression Detector Division-by-Zero Fix

Validates Comment 1 fix: regression detector should handle zero baseline costs
without division-by-zero errors (self-hosted baselines have cost_per_query = 0).
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.pharmaceutical_benchmark_tracker import RegressionDetector


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.fast
class TestRegressionDetectorDivisionByZero:
    """Test division-by-zero protection in regression detection."""

    def test_cost_regression_with_zero_baseline(self):
        """Test that cost regression handles zero baseline without division-by-zero.

        Self-hosted baselines have cost_per_query = 0.0 (no credits).
        Should skip percentage calculation instead of dividing by zero.
        """
        detector = RegressionDetector(
            accuracy_threshold=0.05,
            cost_threshold=0.20,
            latency_threshold=0.50
        )

        # Self-hosted baseline (cost = 0) vs cloud current (cost > 0)
        baseline_metrics = {
            "accuracy": 0.85,
            "cost_per_query": 0.0,  # Self-hosted: free
            "latency_ms": 850.0
        }

        current_metrics = {
            "accuracy": 0.86,
            "cost_per_query": 12.5,  # Cloud: paid
            "latency_ms": 450.0
        }

        # Should NOT raise ZeroDivisionError
        regressions = detector.detect_regression(current_metrics, baseline_metrics)

        # Should NOT flag cost regression (can't calculate % increase from 0)
        cost_regressions = [r for r in regressions if r["type"] == "cost_regression"]
        assert len(cost_regressions) == 0, "Should skip cost regression when baseline is 0"

        # Should still detect other regressions normally
        assert isinstance(regressions, list)

    def test_cost_regression_with_nonzero_baseline(self):
        """Test that cost regression works normally with non-zero baseline."""
        detector = RegressionDetector(
            accuracy_threshold=0.05,
            cost_threshold=0.20,
            latency_threshold=0.50
        )

        # Cloud baseline (cost > 0) vs higher cloud current (cost increase > 20%)
        baseline_metrics = {
            "accuracy": 0.85,
            "cost_per_query": 10.0,
            "latency_ms": 450.0
        }

        current_metrics = {
            "accuracy": 0.86,
            "cost_per_query": 15.0,  # 50% increase (exceeds 20% threshold)
            "latency_ms": 450.0
        }

        regressions = detector.detect_regression(current_metrics, baseline_metrics)

        # Should flag cost regression
        cost_regressions = [r for r in regressions if r["type"] == "cost_regression"]
        assert len(cost_regressions) == 1, "Should detect cost regression when baseline > 0"
        assert cost_regressions[0]["increase_percent"] == 50.0
        assert cost_regressions[0]["baseline"] == 10.0
        assert cost_regressions[0]["current"] == 15.0

    def test_latency_regression_with_zero_baseline(self):
        """Test that latency regression handles zero baseline (edge case)."""
        detector = RegressionDetector(
            accuracy_threshold=0.05,
            cost_threshold=0.20,
            latency_threshold=0.50
        )

        # Zero latency baseline (unlikely but possible edge case)
        baseline_metrics = {
            "accuracy": 0.85,
            "cost_per_query": 10.0,
            "latency_ms": 0.0  # Edge case: instant response
        }

        current_metrics = {
            "accuracy": 0.86,
            "cost_per_query": 10.0,
            "latency_ms": 450.0
        }

        # Should NOT raise ZeroDivisionError
        regressions = detector.detect_regression(current_metrics, baseline_metrics)

        # Should NOT flag latency regression (can't calculate % increase from 0)
        latency_regressions = [r for r in regressions if r["type"] == "latency_regression"]
        assert len(latency_regressions) == 0, "Should skip latency regression when baseline is 0"

    def test_latency_regression_with_nonzero_baseline(self):
        """Test that latency regression works normally with non-zero baseline."""
        detector = RegressionDetector(
            accuracy_threshold=0.05,
            cost_threshold=0.20,
            latency_threshold=0.50
        )

        # Normal latency baseline vs doubled latency (100% increase > 50% threshold)
        baseline_metrics = {
            "accuracy": 0.85,
            "cost_per_query": 10.0,
            "latency_ms": 450.0
        }

        current_metrics = {
            "accuracy": 0.86,
            "cost_per_query": 10.0,
            "latency_ms": 900.0  # 100% increase (exceeds 50% threshold)
        }

        regressions = detector.detect_regression(current_metrics, baseline_metrics)

        # Should flag latency regression
        latency_regressions = [r for r in regressions if r["type"] == "latency_regression"]
        assert len(latency_regressions) == 1, "Should detect latency regression when baseline > 0"
        assert latency_regressions[0]["increase_percent"] == 100.0
        assert latency_regressions[0]["baseline"] == 450.0
        assert latency_regressions[0]["current"] == 900.0

    def test_self_hosted_to_cloud_comparison(self):
        """Test realistic scenario: comparing self-hosted baseline to cloud current.

        This is the exact scenario from Comment 1:
        - Self-hosted baseline: cost_per_query = 0.0
        - Cloud current: cost_per_query = 12.5
        - Should NOT crash with division-by-zero
        """
        detector = RegressionDetector(
            accuracy_threshold=0.05,
            cost_threshold=0.20,
            latency_threshold=0.50
        )

        # From sample_baseline_metadata fixture
        baseline_metrics = {
            "accuracy": 0.82,
            "cost_per_query": 0.0,  # self_hosted baseline
            "latency_ms": 850.0
        }

        # Cloud current metrics
        current_metrics = {
            "accuracy": 0.85,
            "cost_per_query": 12.5,  # cloud current
            "latency_ms": 450.0
        }

        # Should NOT raise ZeroDivisionError
        regressions = detector.detect_regression(current_metrics, baseline_metrics)

        # Verify no cost regression flagged (can't measure % from free tier)
        cost_regressions = [r for r in regressions if r["type"] == "cost_regression"]
        assert len(cost_regressions) == 0

        # Latency improved (450 < 850), so no latency regression
        latency_regressions = [r for r in regressions if r["type"] == "latency_regression"]
        assert len(latency_regressions) == 0

        # Accuracy improved (0.85 > 0.82), so no accuracy regression
        accuracy_regressions = [r for r in regressions if r["type"] == "accuracy_regression"]
        assert len(accuracy_regressions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
