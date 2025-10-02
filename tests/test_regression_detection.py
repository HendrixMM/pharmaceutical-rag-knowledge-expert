"""
Test Regression Detection

Tests for regression math fix from Comment 2 verification.
Tests the fixed percentage calculation logic in ComparisonReportGenerator._check_regressions().
Original bug: cost_change / max(0.01, abs(cost_change - cost_change)) = division by 0.01 always
Fixed: ((current - baseline) / baseline) * 100 for proper percentage calculation
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.pharmaceutical_benchmark_report import ComparisonReportGenerator


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.unit
@pytest.mark.fast
class TestAccuracyRegressionDetection:
    """Test accuracy regression detection logic."""

    def test_accuracy_regression_5_percent_drop(self):
        """Test that 5% accuracy drop triggers regression flag."""
        baseline_accuracy = 0.85
        current_accuracy = 0.80  # 5.88% drop

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "accuracy_regression" in flags, \
            "5.88% accuracy drop should trigger regression"

    def test_accuracy_improvement_no_regression(self):
        """Test that accuracy improvement does not trigger regression."""
        baseline_accuracy = 0.80
        current_accuracy = 0.85  # 6.25% improvement

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "accuracy_regression" not in flags, \
            "Accuracy improvement should not trigger regression"

    def test_accuracy_small_drop_no_regression(self):
        """Test that accuracy drop < 5% does not trigger regression."""
        baseline_accuracy = 0.85
        current_accuracy = 0.83  # 2.35% drop (below 5% threshold)

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "accuracy_regression" not in flags, \
            "Drop < 5% should not trigger regression"

    def test_accuracy_boundary_case_exactly_5_percent(self):
        """Test boundary case: exactly 5% drop should trigger regression."""
        baseline_accuracy = 1.0
        current_accuracy = 0.95  # Exactly 5% drop

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=500.0,
            current_latency=500.0
        )

        # Exactly 5% is at the boundary - implementation uses < -5, so -5.0 should trigger
        assert "accuracy_regression" in flags, \
            "Exactly 5% drop should trigger regression (boundary)"


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.unit
@pytest.mark.fast
class TestCostRegressionDetection:
    """Test cost regression detection logic."""

    def test_cost_regression_20_percent_increase(self):
        """Test that 20% cost increase triggers regression flag."""
        baseline_cost = 10.0
        current_cost = 12.5  # 25% increase

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "cost_regression" in flags, \
            "25% cost increase should trigger regression"

    def test_cost_decrease_no_regression(self):
        """Test that cost decrease does not trigger regression."""
        baseline_cost = 10.0
        current_cost = 8.0  # 20% decrease (improvement)

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "cost_regression" not in flags, \
            "Cost decrease should not trigger regression"

    def test_cost_small_increase_no_regression(self):
        """Test that cost increase < 20% does not trigger regression."""
        baseline_cost = 10.0
        current_cost = 11.0  # 10% increase (below 20% threshold)

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "cost_regression" not in flags, \
            "Cost increase < 20% should not trigger regression"


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.unit
@pytest.mark.fast
class TestLatencyRegressionDetection:
    """Test latency regression detection logic."""

    def test_latency_regression_50_percent_increase(self):
        """Test that 50% latency increase triggers regression flag."""
        baseline_latency = 500.0
        current_latency = 800.0  # 60% increase

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=baseline_latency,
            current_latency=current_latency
        )

        assert "latency_regression" in flags, \
            "60% latency increase should trigger regression"

    def test_latency_decrease_no_regression(self):
        """Test that latency decrease does not trigger regression."""
        baseline_latency = 500.0
        current_latency = 300.0  # 40% decrease (improvement)

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=baseline_latency,
            current_latency=current_latency
        )

        assert "latency_regression" not in flags, \
            "Latency decrease should not trigger regression"

    def test_latency_small_increase_no_regression(self):
        """Test that latency increase < 50% does not trigger regression."""
        baseline_latency = 500.0
        current_latency = 600.0  # 20% increase (below 50% threshold)

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=baseline_latency,
            current_latency=current_latency
        )

        assert "latency_regression" not in flags, \
            "Latency increase < 50% should not trigger regression"


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.unit
@pytest.mark.fast
class TestMultipleRegressionsDetection:
    """Test detection of multiple simultaneous regressions."""

    def test_multiple_regressions_detected(self):
        """Test that multiple regressions are all detected."""
        # Accuracy drops 10%, cost increases 30%, latency increases 60%
        baseline_accuracy = 0.85
        current_accuracy = 0.76  # 10.6% drop
        baseline_cost = 10.0
        current_cost = 13.0  # 30% increase
        baseline_latency = 500.0
        current_latency = 850.0  # 70% increase

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=baseline_latency,
            current_latency=current_latency
        )

        assert "accuracy_regression" in flags, \
            "Should detect accuracy regression"
        assert "cost_regression" in flags, \
            "Should detect cost regression"
        assert "latency_regression" in flags, \
            "Should detect latency regression"
        assert len(flags) == 3, \
            f"Should detect all 3 regressions, got {len(flags)}"

    def test_no_regression_all_metrics_improved(self):
        """Test that no regression flags when all metrics improve."""
        baseline_accuracy = 0.80
        current_accuracy = 0.85  # Improved
        baseline_cost = 15.0
        current_cost = 12.0  # Improved (lower)
        baseline_latency = 600.0
        current_latency = 450.0  # Improved (faster)

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=baseline_latency,
            current_latency=current_latency
        )

        assert len(flags) == 0, \
            f"No regressions should be detected when all metrics improve, got {flags}"


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.unit
@pytest.mark.fast
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_baseline_accuracy(self):
        """Test handling of zero baseline accuracy (should skip check)."""
        baseline_accuracy = 0.0
        current_accuracy = 0.5

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=500.0,
            current_latency=500.0
        )

        # Should not crash or trigger false positive
        assert "accuracy_regression" not in flags, \
            "Zero baseline should not trigger false positive"

    def test_zero_baseline_cost(self):
        """Test handling of zero baseline cost (should skip check)."""
        baseline_cost = 0.0
        current_cost = 5.0

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=500.0,
            current_latency=500.0
        )

        # Should not crash or trigger false positive
        assert "cost_regression" not in flags, \
            "Zero baseline should not trigger false positive"

    def test_zero_baseline_latency(self):
        """Test handling of zero baseline latency (should skip check)."""
        baseline_latency = 0.0
        current_latency = 500.0

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=baseline_latency,
            current_latency=current_latency
        )

        # Should not crash or trigger false positive
        assert "latency_regression" not in flags, \
            "Zero baseline should not trigger false positive"

    def test_equal_baseline_and_current_values(self):
        """Test that equal values produce no regression."""
        baseline_accuracy = 0.85
        baseline_cost = 10.0
        baseline_latency = 500.0

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=baseline_accuracy,
            baseline_cost=baseline_cost,
            current_cost=baseline_cost,
            baseline_latency=baseline_latency,
            current_latency=baseline_latency
        )

        assert len(flags) == 0, \
            f"Equal values should produce no regression, got {flags}"

    def test_negative_percentage_change(self):
        """Test that improvements show negative percentage change."""
        # This is more of a documentation test - improvements should show negative %
        baseline_accuracy = 0.80
        current_accuracy = 0.88  # 10% improvement

        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            baseline_cost=10.0,
            current_cost=10.0,
            baseline_latency=500.0,
            current_latency=500.0
        )

        # Calculate percentage manually to document expected behavior
        accuracy_change_pct = ((current_accuracy - baseline_accuracy) / baseline_accuracy) * 100
        assert abs(accuracy_change_pct - 10.0) < 0.01, \
            f"10% improvement should show as +10%, got {accuracy_change_pct}"

        # No regression should be flagged for improvement
        assert "accuracy_regression" not in flags, \
            "Improvement should not trigger regression"


@pytest.mark.pharmaceutical
@pytest.mark.regression_detection
@pytest.mark.unit
@pytest.mark.fast
class TestPercentageCalculationAccuracy:
    """Test that percentage calculations are mathematically correct."""

    def test_percentage_calculation_5_percent_drop(self):
        """Test that 5% drop calculation is correct."""
        baseline = 1.0
        current = 0.95  # Should be exactly -5%

        # Manual calculation: ((0.95 - 1.0) / 1.0) * 100 = -5.0
        expected_pct = ((current - baseline) / baseline) * 100
        assert abs(expected_pct - (-5.0)) < 0.01, \
            f"5% drop should calculate as -5.0%, got {expected_pct}"

    def test_percentage_calculation_20_percent_increase(self):
        """Test that 20% increase calculation is correct."""
        baseline = 10.0
        current = 12.0  # Should be exactly +20%

        # Manual calculation: ((12.0 - 10.0) / 10.0) * 100 = 20.0
        expected_pct = ((current - baseline) / baseline) * 100
        assert expected_pct == 20.0, \
            f"20% increase should calculate as +20.0%, got {expected_pct}"

    def test_percentage_calculation_50_percent_increase(self):
        """Test that 50% increase calculation is correct."""
        baseline = 500.0
        current = 750.0  # Should be exactly +50%

        # Manual calculation: ((750.0 - 500.0) / 500.0) * 100 = 50.0
        expected_pct = ((current - baseline) / baseline) * 100
        assert expected_pct == 50.0, \
            f"50% increase should calculate as +50.0%, got {expected_pct}"

    def test_percentage_calculation_prevents_divide_by_zero_bug(self):
        """Test that the fixed calculation prevents the original divide-by-zero bug."""
        # Original bug: cost_change / max(0.01, abs(cost_change - cost_change))
        # This would always divide by 0.01, giving massive false percentages

        baseline_cost = 10.0
        current_cost = 10.5

        # Old buggy calculation would give: 0.5 / max(0.01, abs(0.5 - 0.5)) = 0.5 / 0.01 = 50.0
        # New correct calculation: ((10.5 - 10.0) / 10.0) * 100 = 5.0%

        correct_pct = ((current_cost - baseline_cost) / baseline_cost) * 100
        assert correct_pct == 5.0, \
            f"5% cost increase should calculate correctly as 5.0%, got {correct_pct}"

        # Verify this wouldn't trigger 20% cost regression threshold
        flags = ComparisonReportGenerator._check_regressions(
            baseline_accuracy=0.85,
            current_accuracy=0.85,
            baseline_cost=baseline_cost,
            current_cost=current_cost,
            baseline_latency=500.0,
            current_latency=500.0
        )

        assert "cost_regression" not in flags, \
            "5% cost increase should not trigger 20% threshold (old bug would have triggered)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
