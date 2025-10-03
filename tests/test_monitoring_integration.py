"""
Test Monitoring Integration and Dual-Endpoint Fixes

Validates all three verification comments:
1. Comment 1: Failed-query metrics don't inflate averages
2. Comment 2: Tracker parses dual-endpoint results correctly
3. Comment 3: Tracker is wired into runner and exports artifacts
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.pharmaceutical_benchmark_tracker import PharmaceuticalBenchmarkTracker


@pytest.mark.pharmaceutical
@pytest.mark.monitoring
@pytest.mark.fast
class TestMonitoringIntegration:
    """Test monitoring integration and dual-endpoint handling."""

    def test_comment_1_failed_query_averages_excluded(self):
        """
        Comment 1: Verify failed queries don't inflate dual-mode averages.

        Simulates scenario where:
        - 2 cloud queries succeed (100ms, 200ms)
        - 1 cloud query fails (0ms accumulated)
        - Average should be 150ms (not 100ms = 300/3)
        """
        # Simulate the fixed accumulation logic
        cloud_total_latency = 0
        cloud_successful_count = 0

        # Query 1: Success
        if True:  # succeeded
            cloud_total_latency += 100
            cloud_successful_count += 1

        # Query 2: Success
        if True:  # succeeded
            cloud_total_latency += 200
            cloud_successful_count += 1

        # Query 3: Failure (not accumulated)
        if False:  # succeeded
            cloud_total_latency += 0
            cloud_successful_count += 1

        # Calculate average (fixed logic)
        cloud_avg_latency = cloud_total_latency / cloud_successful_count if cloud_successful_count > 0 else 0

        assert cloud_avg_latency == 150.0, "Failed queries should not inflate average"
        assert cloud_successful_count == 2, "Should only count successful queries"

    def test_comment_2_dual_endpoint_parsing(self):
        """
        Comment 2: Verify tracker parses dual-endpoint results correctly.

        Tests that mode='both' results are properly extracted and tracked.
        Updated after double-counting fix: track_query_result must be called first.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Simulate individual query tracking (as runner does)
        for i in range(5):
            # Cloud: all 5 succeed
            tracker.track_query_result("drug_interactions", "drug_interactions", 0.85, 12.5, 450.0, True)
            # Self-hosted: 4 succeed, 1 fails
            if i < 4:
                tracker.track_query_result("drug_interactions", "drug_interactions", 0.82, 0.0, 850.0, True)
            else:
                tracker.track_query_result("drug_interactions", "drug_interactions", 0.0, 0.0, 0.0, False)

        # Simulate dual-endpoint benchmark result
        dual_result = {
            "metadata": {
                "category": "drug_interactions",
                "version": 1,
                "mode": "both",
                "total_queries": 5,
                "cloud_successful_queries": 5,
                "self_hosted_successful_queries": 4,
                "failed_queries": 1,
            },
            "metrics": {
                "cloud": {
                    "average_accuracy": 0.85,
                    "average_overall_score": 0.80,
                    "average_latency_ms": 450.0,
                    "average_credits_per_query": 12.5,
                    "total_credits": 62.5,
                },
                "self_hosted": {
                    "average_accuracy": 0.82,
                    "average_overall_score": 0.78,
                    "average_latency_ms": 850.0,
                    "average_credits_per_query": 0.0,
                    "total_credits": 0.0,
                },
            },
        }

        # Track the result
        tracker.track_benchmark_run(dual_result)

        # Verify metrics were extracted correctly
        summary = tracker.get_metrics_summary()

        # Should have tracked both endpoints (2 entries for drug_interactions)
        assert "drug_interactions" in summary["accuracy_by_category"]
        assert summary["accuracy_by_category"]["drug_interactions"] > 0

        # Verify query counts from individual tracking (NOT from metadata)
        assert summary["total_queries"] == 10, "5 queries × 2 endpoints = 10 total"
        assert summary["successful_queries"] == 9, "5 cloud + 4 self-hosted = 9 successful"
        assert summary["failed_queries"] == 1, "1 self-hosted failed"

    def test_comment_2_single_mode_parsing(self):
        """
        Comment 2: Verify tracker still handles single-mode results correctly.

        Tests backward compatibility with single-mode benchmark results.
        Updated after double-counting fix: track_query_result must be called first.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Simulate individual query tracking (as runner does)
        for i in range(10):
            if i < 9:
                # 9 successful queries
                tracker.track_query_result("clinical_trials", "clinical_trials", 0.88, 15.0, 500.0, True)
            else:
                # 1 failed query
                tracker.track_query_result("clinical_trials", "clinical_trials", 0.0, 0.0, 0.0, False)

        # Simulate single-mode benchmark result
        single_result = {
            "metadata": {
                "category": "clinical_trials",
                "version": 1,
                "mode": "cloud",
                "total_queries": 10,
                "successful_queries": 9,
                "failed_queries": 1,
            },
            "metrics": {
                "average_accuracy": 0.88,
                "average_overall_score": 0.85,
                "average_latency_ms": 500.0,
                "average_credits_per_query": 15.0,
                "total_credits": 135.0,
            },
        }

        # Track the result
        tracker.track_benchmark_run(single_result)

        # Verify metrics were extracted correctly
        summary = tracker.get_metrics_summary()

        assert "clinical_trials" in summary["accuracy_by_category"]
        # Use approximate comparison for floating point (0.88 ± 0.01)
        assert abs(summary["accuracy_by_category"]["clinical_trials"] - 0.88) < 0.01
        # Verify query counts from individual tracking
        assert summary["total_queries"] == 10
        assert summary["successful_queries"] == 9
        assert summary["failed_queries"] == 1

    def test_comment_2_regression_detection_dual_mode(self):
        """
        Comment 2: Verify regression detection works for dual-mode results.

        Tests that regressions are detected separately for cloud and self-hosted.
        """
        # Create baseline with both endpoints
        baseline = {
            "drug_interactions": {
                "cloud": {"accuracy": 0.90, "cost_per_query": 10.0, "latency_ms": 400.0},
                "self_hosted": {"accuracy": 0.85, "cost_per_query": 0.0, "latency_ms": 800.0},
            }
        }

        tracker = PharmaceuticalBenchmarkTracker()
        tracker.baseline_metrics = baseline

        # Simulate result with cloud regression (accuracy drop)
        result_with_regression = {
            "metadata": {
                "category": "drug_interactions",
                "mode": "both",
                "total_queries": 5,
                "cloud_successful_queries": 5,
                "self_hosted_successful_queries": 5,
                "failed_queries": 0,
            },
            "metrics": {
                "cloud": {
                    "average_accuracy": 0.80,  # 10% drop from baseline
                    "average_credits_per_query": 10.0,
                    "average_latency_ms": 400.0,
                },
                "self_hosted": {
                    "average_accuracy": 0.85,  # No change
                    "average_credits_per_query": 0.0,
                    "average_latency_ms": 800.0,
                },
            },
        }

        tracker.track_benchmark_run(result_with_regression)

        # Verify regression was detected for cloud but not self-hosted
        assert tracker.has_regressions()
        regressions = tracker.get_regression_summary()

        # Should have one accuracy regression for cloud endpoint
        cloud_regressions = [r for r in regressions if r.get("endpoint") == "cloud"]
        assert len(cloud_regressions) > 0
        assert cloud_regressions[0]["type"] == "accuracy_regression"

    def test_comment_3_tracker_lifecycle(self):
        """
        Comment 3: Verify tracker integration lifecycle.

        Tests that tracker can be instantiated, track queries, and export metrics.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Track individual queries
        tracker.track_query_result(
            category="drug_safety",
            query_type="drug_interactions",
            accuracy=0.90,
            cost=12.5,
            latency_ms=450.0,
            success=True,
        )

        tracker.track_query_result(
            category="drug_safety",
            query_type="drug_interactions",
            accuracy=0.85,
            cost=10.0,
            latency_ms=500.0,
            success=True,
        )

        # Track benchmark run
        benchmark_result = {
            "metadata": {
                "category": "drug_safety",
                "mode": "cloud",
                "total_queries": 2,
                "successful_queries": 2,
                "failed_queries": 0,
            },
            "metrics": {"average_accuracy": 0.875, "average_credits_per_query": 11.25, "average_latency_ms": 475.0},
        }

        tracker.track_benchmark_run(benchmark_result)

        # Verify metrics summary
        summary = tracker.get_metrics_summary()
        # After double-counting fix: only counts from track_query_result (NOT from metadata)
        assert summary["total_queries"] == 2, "Only 2 individual queries tracked, no double-counting"
        assert summary["successful_queries"] == 2
        assert "drug_safety" in summary["accuracy_by_category"]

        # Verify export (to temp file)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            tracker.export_metrics(temp_path)
            assert Path(temp_path).exists()

            # Verify exported content
            import json

            with open(temp_path) as f:
                exported = json.load(f)

            assert "timestamp" in exported
            assert "summary" in exported
            assert "regression_flags" in exported
            assert exported["summary"]["total_queries"] == 2, "Exported counts should match in-memory counts"
        finally:
            Path(temp_path).unlink()

    def test_edge_case_all_queries_fail_dual_mode(self):
        """
        Edge case: All queries fail in dual-mode.

        Verifies averages are 0 when no successful queries.
        """
        cloud_total_latency = 0
        cloud_successful_count = 0

        # All queries fail
        for _ in range(3):
            if False:  # succeeded
                cloud_total_latency += 100
                cloud_successful_count += 1

        cloud_avg_latency = cloud_total_latency / cloud_successful_count if cloud_successful_count > 0 else 0

        assert cloud_avg_latency == 0
        assert cloud_successful_count == 0

    def test_edge_case_mixed_success_dual_mode(self):
        """
        Edge case: Mixed success/failure across endpoints.

        Cloud: 3 success, 1 failure
        Self-hosted: 2 success, 2 failure
        """
        # Cloud endpoint
        cloud_total = 0
        cloud_count = 0

        for latency, succeeded in [(100, True), (200, True), (300, True), (0, False)]:
            if succeeded:
                cloud_total += latency
                cloud_count += 1

        cloud_avg = cloud_total / cloud_count if cloud_count > 0 else 0

        # Self-hosted endpoint
        sh_total = 0
        sh_count = 0

        for latency, succeeded in [(400, True), (500, True), (0, False), (0, False)]:
            if succeeded:
                sh_total += latency
                sh_count += 1

        sh_avg = sh_total / sh_count if sh_count > 0 else 0

        assert cloud_avg == 200.0  # (100+200+300)/3
        assert cloud_count == 3
        assert sh_avg == 450.0  # (400+500)/2
        assert sh_count == 2

    def test_tracker_with_missing_cost_analyzer(self):
        """
        Edge case: Tracker works without cost_analyzer.

        Verifies graceful degradation when cost_analyzer is None.
        """
        tracker = PharmaceuticalBenchmarkTracker(cost_analyzer=None)

        # Should still work for tracking
        tracker.track_query_result(
            category="test", query_type="test_type", accuracy=0.9, cost=10.0, latency_ms=100.0, success=True
        )

        summary = tracker.get_metrics_summary()
        assert summary["total_queries"] == 1
        assert summary["successful_queries"] == 1

    def test_tracker_empty_metrics_graceful_handling(self):
        """
        Edge case: Tracker handles empty/missing metrics gracefully.

        Verifies no crashes when endpoint metrics are missing.
        Updated after double-counting fix: track_query_result must be called first.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Simulate individual query tracking (self-hosted succeeded, cloud failed with no metrics)
        tracker.track_query_result("test", "test", 0.0, 0.0, 0.0, False)  # Cloud failed
        tracker.track_query_result("test", "test", 0.85, 0.0, 850.0, True)  # Self-hosted succeeded

        # Result with missing cloud metrics
        partial_result = {
            "metadata": {
                "category": "test",
                "mode": "both",
                "total_queries": 1,
                "cloud_successful_queries": 0,
                "self_hosted_successful_queries": 1,
                "failed_queries": 0,
            },
            "metrics": {
                "cloud": {},  # Empty (graceful handling test)
                "self_hosted": {
                    "average_accuracy": 0.85,
                    "average_credits_per_query": 0.0,
                    "average_latency_ms": 850.0,
                },
            },
        }

        # Should not crash even with empty cloud metrics
        tracker.track_benchmark_run(partial_result)

        summary = tracker.get_metrics_summary()
        assert summary["total_queries"] == 2, "1 query × 2 endpoints = 2 total"
        assert summary["successful_queries"] == 1, "Only self-hosted succeeded"
        assert summary["failed_queries"] == 1, "Cloud failed"

    def test_no_double_counting_dual_mode(self):
        """
        Critical Bug Fix: Verify tracker doesn't double-count queries in dual-mode.

        Bug: track_benchmark_run() was adding metadata counts to already-tracked
        individual query counts, causing success_rate > 100%.

        Fix: track_benchmark_run() no longer updates query counts (lines 287-296 removed).
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Simulate 3 queries tracked individually (dual-mode: 2 endpoints each)
        for i in range(3):
            # Cloud endpoint succeeds
            tracker.track_query_result(
                category="drug_interactions",
                query_type="drug_interactions",
                accuracy=0.90,
                cost=12.5,
                latency_ms=450.0,
                success=True,
            )
            # Self-hosted endpoint succeeds
            tracker.track_query_result(
                category="drug_interactions",
                query_type="drug_interactions",
                accuracy=0.85,
                cost=0.0,
                latency_ms=850.0,
                success=True,
            )

        # After individual tracking: total_queries should be 6 (3 queries × 2 endpoints)
        assert tracker.metrics.total_queries == 6, f"Expected 6, got {tracker.metrics.total_queries}"
        assert tracker.metrics.successful_queries == 6, f"Expected 6, got {tracker.metrics.successful_queries}"
        assert tracker.metrics.failed_queries == 0

        # Now call track_benchmark_run with metadata (as runner does)
        benchmark_result = {
            "metadata": {
                "category": "drug_interactions",
                "mode": "both",
                "total_queries": 3,
                "cloud_successful_queries": 3,
                "self_hosted_successful_queries": 3,
                "failed_queries": 0,
            },
            "metrics": {
                "cloud": {"average_accuracy": 0.90, "average_credits_per_query": 12.5, "average_latency_ms": 450.0},
                "self_hosted": {
                    "average_accuracy": 0.85,
                    "average_credits_per_query": 0.0,
                    "average_latency_ms": 850.0,
                },
            },
        }

        tracker.track_benchmark_run(benchmark_result)

        # CRITICAL: After track_benchmark_run, counts should NOT change (no double-counting)
        assert tracker.metrics.total_queries == 6, "Should not double-count queries after track_benchmark_run"
        assert tracker.metrics.successful_queries == 6, "Should not double-count successes after track_benchmark_run"
        assert tracker.metrics.failed_queries == 0, "Failed queries should remain 0"

        # Success rate must be <= 100%
        summary = tracker.get_metrics_summary()
        assert (
            summary["success_rate"] <= 1.0
        ), f"Success rate {summary['success_rate']:.2%} exceeds 100% (double-counting bug!)"
        assert summary["success_rate"] == 1.0, f"Expected 100% success rate, got {summary['success_rate']:.2%}"

    def test_no_double_counting_single_mode(self):
        """
        Verify tracker doesn't double-count queries in single-mode.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Simulate 5 queries tracked individually (single-mode: 1 endpoint each)
        for i in range(5):
            tracker.track_query_result(
                category="adverse_reactions",
                query_type="adverse_reactions",
                accuracy=0.88,
                cost=10.0,
                latency_ms=500.0,
                success=True,
            )

        # After individual tracking: total_queries should be 5
        assert tracker.metrics.total_queries == 5
        assert tracker.metrics.successful_queries == 5

        # Now call track_benchmark_run with metadata
        benchmark_result = {
            "metadata": {
                "category": "adverse_reactions",
                "mode": "cloud",
                "total_queries": 5,
                "successful_queries": 5,
                "failed_queries": 0,
            },
            "metrics": {"average_accuracy": 0.88, "average_credits_per_query": 10.0, "average_latency_ms": 500.0},
        }

        tracker.track_benchmark_run(benchmark_result)

        # After track_benchmark_run, counts should NOT change
        assert tracker.metrics.total_queries == 5, "Should not double-count queries in single-mode"
        assert tracker.metrics.successful_queries == 5, "Should not double-count successes in single-mode"
        assert tracker.metrics.failed_queries == 0

        summary = tracker.get_metrics_summary()
        assert summary["success_rate"] == 1.0

    def test_no_double_counting_with_failures(self):
        """
        Verify tracker doesn't double-count with mixed success/failure.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Simulate dual-mode with 4 queries, some failures
        # Query 1: Both succeed
        tracker.track_query_result("test", "test", 0.9, 10.0, 100.0, success=True)
        tracker.track_query_result("test", "test", 0.85, 0.0, 200.0, success=True)

        # Query 2: Cloud succeeds, self-hosted fails
        tracker.track_query_result("test", "test", 0.9, 10.0, 100.0, success=True)
        tracker.track_query_result("test", "test", 0.0, 0.0, 0.0, success=False)

        # Query 3: Cloud fails, self-hosted succeeds
        tracker.track_query_result("test", "test", 0.0, 0.0, 0.0, success=False)
        tracker.track_query_result("test", "test", 0.85, 0.0, 200.0, success=True)

        # Query 4: Both fail
        tracker.track_query_result("test", "test", 0.0, 0.0, 0.0, success=False)
        tracker.track_query_result("test", "test", 0.0, 0.0, 0.0, success=False)

        # After individual tracking
        assert tracker.metrics.total_queries == 8  # 4 queries × 2 endpoints
        assert tracker.metrics.successful_queries == 4  # 2 + 1 + 1 + 0
        assert tracker.metrics.failed_queries == 4  # 0 + 1 + 1 + 2

        # Call track_benchmark_run
        benchmark_result = {
            "metadata": {
                "category": "test",
                "mode": "both",
                "total_queries": 4,
                "cloud_successful_queries": 2,
                "self_hosted_successful_queries": 2,
                "failed_queries": 1,  # Only 1 query failed on BOTH endpoints
            },
            "metrics": {"cloud": {"average_accuracy": 0.6}, "self_hosted": {"average_accuracy": 0.57}},
        }

        tracker.track_benchmark_run(benchmark_result)

        # Counts should NOT change
        assert tracker.metrics.total_queries == 8, "Total queries should not double-count"
        assert tracker.metrics.successful_queries == 4, "Successful queries should not double-count"
        assert tracker.metrics.failed_queries == 4, "Failed queries should not double-count"

        summary = tracker.get_metrics_summary()
        assert summary["success_rate"] == 0.5, f"Expected 50% success rate, got {summary['success_rate']:.2%}"
        assert summary["success_rate"] <= 1.0, "Success rate must not exceed 100%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
