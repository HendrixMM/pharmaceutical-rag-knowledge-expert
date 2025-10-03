"""
Pharmaceutical Benchmark Tracker

Tracks benchmark performance metrics and integrates with PharmaceuticalCostAnalyzer.
Monitors accuracy, cost efficiency, latency, and regression detection.
"""
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BenchmarkMetrics:
    """Container for benchmark performance metrics."""

    def __init__(self):
        self.accuracy_by_category: Dict[str, List[float]] = defaultdict(list)
        self.cost_by_category: Dict[str, List[float]] = defaultdict(list)
        self.cost_by_query_type: Dict[str, List[float]] = defaultdict(list)
        self.latency_by_category: Dict[str, List[float]] = defaultdict(list)
        self.tokens_by_category: Dict[str, List[int]] = defaultdict(list)
        self.regression_flags: List[Dict[str, Any]] = []
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy_by_category": {k: sum(v) / len(v) if v else 0 for k, v in self.accuracy_by_category.items()},
            "cost_by_category": {k: sum(v) / len(v) if v else 0 for k, v in self.cost_by_category.items()},
            "cost_by_query_type": {k: sum(v) / len(v) if v else 0 for k, v in self.cost_by_query_type.items()},
            "latency_by_category": {k: sum(v) / len(v) if v else 0 for k, v in self.latency_by_category.items()},
            "tokens_by_category": {k: int(sum(v) / len(v)) if v else 0 for k, v in self.tokens_by_category.items()},
            "regression_flags": self.regression_flags,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.successful_queries / self.total_queries if self.total_queries > 0 else 0,
        }


class RegressionDetector:
    """Detects performance regressions in benchmarks."""

    def __init__(self, accuracy_threshold: float = 0.05, cost_threshold: float = 0.20, latency_threshold: float = 0.50):
        self.accuracy_threshold = accuracy_threshold
        self.cost_threshold = cost_threshold
        self.latency_threshold = latency_threshold

    def detect_regression(
        self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect regressions by comparing current metrics to baseline."""
        regressions = []

        # Check accuracy regression
        if "accuracy" in current_metrics and "accuracy" in baseline_metrics:
            accuracy_drop = baseline_metrics["accuracy"] - current_metrics["accuracy"]
            if accuracy_drop > self.accuracy_threshold:
                regressions.append(
                    {
                        "type": "accuracy_regression",
                        "severity": "high",
                        "baseline": baseline_metrics["accuracy"],
                        "current": current_metrics["accuracy"],
                        "drop_percent": accuracy_drop * 100,
                        "threshold": self.accuracy_threshold * 100,
                    }
                )

        # Check cost regression
        if "cost_per_query" in current_metrics and "cost_per_query" in baseline_metrics:
            baseline_cost = baseline_metrics["cost_per_query"]
            current_cost = current_metrics["cost_per_query"]

            # Guard against division by zero (e.g., self-hosted baseline cost is always 0)
            if baseline_cost > 0:
                cost_increase = (current_cost - baseline_cost) / baseline_cost
                if cost_increase > self.cost_threshold:
                    regressions.append(
                        {
                            "type": "cost_regression",
                            "severity": "medium",
                            "baseline": baseline_cost,
                            "current": current_cost,
                            "increase_percent": cost_increase * 100,
                            "threshold": self.cost_threshold * 100,
                        }
                    )
            # If baseline is 0, skip percentage check (can't measure % increase from free tier)

        # Check latency regression
        if "latency_ms" in current_metrics and "latency_ms" in baseline_metrics:
            baseline_latency = baseline_metrics["latency_ms"]
            current_latency = current_metrics["latency_ms"]

            # Guard against division by zero (unlikely but possible)
            if baseline_latency > 0:
                latency_increase = (current_latency - baseline_latency) / baseline_latency
                if latency_increase > self.latency_threshold:
                    regressions.append(
                        {
                            "type": "latency_regression",
                            "severity": "medium",
                            "baseline": baseline_latency,
                            "current": current_latency,
                            "increase_percent": latency_increase * 100,
                            "threshold": self.latency_threshold * 100,
                        }
                    )
            # If baseline is 0, skip percentage check

        return regressions


class PharmaceuticalBenchmarkTracker:
    """
    Tracks pharmaceutical benchmark performance and integrates with cost monitoring.

    Integrates with:
    - PharmaceuticalCostAnalyzer for credit tracking
    - EnhancedNeMoClient for query execution metrics
    - BenchmarkRunner for results aggregation
    """

    def __init__(self, cost_analyzer: Optional[Any] = None, baseline_path: Optional[str] = None):
        self.cost_analyzer = cost_analyzer
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.metrics = BenchmarkMetrics()
        self.regression_detector = RegressionDetector()
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        # Controls whether this tracker forwards costs to the analyzer.
        # Runner should explicitly enable this for dual-endpoint mode to avoid duplicates.
        self.cost_forwarding_enabled: bool = False
        # Optional identifier for correlating a benchmark run across systems
        self.current_run_id: Optional[str] = None
        # Optional research project id for budget tracking and analyzer routing
        self.project_id: Optional[str] = None

        if self.baseline_path and self.baseline_path.exists():
            self.load_baseline()

    def load_baseline(self) -> None:
        """Load baseline metrics from file."""
        try:
            with open(self.baseline_path) as f:
                self.baseline_metrics = json.load(f)
            logger.info(f"Loaded baseline metrics from {self.baseline_path}")
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            self.baseline_metrics = None

    def track_query_result(
        self,
        category: str,
        query_type: str,
        accuracy: float,
        cost: float,
        latency_ms: float,
        success: bool = True,
        endpoint: Optional[str] = None,
        run_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
    ) -> None:
        """Track individual query result."""
        self.metrics.total_queries += 1

        if success:
            self.metrics.successful_queries += 1
            self.metrics.accuracy_by_category[category].append(accuracy)
            self.metrics.cost_by_category[category].append(cost)
            self.metrics.cost_by_query_type[query_type].append(cost)
            self.metrics.latency_by_category[category].append(latency_ms)
            try:
                if estimated_tokens is not None:
                    self.metrics.tokens_by_category[category].append(int(estimated_tokens))
            except Exception:
                pass
        else:
            self.metrics.failed_queries += 1
            try:
                if estimated_tokens is not None:
                    self.metrics.tokens_by_category[category].append(int(estimated_tokens))
            except Exception:
                pass

        # Forward to cost analyzer if available using the supported API
        if (
            self.cost_analyzer
            and self.cost_forwarding_enabled
            and hasattr(self.cost_analyzer, "record_pharmaceutical_query")
        ):
            try:
                # Construct a minimal benchmark query record
                query_id = f"bench_{uuid.uuid4().hex}"
                # Provide a synthetic query text for auditability
                effective_run_id = run_id or self.current_run_id
                run_part = f" run={effective_run_id}" if effective_run_id else ""
                query_text = f"[benchmark]{run_part} {category}/{query_type} ({'success' if success else 'failure'})"
                # Infer cost tier: any non-zero credits treated as infrastructure; zero as free tier
                cost_tier = "infrastructure" if (cost or 0) > 0 else "free_tier"
                # Prefer provided estimated_tokens; otherwise heuristic from latency
                est_tokens = (
                    int(estimated_tokens)
                    if isinstance(estimated_tokens, (int, float)) and int(estimated_tokens) > 0
                    else max(1, int((latency_ms or 0) / 2))
                )

                # Tag for downstream analysis to identify tracker-forwarded events
                endpoint_tag = (
                    f"endpoint:{endpoint}"
                    if endpoint
                    else ("endpoint:cloud" if (cost or 0) > 0 else "endpoint:self_hosted")
                )
                tags = [
                    "benchmark",
                    category,
                    query_type,
                    endpoint_tag,
                    f"run:{effective_run_id}" if effective_run_id else "run:unknown",
                    "success" if success else "failure",
                ]
                if idempotency_key:
                    tags.append(f"idem:{idempotency_key}")

                self.cost_analyzer.record_pharmaceutical_query(
                    query_id=query_id,
                    query_text=query_text,
                    cost_tier=cost_tier,
                    estimated_tokens=est_tokens,
                    project_id=(self.project_id or "benchmark_run"),
                    tags=tags,
                )
            except Exception as e:
                # Don't let cost recording break metric tracking
                logger.debug(f"Cost analyzer recording skipped: {e}")

    def _track_endpoint_metrics(self, category: str, endpoint_metrics: Dict[str, Any]) -> None:
        """
        Extract and track metrics from a single endpoint.

        Args:
            category: Benchmark category (e.g., "drug_interactions")
            endpoint_metrics: Metrics dictionary for single endpoint
        """
        if "average_accuracy" in endpoint_metrics:
            self.metrics.accuracy_by_category[category].append(endpoint_metrics["average_accuracy"])

        if "average_credits_per_query" in endpoint_metrics:
            self.metrics.cost_by_category[category].append(endpoint_metrics["average_credits_per_query"])

        if "average_latency_ms" in endpoint_metrics:
            self.metrics.latency_by_category[category].append(endpoint_metrics["average_latency_ms"])

    def track_benchmark_run(self, benchmark_result: Dict[str, Any]) -> None:
        """Track complete benchmark run results."""
        metadata = benchmark_result.get("metadata", {})
        metrics = benchmark_result.get("metrics", {})
        category = metadata.get("category", "unknown")
        mode = metadata.get("mode", "cloud")

        def _normalize_baseline_keys(b: Dict[str, Any]) -> Dict[str, float]:
            """Normalize baseline keys to match detector expectations.

            Accepts either canonical keys (accuracy, cost_per_query, latency_ms)
            or aggregate-style keys (average_accuracy, average_credits_per_query, average_latency_ms).
            """
            if not isinstance(b, dict):
                return {}
            # Prefer canonical if present; otherwise map from average_* keys
            # Cost key can appear as average_cost_per_query (dataset) or average_credits_per_query (runtime)
            cost_value = b.get("cost_per_query", b.get("average_credits_per_query", b.get("average_cost_per_query", 0)))
            return {
                "accuracy": float(b.get("accuracy", b.get("average_accuracy", 0)) or 0),
                "cost_per_query": float(cost_value or 0),
                "latency_ms": float(b.get("latency_ms", b.get("average_latency_ms", 0)) or 0),
            }

        # Handle dual-endpoint results (mode="both")
        if mode == "both":
            # Track cloud endpoint metrics
            cloud_metrics = metrics.get("cloud", {})
            if cloud_metrics:
                self._track_endpoint_metrics(category, cloud_metrics)

            # Track self-hosted endpoint metrics
            sh_metrics = metrics.get("self_hosted", {})
            if sh_metrics:
                self._track_endpoint_metrics(category, sh_metrics)

            # Check for regressions against both baselines
            if self.baseline_metrics:
                baseline = self.baseline_metrics.get(category, {})

                # Check cloud regressions
                if cloud_metrics and "cloud" in baseline:
                    cloud_current = {
                        "accuracy": cloud_metrics.get("average_accuracy", 0),
                        "cost_per_query": cloud_metrics.get("average_credits_per_query", 0),
                        "latency_ms": cloud_metrics.get("average_latency_ms", 0),
                    }
                    cloud_baseline = _normalize_baseline_keys(baseline["cloud"])
                    cloud_regressions = self.regression_detector.detect_regression(cloud_current, cloud_baseline)
                    for regression in cloud_regressions:
                        regression["category"] = category
                        regression["endpoint"] = "cloud"
                        regression["timestamp"] = datetime.now().isoformat()
                        self.metrics.regression_flags.append(regression)
                        logger.warning(f"Cloud regression detected in {category}: {regression['type']}")

                # Check self-hosted regressions
                if sh_metrics and "self_hosted" in baseline:
                    sh_current = {
                        "accuracy": sh_metrics.get("average_accuracy", 0),
                        "cost_per_query": sh_metrics.get("average_credits_per_query", 0),
                        "latency_ms": sh_metrics.get("average_latency_ms", 0),
                    }
                    sh_baseline = _normalize_baseline_keys(baseline["self_hosted"])
                    sh_regressions = self.regression_detector.detect_regression(sh_current, sh_baseline)
                    for regression in sh_regressions:
                        regression["category"] = category
                        regression["endpoint"] = "self_hosted"
                        regression["timestamp"] = datetime.now().isoformat()
                        self.metrics.regression_flags.append(regression)
                        logger.warning(f"Self-hosted regression detected in {category}: {regression['type']}")
        else:
            # Single-mode metrics (original behavior)
            self._track_endpoint_metrics(category, metrics)

            # Check for regressions (single-mode)
            if self.baseline_metrics:
                current = {
                    "accuracy": metrics.get("average_accuracy", 0),
                    "cost_per_query": metrics.get("average_credits_per_query", 0),
                    "latency_ms": metrics.get("average_latency_ms", 0),
                }
                baseline = self.baseline_metrics.get(category, {})

                # For single-mode, baseline might be nested under mode key
                if mode in baseline:
                    baseline = baseline[mode]
                # Normalize baseline keys for regression detector
                baseline_norm = _normalize_baseline_keys(baseline)

                regressions = self.regression_detector.detect_regression(current, baseline_norm)
                if regressions:
                    for regression in regressions:
                        regression["category"] = category
                        regression["timestamp"] = datetime.now().isoformat()
                        self.metrics.regression_flags.append(regression)
                        logger.warning(f"Regression detected in {category}: {regression['type']}")

        # Update query counts from metadata IF per-query results haven't been recorded.
        # This avoids double counting when track_query_result() has already been used.
        if (
            self.metrics.total_queries == 0
            and self.metrics.successful_queries == 0
            and self.metrics.failed_queries == 0
        ):
            try:
                if mode == "both":
                    total = int(metadata.get("total_queries", 0) or 0)
                    cloud_success = int(metadata.get("cloud_successful_queries", 0) or 0)
                    sh_success = int(metadata.get("self_hosted_successful_queries", 0) or 0)

                    # Each query executes on two endpoints in dual-mode
                    total_endpoints = total * 2
                    successes = cloud_success + sh_success
                    failures = max(0, total_endpoints - successes)

                    self.metrics.total_queries = total_endpoints
                    self.metrics.successful_queries = successes
                    self.metrics.failed_queries = failures
                else:
                    total = int(metadata.get("total_queries", 0) or 0)
                    successes = int(metadata.get("successful_queries", 0) or 0)
                    failures = int(metadata.get("failed_queries", total - successes) or 0)

                    self.metrics.total_queries = total
                    self.metrics.successful_queries = successes
                    self.metrics.failed_queries = failures
            except Exception as e:
                # Be conservative: if parsing fails, do not mutate counts
                logger.warning(f"Could not update query counts from metadata: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics."""
        summary = self.metrics.to_dict()

        # Add cost efficiency metrics
        summary["cost_efficiency"] = {}
        for category, costs in self.metrics.cost_by_category.items():
            accuracies = self.metrics.accuracy_by_category.get(category, [])
            tokens = self.metrics.tokens_by_category.get(category, [])
            if costs and accuracies:
                avg_cost = sum(costs) / len(costs)
                avg_accuracy = sum(accuracies) / len(accuracies)
                efficiency = avg_accuracy / avg_cost if avg_cost > 0 else 0
                summary["cost_efficiency"][category] = {
                    "avg_cost": round(avg_cost, 2),
                    "avg_accuracy": round(avg_accuracy, 3),
                    "avg_tokens": int(sum(tokens) / len(tokens)) if tokens else 0,
                    "efficiency_score": round(efficiency, 4),
                }

        return summary

    def check_performance_targets(self, targets: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """Check if performance meets targets."""
        results = {}

        for category, category_targets in targets.items():
            if category not in self.metrics.accuracy_by_category:
                results[category] = False
                continue

            # Check accuracy target
            avg_accuracy = sum(self.metrics.accuracy_by_category[category]) / len(
                self.metrics.accuracy_by_category[category]
            )
            accuracy_met = avg_accuracy >= category_targets.get("accuracy", 0.8)

            # Check cost target if available
            cost_met = True
            if "cost" in category_targets and category in self.metrics.cost_by_category:
                avg_cost = sum(self.metrics.cost_by_category[category]) / len(self.metrics.cost_by_category[category])
                cost_met = avg_cost <= category_targets["cost"]

            results[category] = accuracy_met and cost_met

        return results

    def export_metrics(self, output_path: str) -> None:
        """Export metrics to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_export = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.current_run_id,
            "summary": self.get_metrics_summary(),
            "regression_flags": self.metrics.regression_flags,
        }

        with open(output_path, "w") as f:
            json.dump(metrics_export, f, indent=2)

        logger.info(f"Exported metrics to {output_path}")

    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return len(self.metrics.regression_flags) > 0

    def get_regression_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all detected regressions."""
        return self.metrics.regression_flags
