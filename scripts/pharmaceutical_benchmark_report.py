#!/usr/bin/env python3
"""
Pharmaceutical Benchmark Reporter

Generates comprehensive reports from benchmark results.
Supports multiple formats: JSON, YAML, Markdown, HTML.

Usage:
    python scripts/pharmaceutical_benchmark_report.py results/benchmark_results.json
    python scripts/pharmaceutical_benchmark_report.py results/ --format markdown
    python scripts/pharmaceutical_benchmark_report.py results/ --compare v1 v2
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from benchmark results."""

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results

    @staticmethod
    def _is_dual_mode(result: Dict[str, Any]) -> bool:
        """Check if result uses dual-endpoint mode."""
        return result.get("metadata", {}).get("mode") == "both"

    @staticmethod
    def _get_successful_queries(result: Dict[str, Any]) -> int:
        """Extract successful query count, handling both modes."""
        if ReportGenerator._is_dual_mode(result):
            # For dual-mode, use max of cloud/self-hosted
            cloud = result.get("metadata", {}).get("cloud_successful_queries", 0)
            sh = result.get("metadata", {}).get("self_hosted_successful_queries", 0)
            return max(cloud, sh)
        else:
            return result.get("metadata", {}).get("successful_queries", 0)

    @staticmethod
    def _get_metrics_list(result: Dict[str, Any]) -> List[tuple]:
        """
        Extract metrics as list of (endpoint_name, metrics_dict) tuples.

        Returns:
            For dual-mode: [("cloud", cloud_metrics), ("self_hosted", sh_metrics)]
            For single-mode: [(mode_name, metrics)]
        """
        if ReportGenerator._is_dual_mode(result):
            return [
                ("cloud", result.get("metrics", {}).get("cloud", {})),
                ("self_hosted", result.get("metrics", {}).get("self_hosted", {}))
            ]
        else:
            mode = result.get("metadata", {}).get("mode", "single")
            return [(mode, result.get("metrics", {}))]

    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of benchmark results."""
        total_queries = sum(r.get("metadata", {}).get("total_queries", 0) for r in self.results)
        total_successful = sum(self._get_successful_queries(r) for r in self.results)
        total_failed = sum(r.get("metadata", {}).get("failed_queries", 0) for r in self.results)

        # Flatten all endpoint metrics for averaging
        all_scores = []
        all_credits = []

        for result in self.results:
            for endpoint_name, metrics in self._get_metrics_list(result):
                score = metrics.get("average_overall_score", 0)
                credits = metrics.get("total_credits", 0)
                all_scores.append(score)
                all_credits.append(credits)

        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        total_credits = sum(all_credits)

        return {
            "total_benchmarks": len(self.results),
            "total_queries": total_queries,
            "successful_queries": total_successful,
            "failed_queries": total_failed,
            "success_rate": total_successful / total_queries if total_queries > 0 else 0,
            "overall_average_score": round(overall_avg, 3),
            "total_credits_used": total_credits,
            "average_credits_per_query": round(total_credits / total_queries, 2) if total_queries > 0 else 0
        }

    def generate_category_breakdown(self) -> Dict[str, Any]:
        """Generate breakdown by category."""
        breakdown = {}

        for result in self.results:
            category = result.get("metadata", {}).get("category", "unknown")
            total_queries = result.get("metadata", {}).get("total_queries", 0)
            successful_queries = self._get_successful_queries(result)

            # Get all endpoint metrics (single or dual)
            for endpoint_name, metrics in self._get_metrics_list(result):
                # For dual-mode, append endpoint name to category
                if self._is_dual_mode(result):
                    category_key = f"{category} ({endpoint_name})"
                else:
                    category_key = category

                breakdown[category_key] = {
                    "total_queries": total_queries,
                    "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                    "average_accuracy": metrics.get("average_accuracy", 0),
                    "average_score": metrics.get("average_overall_score", 0),
                    "average_latency_ms": metrics.get("average_latency_ms", 0),
                    "average_credits": metrics.get("average_credits_per_query", 0),
                    "total_credits": metrics.get("total_credits", 0),
                    "average_tokens": metrics.get("average_tokens", 0)
                }

        return breakdown

    def generate_cost_analysis(self) -> Dict[str, Any]:
        """Generate cost analysis."""
        cost_by_category = defaultdict(list)

        for result in self.results:
            category = result.get("metadata", {}).get("category", "unknown")
            total_queries = result.get("metadata", {}).get("total_queries", 0)

            # Get all endpoint metrics (single or dual)
            for endpoint_name, metrics in self._get_metrics_list(result):
                # For dual-mode, append endpoint name to category
                if self._is_dual_mode(result):
                    category_key = f"{category} ({endpoint_name})"
                else:
                    category_key = category

                cost_by_category[category_key].append({
                    "avg_credits_per_query": metrics.get("average_credits_per_query", 0),
                    "total_credits": metrics.get("total_credits", 0),
                    "query_count": total_queries
                })

        # Calculate cost efficiency (score per credit)
        cost_efficiency = {}
        for result in self.results:
            category = result.get("metadata", {}).get("category", "unknown")

            for endpoint_name, metrics in self._get_metrics_list(result):
                if self._is_dual_mode(result):
                    category_key = f"{category} ({endpoint_name})"
                else:
                    category_key = category

                avg_score = metrics.get("average_overall_score", 0)
                avg_cost = metrics.get("average_credits_per_query", 0)
                efficiency = avg_score / avg_cost if avg_cost > 0 else 0
                cost_efficiency[category_key] = round(efficiency, 4)

        # Calculate total cost from all endpoints
        total_cost = 0
        for result in self.results:
            for endpoint_name, metrics in self._get_metrics_list(result):
                total_cost += metrics.get("total_credits", 0)

        return {
            "cost_by_category": dict(cost_by_category),
            "cost_efficiency": cost_efficiency,
            "total_cost": total_cost
        }

    def generate_full_report(self) -> Dict[str, Any]:
        """Generate complete report."""
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0"
            },
            "executive_summary": self.generate_executive_summary(),
            "category_breakdown": self.generate_category_breakdown(),
            "cost_analysis": self.generate_cost_analysis(),
            "detailed_results": self.results
        }


class MarkdownReportGenerator:
    """Generates Markdown-formatted reports."""

    @staticmethod
    def generate(report: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        lines = []

        # Header
        lines.append("# Pharmaceutical Benchmark Report")
        lines.append("")
        lines.append(f"Generated: {report['report_metadata']['generated_at']}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        summary = report["executive_summary"]
        lines.append(f"- **Total Benchmarks**: {summary['total_benchmarks']}")
        lines.append(f"- **Total Queries**: {summary['total_queries']}")
        lines.append(f"- **Success Rate**: {summary['success_rate']:.1%}")
        lines.append(f"- **Overall Average Score**: {summary['overall_average_score']:.3f}")
        lines.append(f"- **Total Credits Used**: {summary['total_credits_used']:.2f}")
        lines.append(f"- **Average Credits/Query**: {summary['average_credits_per_query']:.2f}")
        lines.append("")

        # Category Breakdown
        lines.append("## Category Breakdown")
        lines.append("")
        lines.append("| Category | Queries | Success Rate | Avg Score | Avg Latency (ms) | Avg Credits | Avg Tokens |")
        lines.append("|----------|---------|--------------|-----------|------------------|-------------|------------|")

        for category, data in report["category_breakdown"].items():
            lines.append(
                f"| {category} | {data['total_queries']} | "
                f"{data['success_rate']:.1%} | {data['average_score']:.3f} | "
                f"{data['average_latency_ms']:.2f} | {data['average_credits']:.2f} | {int(data.get('average_tokens', 0) or 0)} |"
            )

        lines.append("")

        # Cost Analysis
        lines.append("## Cost Analysis")
        lines.append("")
        cost = report["cost_analysis"]
        lines.append(f"**Total Cost**: {cost['total_cost']:.2f} credits")
        lines.append("")
        lines.append("### Cost Efficiency (Score/Credit)")
        lines.append("")
        for category, efficiency in cost["cost_efficiency"].items():
            lines.append(f"- **{category}**: {efficiency:.4f}")

        lines.append("")

        return "\n".join(lines)


class ComparisonReportGenerator:
    """Generates comparison reports between versions."""

    def __init__(self, baseline: List[Dict], current: List[Dict]):
        self.baseline = baseline
        self.current = current

    @staticmethod
    def _extract_metrics_for_comparison(result: Dict[str, Any], preferred_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metrics from result, handling both single and dual-mode.

        Args:
            result: Benchmark result dict
            preferred_mode: If result is dual-mode, which endpoint to extract (cloud/self_hosted)

        Returns:
            Metrics dict with average_accuracy, average_credits_per_query, average_latency_ms
        """
        if not result:
            return {}

        mode = result.get("metadata", {}).get("mode")
        metrics = result.get("metrics", {})

        if mode == "both":
            # Dual-mode result
            if preferred_mode and preferred_mode in metrics:
                # Extract specific endpoint
                return metrics.get(preferred_mode, {})
            else:
                # Default to cloud for backward compatibility
                return metrics.get("cloud", {})
        else:
            # Single-mode result (cloud or self_hosted)
            return metrics

    def generate_comparison(self) -> Dict[str, Any]:
        """Generate comparison between versions."""
        baseline_by_category = {r.get("metadata", {}).get("category", "unknown"): r for r in self.baseline}
        current_by_category = {r.get("metadata", {}).get("category", "unknown"): r for r in self.current}

        comparisons = {}

        for category in set(baseline_by_category.keys()) | set(current_by_category.keys()):
            baseline_result = baseline_by_category.get(category, {})
            current_result = current_by_category.get(category, {})

            if not baseline_result or not current_result:
                continue

            # Detect modes
            baseline_mode = baseline_result.get("metadata", {}).get("mode")
            current_mode = current_result.get("metadata", {}).get("mode")

            # Handle different mode combinations
            if baseline_mode == "both" and current_mode == "both":
                # Both dual-mode: compare cloud-to-cloud and self_hosted-to-self_hosted
                for endpoint in ["cloud", "self_hosted"]:
                    baseline_metrics = self._extract_metrics_for_comparison(baseline_result, endpoint)
                    current_metrics = self._extract_metrics_for_comparison(current_result, endpoint)

                    category_key = f"{category} ({endpoint})"
                    comparisons[category_key] = self._build_comparison(
                        baseline_metrics, current_metrics
                    )
            elif baseline_mode == "both" or current_mode == "both":
                # Mode mismatch: extract matching endpoint from dual-mode result
                # Determine which mode to extract from dual-mode result
                single_mode = baseline_mode if baseline_mode != "both" else current_mode

                # Map single mode to endpoint (cloud -> cloud, self_hosted -> self_hosted)
                endpoint = single_mode if single_mode in ["cloud", "self_hosted"] else "cloud"

                baseline_metrics = self._extract_metrics_for_comparison(baseline_result, endpoint)
                current_metrics = self._extract_metrics_for_comparison(current_result, endpoint)

                comparisons[category] = self._build_comparison(baseline_metrics, current_metrics)
            else:
                # Both single-mode: compare directly
                baseline_metrics = self._extract_metrics_for_comparison(baseline_result)
                current_metrics = self._extract_metrics_for_comparison(current_result)

                comparisons[category] = self._build_comparison(baseline_metrics, current_metrics)

        return comparisons

    def _build_comparison(self, baseline_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build comparison dict from baseline and current metrics."""
        accuracy_change = current_metrics.get("average_accuracy", 0) - baseline_metrics.get("average_accuracy", 0)
        cost_change = current_metrics.get("average_credits_per_query", 0) - baseline_metrics.get("average_credits_per_query", 0)
        latency_change = current_metrics.get("average_latency_ms", 0) - baseline_metrics.get("average_latency_ms", 0)

        return {
            "baseline": {
                "accuracy": baseline_metrics.get("average_accuracy", 0),
                "cost": baseline_metrics.get("average_credits_per_query", 0),
                "latency": baseline_metrics.get("average_latency_ms", 0)
            },
            "current": {
                "accuracy": current_metrics.get("average_accuracy", 0),
                "cost": current_metrics.get("average_credits_per_query", 0),
                "latency": current_metrics.get("average_latency_ms", 0)
            },
            "changes": {
                "accuracy_change": round(accuracy_change, 3),
                "cost_change": round(cost_change, 2),
                "latency_change": round(latency_change, 2)
            },
            "regression_flags": self._check_regressions(
                baseline_accuracy=baseline_metrics.get("average_accuracy", 0),
                current_accuracy=current_metrics.get("average_accuracy", 0),
                baseline_cost=baseline_metrics.get("average_credits_per_query", 0),
                current_cost=current_metrics.get("average_credits_per_query", 0),
                baseline_latency=baseline_metrics.get("average_latency_ms", 0),
                current_latency=current_metrics.get("average_latency_ms", 0)
            )
        }

    @staticmethod
    def _check_regressions(
        baseline_accuracy: float,
        current_accuracy: float,
        baseline_cost: float,
        current_cost: float,
        baseline_latency: float,
        current_latency: float
    ) -> List[str]:
        """
        Check for performance regressions using proper percentage calculations.

        Args:
            baseline_accuracy: Baseline accuracy score (0-1)
            current_accuracy: Current accuracy score (0-1)
            baseline_cost: Baseline cost per query (credits)
            current_cost: Current cost per query (credits)
            baseline_latency: Baseline latency (milliseconds)
            current_latency: Current latency (milliseconds)

        Returns:
            List of regression flag strings
        """
        flags = []

        # Accuracy regression (5% drop from baseline)
        if baseline_accuracy > 0:
            acc_change_pct = ((current_accuracy - baseline_accuracy) / baseline_accuracy) * 100
            if acc_change_pct < -5:
                flags.append("accuracy_regression")

        # Cost regression (20% increase from baseline)
        if baseline_cost > 0:
            cost_change_pct = ((current_cost - baseline_cost) / baseline_cost) * 100
            if cost_change_pct > 20:
                flags.append("cost_regression")

        # Latency regression (50% increase from baseline)
        if baseline_latency > 0:
            lat_change_pct = ((current_latency - baseline_latency) / baseline_latency) * 100
            if lat_change_pct > 50:
                flags.append("latency_regression")

        return flags


def load_results(path: Path) -> List[Dict[str, Any]]:
    """Load benchmark results from file or directory."""
    if path.is_file():
        with open(path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    elif path.is_dir():
        results = []
        for file in sorted(path.glob("benchmark_results_*.json")):
            with open(file, 'r') as f:
                data = json.load(f)
                results.extend(data if isinstance(data, list) else [data])
        return results

    else:
        raise FileNotFoundError(f"Path not found: {path}")


def main():
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate pharmaceutical benchmark reports"
    )
    parser.add_argument(
        'results_path',
        help='Path to results file or directory'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'yaml', 'markdown', 'all'],
        default='markdown',
        help='Output format'
    )
    parser.add_argument(
        '--output',
        help='Output file path'
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('BASELINE', 'CURRENT'),
        help='Compare two result files'
    )

    args = parser.parse_args()

    try:
        # Load results
        results_path = Path(args.results_path)
        results = load_results(results_path)
        logger.info(f"Loaded {len(results)} benchmark results")

        # Generate report
        if args.compare:
            # Comparison mode
            baseline = load_results(Path(args.compare[0]))
            current = load_results(Path(args.compare[1]))

            comparator = ComparisonReportGenerator(baseline, current)
            report = {
                "comparison": comparator.generate_comparison(),
                "baseline_summary": ReportGenerator(baseline).generate_executive_summary(),
                "current_summary": ReportGenerator(current).generate_executive_summary()
            }
        else:
            # Standard report
            generator = ReportGenerator(results)
            report = generator.generate_full_report()

        # Output report
        if args.format == 'json' or args.format == 'all':
            output = json.dumps(report, indent=2)
            if args.output:
                output_path = Path(args.output).with_suffix('.json')
                with open(output_path, 'w') as f:
                    f.write(output)
                logger.info(f"JSON report saved to {output_path}")
            else:
                print(output)

        if args.format == 'yaml' or args.format == 'all':
            output = yaml.dump(report, default_flow_style=False)
            if args.output:
                output_path = Path(args.output).with_suffix('.yaml')
                with open(output_path, 'w') as f:
                    f.write(output)
                logger.info(f"YAML report saved to {output_path}")
            else:
                print(output)

        if args.format == 'markdown' or args.format == 'all':
            output = MarkdownReportGenerator.generate(report)
            if args.output:
                output_path = Path(args.output).with_suffix('.md')
                with open(output_path, 'w') as f:
                    f.write(output)
                logger.info(f"Markdown report saved to {output_path}")
            else:
                print(output)

        logger.info("Report generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
