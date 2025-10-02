"""
Pharmaceutical Benchmark Test Suite

Comprehensive tests for pharmaceutical benchmarking system.
Tests benchmark datasets, execution, tracking, and reporting.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import benchmark components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pharmaceutical_benchmarks import BenchmarkLoader, QueryEvaluator, BenchmarkRunner, BenchmarkConfig
from scripts.generate_pharmaceutical_benchmarks import BenchmarkGenerator, DrugDataLoader
from src.monitoring.pharmaceutical_benchmark_tracker import (
    BenchmarkMetrics,
    RegressionDetector,
    PharmaceuticalBenchmarkTracker
)


class TestBenchmarkDatasets:
    """Test benchmark dataset integrity and structure."""

    @pytest.fixture
    def benchmarks_dir(self):
        """Fixture for benchmarks directory."""
        return Path("benchmarks")

    def test_all_datasets_exist(self, benchmarks_dir):
        """Test that all required benchmark datasets exist."""
        categories = [
            "drug_interactions",
            "pharmacokinetics",
            "clinical_terminology",
            "mechanism_of_action",
            "adverse_reactions"
        ]

        for category in categories:
            file_path = benchmarks_dir / f"{category}_v1.json"
            assert file_path.exists(), f"Missing benchmark: {file_path}"

    def test_dataset_structure(self, benchmarks_dir):
        """Test that datasets have correct structure."""
        file_path = benchmarks_dir / "drug_interactions_v1.json"

        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Check required top-level keys
            assert "metadata" in data
            assert "queries" in data

            # Check metadata structure
            metadata = data["metadata"]
            assert "version" in metadata
            assert "category" in metadata
            assert "created_date" in metadata
            assert "total_queries" in metadata

            # Check queries structure
            queries = data["queries"]
            assert isinstance(queries, list)
            assert len(queries) > 0

            # Check first query structure
            query = queries[0]
            assert "id" in query
            assert "query" in query
            assert "expected_type" in query
            assert "expected_content" in query
            assert "evaluation_criteria" in query
            assert "tags" in query

    def test_query_count_matches_metadata(self, benchmarks_dir):
        """Test that actual query count matches metadata."""
        for file_path in benchmarks_dir.glob("*_v1.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)

            metadata_count = data["metadata"]["total_queries"]
            actual_count = len(data["queries"])

            assert metadata_count == actual_count, \
                f"Query count mismatch in {file_path.name}: metadata={metadata_count}, actual={actual_count}"

    def test_evaluation_criteria_weights(self, benchmarks_dir):
        """Test that evaluation criteria weights sum to 1.0."""
        for file_path in benchmarks_dir.glob("*_v1.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)

            for query in data["queries"]:
                criteria = query["evaluation_criteria"]
                total_weight = (
                    criteria.get("accuracy_weight", 0) +
                    criteria.get("completeness_weight", 0) +
                    criteria.get("relevance_weight", 0)
                )

                assert abs(total_weight - 1.0) < 0.01, \
                    f"Weights don't sum to 1.0 in {file_path.name}, query {query['id']}: {total_weight}"


class TestBenchmarkLoader:
    """Test benchmark loading functionality."""

    @pytest.fixture
    def loader(self):
        """Fixture for BenchmarkLoader."""
        return BenchmarkLoader("benchmarks")

    def test_load_benchmark(self, loader):
        """Test loading a specific benchmark."""
        benchmark = loader.load_benchmark("drug_interactions", version=1)

        assert "metadata" in benchmark
        assert "queries" in benchmark
        assert benchmark["metadata"]["category"] == "drug_interactions"

    def test_load_nonexistent_benchmark(self, loader):
        """Test loading non-existent benchmark raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load_benchmark("nonexistent_category", version=1)

    def test_list_available_benchmarks(self, loader):
        """Test listing available benchmarks."""
        benchmarks = loader.list_available_benchmarks()

        assert isinstance(benchmarks, list)
        assert len(benchmarks) >= 5  # At least 5 v1 benchmarks

        # Check structure
        for category, version in benchmarks:
            assert isinstance(category, str)
            assert isinstance(version, int)


class TestQueryEvaluator:
    """Test query evaluation logic."""

    @pytest.fixture
    def evaluator(self):
        """Fixture for QueryEvaluator."""
        return QueryEvaluator()

    def test_perfect_match(self, evaluator):
        """Test evaluation with perfect keyword match."""
        response = "This response contains interaction mechanism and clinical significance"
        expected = ["interaction mechanism", "clinical significance"]
        weights = {"accuracy_weight": 0.4, "completeness_weight": 0.3, "relevance_weight": 0.3}

        scores = evaluator.calculate_score(response, expected, weights)

        assert scores["accuracy"] == 1.0
        assert scores["overall"] > 0.8

    def test_partial_match(self, evaluator):
        """Test evaluation with partial keyword match."""
        response = "This response contains interaction mechanism only"
        expected = ["interaction mechanism", "clinical significance"]
        weights = {"accuracy_weight": 0.4, "completeness_weight": 0.3, "relevance_weight": 0.3}

        scores = evaluator.calculate_score(response, expected, weights)

        assert scores["accuracy"] == 0.5  # 1 out of 2 keywords

    def test_empty_response(self, evaluator):
        """Test evaluation with empty response."""
        response = ""
        expected = ["keyword1", "keyword2"]
        weights = {"accuracy_weight": 0.4, "completeness_weight": 0.3, "relevance_weight": 0.3}

        scores = evaluator.calculate_score(response, expected, weights)

        assert scores["accuracy"] == 0.0
        assert scores["overall"] == 0.0


class TestBenchmarkGenerator:
    """Test benchmark generation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Fixture for temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def generator(self, temp_output_dir):
        """Fixture for BenchmarkGenerator."""
        return BenchmarkGenerator(output_dir=str(temp_output_dir))

    def test_generate_metadata(self, generator):
        """Test metadata generation."""
        metadata = generator.generate_metadata("drug_interactions", version=2, total_queries=50)

        assert metadata["version"] == "2"
        assert metadata["category"] == "drug_interactions"
        assert metadata["total_queries"] == 50
        assert "created_date" in metadata

    def test_generate_sample_queries(self, generator):
        """Test sample query generation."""
        queries = generator.generate_sample_queries("drug_interactions", count=5)

        assert len(queries) == 5
        for query in queries:
            assert "id" in query
            assert "query" in query
            assert "expected_type" in query

    def test_generate_benchmark(self, generator):
        """Test complete benchmark generation."""
        benchmark = generator.generate_benchmark("pharmacokinetics", version=1, num_queries=10)

        assert "metadata" in benchmark
        assert "queries" in benchmark
        assert len(benchmark["queries"]) == 10


class TestRegressionDetector:
    """Test regression detection logic."""

    @pytest.fixture
    def detector(self):
        """Fixture for RegressionDetector."""
        return RegressionDetector(
            accuracy_threshold=0.05,
            cost_threshold=0.20,
            latency_threshold=0.50
        )

    def test_accuracy_regression(self, detector):
        """Test detection of accuracy regression."""
        baseline = {"accuracy": 0.85}
        current = {"accuracy": 0.75}  # 10% drop

        regressions = detector.detect_regression(current, baseline)

        assert len(regressions) > 0
        assert any(r["type"] == "accuracy_regression" for r in regressions)

    def test_cost_regression(self, detector):
        """Test detection of cost regression."""
        baseline = {"cost_per_query": 10.0}
        current = {"cost_per_query": 15.0}  # 50% increase

        regressions = detector.detect_regression(current, baseline)

        assert len(regressions) > 0
        assert any(r["type"] == "cost_regression" for r in regressions)

    def test_no_regression(self, detector):
        """Test no regression when metrics improve."""
        baseline = {"accuracy": 0.80, "cost_per_query": 15.0}
        current = {"accuracy": 0.85, "cost_per_query": 12.0}

        regressions = detector.detect_regression(current, baseline)

        assert len(regressions) == 0


class TestPharmaceuticalBenchmarkTracker:
    """Test benchmark tracking functionality."""

    @pytest.fixture
    def tracker(self):
        """Fixture for PharmaceuticalBenchmarkTracker."""
        return PharmaceuticalBenchmarkTracker()

    def test_track_query_result(self, tracker):
        """Test tracking individual query results."""
        tracker.track_query_result(
            category="drug_interactions",
            query_type="comparison",
            accuracy=0.85,
            cost=12.5,
            latency_ms=350.0,
            success=True
        )

        assert tracker.metrics.total_queries == 1
        assert tracker.metrics.successful_queries == 1
        assert len(tracker.metrics.accuracy_by_category["drug_interactions"]) == 1

    def test_track_benchmark_run(self, tracker):
        """Test tracking complete benchmark run."""
        result = {
            "metadata": {
                "category": "pharmacokinetics",
                "total_queries": 50,
                "successful_queries": 48,
                "failed_queries": 2
            },
            "metrics": {
                "average_accuracy": 0.82,
                "average_credits_per_query": 11.5,
                "average_latency_ms": 420.0
            }
        }

        tracker.track_benchmark_run(result)

        assert tracker.metrics.total_queries == 50
        assert tracker.metrics.successful_queries == 48

    def test_get_metrics_summary(self, tracker):
        """Test getting metrics summary."""
        tracker.track_query_result("test_category", "test_type", 0.9, 10.0, 300.0, True)

        summary = tracker.get_metrics_summary()

        assert "accuracy_by_category" in summary
        assert "cost_by_category" in summary
        assert "total_queries" in summary
        assert summary["total_queries"] == 1

    def test_regression_baseline_key_normalization(self):
        """Regression: baseline keys normalized for detector.

        Baseline may store aggregate-style keys like average_accuracy, etc.
        Ensure track_benchmark_run() normalizes them to accuracy/cost_per_query/latency_ms.
        """
        tracker = PharmaceuticalBenchmarkTracker()

        # Baseline with aggregate-style keys (single-mode, no nested mode key)
        tracker.baseline_metrics = {
            "drug_safety": {
                "average_accuracy": 0.90,
                "average_credits_per_query": 10.0,
                "average_latency_ms": 400.0
            }
        }

        # Result indicating an accuracy drop triggering regression
        result = {
            "metadata": {
                "category": "drug_safety",
                "mode": "cloud",
                "total_queries": 5,
                "successful_queries": 5,
                "failed_queries": 0
            },
            "metrics": {
                "average_accuracy": 0.80,  # 10% drop from baseline
                "average_credits_per_query": 10.0,
                "average_latency_ms": 400.0
            }
        }

        tracker.track_benchmark_run(result)

        # A regression should be detected due to accuracy drop
        assert tracker.has_regressions()
        regressions = tracker.get_regression_summary()
        assert any(r.get("type") == "accuracy_regression" for r in regressions)


class TestBenchmarkRunner:
    """Test benchmark execution."""

    @pytest.fixture
    def config(self):
        """Fixture for BenchmarkConfig."""
        return BenchmarkConfig()

    @pytest.fixture
    def runner(self, config):
        """Fixture for BenchmarkRunner."""
        # Use simulation to avoid hitting real endpoints in unit tests
        return BenchmarkRunner(config, use_real_clients=False)

    def test_execute_query(self, runner):
        """Test query execution."""
        response, latency, credits = runner.execute_query("Test query")

        assert isinstance(response, str)
        assert isinstance(latency, float)
        assert isinstance(credits, int)
        assert latency >= 0
        assert credits >= 0


class TestConfiguration:
    """Test configuration loading."""

    def test_load_config(self):
        """Test loading benchmark configuration."""
        config = BenchmarkConfig("config/benchmarks.yaml")

        assert "datasets" in config.config
        assert "evaluation" in config.config

    def test_default_config(self):
        """Test default configuration when file missing."""
        with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp:
            # Use non-existent path
            config = BenchmarkConfig("/nonexistent/config.yaml")

            default = config.config
            assert "datasets" in default
            assert "evaluation" in default


class TestDrugDataLoader:
    """Test drug data loading."""

    @pytest.fixture
    def loader(self):
        """Fixture for DrugDataLoader."""
        return DrugDataLoader("Data")

    def test_load_drugs(self, loader):
        """Test loading drug names."""
        loader.load_drugs()

        # Should have loaded some drugs (if files exist)
        if Path("Data/drugs_brand.txt").exists():
            assert len(loader.brand_drugs) > 0

        if Path("Data/drugs_generic.txt").exists():
            assert len(loader.generic_drugs) > 0

    def test_get_random_drug(self, loader):
        """Test getting random drug name."""
        loader.load_drugs()

        if loader.brand_drugs or loader.generic_drugs:
            drug = loader.get_random_drug()
            assert isinstance(drug, str)
            assert len(drug) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
