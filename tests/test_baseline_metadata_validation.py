"""
Test Baseline Metadata Validation

Tests for baseline metadata from Comment 3 verification.
Validates that all benchmark datasets have proper baseline metadata structure
with cloud, self_hosted, and regression_thresholds sections.
"""
import json

import pytest


@pytest.mark.pharmaceutical
@pytest.mark.baseline_validation
@pytest.mark.unit
@pytest.mark.fast
class TestBaselineMetadataPresence:
    """Test that baseline metadata exists in all benchmark files."""

    @pytest.fixture
    def benchmark_files(self, benchmarks_directory):
        """Get all v1 benchmark JSON files."""
        return list(benchmarks_directory.glob("*_v1.json"))

    def test_all_benchmarks_have_baselines(self, benchmark_files):
        """Test that all 5 benchmark files have metadata.baselines section."""
        assert len(benchmark_files) == 5, f"Expected 5 benchmark files, found {len(benchmark_files)}"

        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            assert "metadata" in data, f"{file_path.name}: Missing 'metadata' key"
            assert "baselines" in data["metadata"], f"{file_path.name}: Missing 'metadata.baselines' key"

    def test_baselines_has_all_subsections(self, benchmark_files):
        """Test that baselines section has cloud, self_hosted, and regression_thresholds."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            baselines = data["metadata"]["baselines"]

            assert "cloud" in baselines, f"{file_path.name}: Missing 'cloud' in baselines"
            assert "self_hosted" in baselines, f"{file_path.name}: Missing 'self_hosted' in baselines"
            assert (
                "regression_thresholds" in baselines
            ), f"{file_path.name}: Missing 'regression_thresholds' in baselines"


@pytest.mark.pharmaceutical
@pytest.mark.baseline_validation
@pytest.mark.unit
@pytest.mark.fast
class TestCloudBaselineStructure:
    """Test cloud baseline metadata structure."""

    @pytest.fixture
    def benchmark_files(self, benchmarks_directory):
        """Get all v1 benchmark JSON files."""
        return list(benchmarks_directory.glob("*_v1.json"))

    def test_cloud_baseline_required_fields(self, benchmark_files):
        """Test that cloud baseline has all required fields."""
        required_fields = ["average_latency_ms", "success_rate", "average_cost_per_query", "average_accuracy", "notes"]

        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            cloud_baseline = data["metadata"]["baselines"]["cloud"]

            for field in required_fields:
                assert field in cloud_baseline, f"{file_path.name}: Missing '{field}' in cloud baseline"

    def test_cloud_baseline_field_types(self, benchmark_files):
        """Test that cloud baseline fields have correct types."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            cloud = data["metadata"]["baselines"]["cloud"]

            assert isinstance(
                cloud["average_latency_ms"], (int, float)
            ), f"{file_path.name}: average_latency_ms should be numeric"
            assert isinstance(cloud["success_rate"], (int, float)), f"{file_path.name}: success_rate should be numeric"
            assert isinstance(
                cloud["average_cost_per_query"], (int, float)
            ), f"{file_path.name}: average_cost_per_query should be numeric"
            assert isinstance(
                cloud["average_accuracy"], (int, float)
            ), f"{file_path.name}: average_accuracy should be numeric"
            assert isinstance(cloud["notes"], str), f"{file_path.name}: notes should be string"

    def test_cloud_baseline_value_ranges(self, benchmark_files):
        """Test that cloud baseline values are in reasonable ranges."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            cloud = data["metadata"]["baselines"]["cloud"]

            # Latency should be positive
            assert cloud["average_latency_ms"] > 0, f"{file_path.name}: average_latency_ms should be positive"

            # Success rate between 0 and 1
            assert 0 <= cloud["success_rate"] <= 1, f"{file_path.name}: success_rate should be between 0 and 1"

            # Accuracy between 0 and 1
            assert 0 <= cloud["average_accuracy"] <= 1, f"{file_path.name}: average_accuracy should be between 0 and 1"

            # Cost should be positive (cloud has cost)
            assert (
                cloud["average_cost_per_query"] > 0
            ), f"{file_path.name}: cloud average_cost_per_query should be positive"

            # Notes should not be empty
            assert len(cloud["notes"]) > 0, f"{file_path.name}: notes should not be empty"


@pytest.mark.pharmaceutical
@pytest.mark.baseline_validation
@pytest.mark.unit
@pytest.mark.fast
class TestSelfHostedBaselineStructure:
    """Test self_hosted baseline metadata structure."""

    @pytest.fixture
    def benchmark_files(self, benchmarks_directory):
        """Get all v1 benchmark JSON files."""
        return list(benchmarks_directory.glob("*_v1.json"))

    def test_self_hosted_baseline_required_fields(self, benchmark_files):
        """Test that self_hosted baseline has all required fields."""
        required_fields = ["average_latency_ms", "success_rate", "average_cost_per_query", "average_accuracy", "notes"]

        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            self_hosted = data["metadata"]["baselines"]["self_hosted"]

            for field in required_fields:
                assert field in self_hosted, f"{file_path.name}: Missing '{field}' in self_hosted baseline"

    def test_self_hosted_baseline_field_types(self, benchmark_files):
        """Test that self_hosted baseline fields have correct types."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            self_hosted = data["metadata"]["baselines"]["self_hosted"]

            assert isinstance(
                self_hosted["average_latency_ms"], (int, float)
            ), f"{file_path.name}: average_latency_ms should be numeric"
            assert isinstance(
                self_hosted["success_rate"], (int, float)
            ), f"{file_path.name}: success_rate should be numeric"
            assert isinstance(
                self_hosted["average_cost_per_query"], (int, float)
            ), f"{file_path.name}: average_cost_per_query should be numeric"
            assert isinstance(
                self_hosted["average_accuracy"], (int, float)
            ), f"{file_path.name}: average_accuracy should be numeric"
            assert isinstance(self_hosted["notes"], str), f"{file_path.name}: notes should be string"

    def test_self_hosted_baseline_zero_cost(self, benchmark_files):
        """Test that self_hosted baseline has zero cost (no API charges)."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            self_hosted = data["metadata"]["baselines"]["self_hosted"]

            assert (
                self_hosted["average_cost_per_query"] == 0.0
            ), f"{file_path.name}: self_hosted cost should be 0.0 (no API charges)"


@pytest.mark.pharmaceutical
@pytest.mark.baseline_validation
@pytest.mark.unit
@pytest.mark.fast
class TestRegressionThresholds:
    """Test regression thresholds structure."""

    @pytest.fixture
    def benchmark_files(self, benchmarks_directory):
        """Get all v1 benchmark JSON files."""
        return list(benchmarks_directory.glob("*_v1.json"))

    def test_regression_thresholds_present(self, benchmark_files):
        """Test that regression thresholds section exists and has required fields."""
        required_fields = ["accuracy_drop_percent", "cost_increase_percent", "latency_increase_percent"]

        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            thresholds = data["metadata"]["baselines"]["regression_thresholds"]

            for field in required_fields:
                assert field in thresholds, f"{file_path.name}: Missing '{field}' in regression_thresholds"

    def test_regression_thresholds_match_config(self, benchmark_files):
        """Test that thresholds match config/benchmarks.yaml values."""
        expected_thresholds = {"accuracy_drop_percent": 5, "cost_increase_percent": 20, "latency_increase_percent": 50}

        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            thresholds = data["metadata"]["baselines"]["regression_thresholds"]

            for key, expected_value in expected_thresholds.items():
                actual_value = thresholds[key]
                assert (
                    actual_value == expected_value
                ), f"{file_path.name}: {key} should be {expected_value}, got {actual_value}"

    def test_regression_thresholds_types(self, benchmark_files):
        """Test that threshold values are numeric."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            thresholds = data["metadata"]["baselines"]["regression_thresholds"]

            assert isinstance(
                thresholds["accuracy_drop_percent"], (int, float)
            ), f"{file_path.name}: accuracy_drop_percent should be numeric"
            assert isinstance(
                thresholds["cost_increase_percent"], (int, float)
            ), f"{file_path.name}: cost_increase_percent should be numeric"
            assert isinstance(
                thresholds["latency_increase_percent"], (int, float)
            ), f"{file_path.name}: latency_increase_percent should be numeric"


@pytest.mark.pharmaceutical
@pytest.mark.baseline_validation
@pytest.mark.unit
@pytest.mark.fast
class TestCloudVsSelfHostedComparison:
    """Test relationships between cloud and self_hosted baselines."""

    @pytest.fixture
    def benchmark_files(self, benchmarks_directory):
        """Get all v1 benchmark JSON files."""
        return list(benchmarks_directory.glob("*_v1.json"))

    def test_cloud_vs_self_hosted_cost_difference(self, benchmark_files):
        """Test that cloud has cost > 0 and self_hosted has cost = 0."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            baselines = data["metadata"]["baselines"]
            cloud_cost = baselines["cloud"]["average_cost_per_query"]
            self_hosted_cost = baselines["self_hosted"]["average_cost_per_query"]

            assert cloud_cost > 0, f"{file_path.name}: cloud cost should be > 0"
            assert self_hosted_cost == 0.0, f"{file_path.name}: self_hosted cost should be 0.0"

    def test_cloud_faster_than_self_hosted(self, benchmark_files):
        """Test that cloud latency is lower than self_hosted latency."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            baselines = data["metadata"]["baselines"]
            cloud_latency = baselines["cloud"]["average_latency_ms"]
            self_hosted_latency = baselines["self_hosted"]["average_latency_ms"]

            assert (
                cloud_latency < self_hosted_latency
            ), f"{file_path.name}: cloud latency ({cloud_latency}ms) should be < self_hosted ({self_hosted_latency}ms)"

    def test_accuracy_comparable_between_modes(self, benchmark_files):
        """Test that accuracy is reasonably comparable between cloud and self_hosted."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            baselines = data["metadata"]["baselines"]
            cloud_accuracy = baselines["cloud"]["average_accuracy"]
            self_hosted_accuracy = baselines["self_hosted"]["average_accuracy"]

            # Accuracy difference should be < 10% (0.1)
            accuracy_diff = abs(cloud_accuracy - self_hosted_accuracy)
            assert (
                accuracy_diff < 0.1
            ), f"{file_path.name}: cloud and self_hosted accuracy too different ({accuracy_diff:.2f})"


@pytest.mark.pharmaceutical
@pytest.mark.baseline_validation
@pytest.mark.unit
@pytest.mark.fast
class TestBaselineMetadataCompleteness:
    """Test overall baseline metadata completeness."""

    @pytest.fixture
    def benchmark_files(self, benchmarks_directory):
        """Get all v1 benchmark JSON files."""
        return list(benchmarks_directory.glob("*_v1.json"))

    def test_baseline_metadata_for_each_category(self, benchmark_files):
        """Test that all 5 categories have valid baseline metadata."""
        expected_categories = [
            "drug_interactions",
            "pharmacokinetics",
            "clinical_terminology",
            "mechanism_of_action",
            "adverse_reactions",
        ]

        found_categories = set()

        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            category = data["metadata"]["category"]
            found_categories.add(category)

            # Verify baselines exist and are complete
            assert "baselines" in data["metadata"], f"{file_path.name}: Missing baselines"
            assert "cloud" in data["metadata"]["baselines"], f"{file_path.name}: Missing cloud baseline"
            assert "self_hosted" in data["metadata"]["baselines"], f"{file_path.name}: Missing self_hosted baseline"
            assert (
                "regression_thresholds" in data["metadata"]["baselines"]
            ), f"{file_path.name}: Missing regression_thresholds"

        # Verify all expected categories were found
        for expected_category in expected_categories:
            assert expected_category in found_categories, f"Missing category: {expected_category}"

    def test_notes_field_populated(self, benchmark_files):
        """Test that notes field is populated with meaningful content."""
        for file_path in benchmark_files:
            with open(file_path) as f:
                data = json.load(f)

            baselines = data["metadata"]["baselines"]

            cloud_notes = baselines["cloud"]["notes"]
            self_hosted_notes = baselines["self_hosted"]["notes"]

            # Notes should be at least 10 characters
            assert len(cloud_notes) >= 10, f"{file_path.name}: cloud notes too short"
            assert len(self_hosted_notes) >= 10, f"{file_path.name}: self_hosted notes too short"

            # Notes should contain relevant keywords
            assert any(
                keyword in cloud_notes.lower() for keyword in ["cloud", "nvidia", "build", "api"]
            ), f"{file_path.name}: cloud notes should mention cloud/API context"
            assert any(
                keyword in self_hosted_notes.lower() for keyword in ["local", "nim", "self", "container", "gpu"]
            ), f"{file_path.name}: self_hosted notes should mention local/NIM context"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
