"""
Test Classifier Validation Feature

Tests for Comment 2 verification: classifier validation functionality.
Tests the new classifier validation against expected_classification in benchmarks.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pharmaceutical_benchmarks import BenchmarkConfig, BenchmarkRunner
from src.pharmaceutical.query_classifier import (
    PharmaceuticalContext,
    PharmaceuticalDomain,
    ResearchPriority,
    SafetyUrgency,
)


@pytest.mark.pharmaceutical
@pytest.mark.classifier
@pytest.mark.unit
@pytest.mark.fast
class TestValidateClassifier:
    """Test _validate_classifier() method."""

    def test_validate_classifier_all_correct(self):
        """Test validation when all fields match."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            safety_urgency=SafetyUrgency.HIGH,
            research_priority=ResearchPriority.NORMAL,
            drug_names=["atorvastatin", "warfarin"],
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "high",
            "research_priority": "normal",
            "drug_names": ["atorvastatin", "warfarin"],
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        assert validation["domain_correct"] is True
        assert validation["safety_urgency_correct"] is True
        assert validation["research_priority_correct"] is True
        assert validation["drug_names_correct"] is True
        assert validation["overall_correct"] is True
        assert len(validation["mismatches"]) == 0

    def test_validate_classifier_domain_mismatch(self):
        """Test validation when domain doesn't match."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.PHARMACOKINETICS,  # Wrong domain
            safety_urgency=SafetyUrgency.HIGH,
            research_priority=ResearchPriority.NORMAL,
            drug_names=["atorvastatin", "warfarin"],
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "high",
            "research_priority": "normal",
            "drug_names": ["atorvastatin", "warfarin"],
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        assert validation["domain_correct"] is False
        assert validation["safety_urgency_correct"] is True
        assert validation["research_priority_correct"] is True
        assert validation["overall_correct"] is False
        assert len(validation["mismatches"]) > 0
        assert any("Domain mismatch" in m for m in validation["mismatches"])

    def test_validate_classifier_safety_urgency_mismatch(self):
        """Test validation when safety urgency doesn't match."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            safety_urgency=SafetyUrgency.LOW,  # Wrong urgency
            research_priority=ResearchPriority.NORMAL,
            drug_names=["atorvastatin", "warfarin"],
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "high",
            "research_priority": "normal",
            "drug_names": ["atorvastatin", "warfarin"],
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        assert validation["domain_correct"] is True
        assert validation["safety_urgency_correct"] is False
        assert validation["research_priority_correct"] is True
        assert validation["overall_correct"] is False
        assert any("Safety urgency mismatch" in m for m in validation["mismatches"])

    def test_validate_classifier_drug_names_partial_match(self):
        """Test validation with partial drug name overlap (>= 50% threshold)."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            safety_urgency=SafetyUrgency.HIGH,
            research_priority=ResearchPriority.NORMAL,
            drug_names=["atorvastatin", "simvastatin"],  # 50% overlap
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "high",
            "research_priority": "normal",
            "drug_names": ["atorvastatin", "warfarin"],
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        # 50% overlap should pass
        assert validation["drug_names_correct"] is True
        assert validation["overall_correct"] is True

    def test_validate_classifier_drug_names_insufficient_overlap(self):
        """Test validation with insufficient drug name overlap (< 50% threshold)."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            safety_urgency=SafetyUrgency.HIGH,
            research_priority=ResearchPriority.NORMAL,
            drug_names=["ibuprofen", "aspirin"],  # 0% overlap
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "high",
            "research_priority": "normal",
            "drug_names": ["atorvastatin", "warfarin"],
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        # 0% overlap should fail
        assert validation["drug_names_correct"] is False
        assert validation["overall_correct"] is False
        assert any("Drug names mismatch" in m for m in validation["mismatches"])

    def test_validate_classifier_case_insensitive(self):
        """Test validation is case-insensitive."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            safety_urgency=SafetyUrgency.HIGH,
            research_priority=ResearchPriority.NORMAL,
            drug_names=["ATORVASTATIN", "WARFARIN"],  # Uppercase
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "HIGH",  # Uppercase
            "research_priority": "NORMAL",  # Uppercase
            "drug_names": ["atorvastatin", "warfarin"],  # Lowercase
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        assert validation["domain_correct"] is True
        assert validation["safety_urgency_correct"] is True
        assert validation["research_priority_correct"] is True
        assert validation["drug_names_correct"] is True
        assert validation["overall_correct"] is True

    def test_validate_classifier_without_drug_names(self):
        """Test validation when drug_names is not provided (optional field)."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        pharma_context = PharmaceuticalContext(
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            safety_urgency=SafetyUrgency.HIGH,
            research_priority=ResearchPriority.NORMAL,
            confidence_score=0.95,
        )

        expected_classification = {
            "domain": "drug_interactions",
            "safety_urgency": "high",
            "research_priority": "normal"
            # No drug_names field
        }

        validation = runner._validate_classifier(pharma_context, expected_classification)

        assert validation["domain_correct"] is True
        assert validation["safety_urgency_correct"] is True
        assert validation["research_priority_correct"] is True
        assert validation["drug_names_correct"] is None  # Not validated
        assert validation["overall_correct"] is True  # Should still pass


@pytest.mark.pharmaceutical
@pytest.mark.classifier
@pytest.mark.integration
@pytest.mark.fast
class TestClassifierValidationIntegration:
    """Test classifier validation integration into run_benchmark."""

    def test_run_benchmark_includes_classifier_validation(self, sample_benchmark_data):
        """Test that run_benchmark includes classifier validation in results."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        # Add expected_classification to sample data
        for query in sample_benchmark_data["queries"]:
            query["expected_classification"] = {
                "domain": "drug_interactions",
                "safety_urgency": "high",
                "research_priority": "normal",
                "drug_names": ["drug1", "drug2"],
            }

        # Mock the loader
        with patch.object(runner.loader, "load_benchmark", return_value=sample_benchmark_data):
            result = runner.run_benchmark("drug_interactions", version=1)

        # Verify query results have classifier validation
        assert len(result["query_results"]) > 0
        for query_result in result["query_results"]:
            if "error" not in query_result:
                assert "actual_classification" in query_result
                assert "classifier_validation" in query_result
                assert "overall_correct" in query_result["classifier_validation"]

        # Verify aggregate metrics include classifier validation
        assert "classifier_validation" in result["metrics"]
        assert "overall_accuracy" in result["metrics"]["classifier_validation"]
        assert "domain_accuracy" in result["metrics"]["classifier_validation"]
        assert "safety_urgency_accuracy" in result["metrics"]["classifier_validation"]
        assert "research_priority_accuracy" in result["metrics"]["classifier_validation"]

    def test_run_benchmark_mode_both_includes_classifier_validation(self, sample_benchmark_data):
        """Test that mode='both' includes classifier validation."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="both")

        # Add expected_classification to sample data
        for query in sample_benchmark_data["queries"]:
            query["expected_classification"] = {
                "domain": "drug_interactions",
                "safety_urgency": "high",
                "research_priority": "normal",
                "drug_names": ["drug1", "drug2"],
            }

        # Mock the loader
        with patch.object(runner.loader, "load_benchmark", return_value=sample_benchmark_data):
            result = runner.run_benchmark("drug_interactions", version=1)

        # Verify classifier validation in dual-mode results
        assert "classifier_validation" in result["metrics"]
        assert result["metadata"]["mode"] == "both"

    def test_regression_detection_includes_classifier_validation(self):
        """Test that regression detection checks classifier validation."""
        config = BenchmarkConfig()
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        # Mock benchmark result with low classifier accuracy
        benchmark_result = {
            "metadata": {"category": "drug_interactions", "version": 1, "mode": "cloud"},
            "metrics": {
                "average_accuracy": 0.85,
                "average_credits_per_query": 12.0,
                "average_latency_ms": 450.0,
                "classifier_validation": {"overall_accuracy": 0.88},  # Below 0.95 baseline by >5%
            },
        }

        baselines = {
            "cloud": {"average_accuracy": 0.85, "average_cost_per_query": 12.5, "average_latency_ms": 450.0},
            "classifier_validation": {"overall_accuracy": 0.95},  # Baseline
        }

        comparison = runner.compare_against_baselines(benchmark_result, baselines)

        # Should detect classifier regression
        assert comparison["has_regressions"] is True
        classifier_regressions = [
            r for r in comparison["regressions"] if r.get("type") == "classifier_validation_regression"
        ]
        assert len(classifier_regressions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
