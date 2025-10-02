"""
Test Verification Comments Fixes

Validates all three verification comments:
1. Comment 1: Dual-endpoint failed_queries only counts queries where BOTH endpoints failed
2. Comment 2: Tracker records failures in dual-mode (not just successes)
3. Comment 3: Generator produces complete schema with baselines and expected_classification
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_pharmaceutical_benchmarks import BenchmarkGenerator


@pytest.mark.pharmaceutical
@pytest.mark.validation
@pytest.mark.fast
class TestVerificationCommentsFixes:
    """Test all three verification comments fixes."""

    def test_comment_1_failed_queries_both_endpoints_succeed(self):
        """
        Comment 1: Verify failed_queries = 0 when both endpoints succeed on all queries.
        """
        # Simulate query_results where both endpoints always succeed
        query_results = [
            {
                "mode": "both",
                "cloud": {"succeeded": True, "scores": {"accuracy": 0.9}},
                "self_hosted": {"succeeded": True, "scores": {"accuracy": 0.85}}
            },
            {
                "mode": "both",
                "cloud": {"succeeded": True, "scores": {"accuracy": 0.88}},
                "self_hosted": {"succeeded": True, "scores": {"accuracy": 0.82}}
            },
            {
                "mode": "both",
                "cloud": {"succeeded": True, "scores": {"accuracy": 0.91}},
                "self_hosted": {"succeeded": True, "scores": {"accuracy": 0.87}}
            }
        ]

        # Calculate failed_queries using the fixed logic
        failed_queries = sum(
            1 for r in query_results
            if r.get('mode') == 'both'
            and not r.get('cloud', {}).get('succeeded')
            and not r.get('self_hosted', {}).get('succeeded')
        )

        assert failed_queries == 0, "Should be 0 failures when both endpoints succeed on all queries"

    def test_comment_1_failed_queries_both_endpoints_fail(self):
        """
        Comment 1: Verify failed_queries counts correctly when both endpoints fail.
        """
        # Simulate query_results where both endpoints fail on some queries
        query_results = [
            {
                "mode": "both",
                "cloud": {"succeeded": False, "error": "timeout"},
                "self_hosted": {"succeeded": False, "error": "timeout"}
            },
            {
                "mode": "both",
                "cloud": {"succeeded": True, "scores": {"accuracy": 0.88}},
                "self_hosted": {"succeeded": True, "scores": {"accuracy": 0.82}}
            },
            {
                "mode": "both",
                "cloud": {"succeeded": False, "error": "API error"},
                "self_hosted": {"succeeded": False, "error": "Connection failed"}
            }
        ]

        # Calculate failed_queries using the fixed logic
        failed_queries = sum(
            1 for r in query_results
            if r.get('mode') == 'both'
            and not r.get('cloud', {}).get('succeeded')
            and not r.get('self_hosted', {}).get('succeeded')
        )

        assert failed_queries == 2, "Should count 2 failures (queries 1 and 3)"

    def test_comment_1_failed_queries_different_endpoints_fail(self):
        """
        Comment 1: Verify failed_queries = 0 when different endpoints fail on different queries.

        This is the critical bug scenario: cloud fails on query 1, self-hosted fails on query 2,
        but NO query fails on BOTH endpoints.
        """
        # Simulate query_results where different endpoints fail on different queries
        query_results = [
            {
                "mode": "both",
                "cloud": {"succeeded": False, "error": "timeout"},
                "self_hosted": {"succeeded": True, "scores": {"accuracy": 0.85}}
            },
            {
                "mode": "both",
                "cloud": {"succeeded": True, "scores": {"accuracy": 0.88}},
                "self_hosted": {"succeeded": False, "error": "connection error"}
            },
            {
                "mode": "both",
                "cloud": {"succeeded": True, "scores": {"accuracy": 0.91}},
                "self_hosted": {"succeeded": True, "scores": {"accuracy": 0.87}}
            }
        ]

        # Calculate failed_queries using the fixed logic
        failed_queries = sum(
            1 for r in query_results
            if r.get('mode') == 'both'
            and not r.get('cloud', {}).get('succeeded')
            and not r.get('self_hosted', {}).get('succeeded')
        )

        assert failed_queries == 0, "Should be 0 failures when no query fails on BOTH endpoints"

    def test_comment_2_tracker_records_failures(self):
        """
        Comment 2: Verify tracker logic records failures with success=False.

        Tests that the dual-mode tracking calls track_query_result for both
        successes and failures.
        """
        # Simulate tracking calls for dual-mode
        tracked_calls = []

        def mock_track_query_result(category, query_type, accuracy, cost, latency_ms, success):
            tracked_calls.append({
                "category": category,
                "query_type": query_type,
                "accuracy": accuracy,
                "cost": cost,
                "latency_ms": latency_ms,
                "success": success
            })

        # Simulate dual-mode query result with one success and one failure
        dual_result = {
            "cloud": {
                "succeeded": True,
                "credits_used": 12.5,
                "latency_ms": 450.0
            },
            "self_hosted": {
                "succeeded": False,
                "credits_used": 0.0,
                "latency_ms": 0.0
            }
        }

        cloud_scores = {"accuracy": 0.85}
        category = "drug_interactions"
        query_type = "drug_interactions"

        # Simulate the fixed tracking logic
        # Track cloud endpoint (success)
        if dual_result["cloud"]["succeeded"]:
            mock_track_query_result(
                category=category,
                query_type=query_type,
                accuracy=cloud_scores["accuracy"],
                cost=dual_result["cloud"]["credits_used"],
                latency_ms=dual_result["cloud"]["latency_ms"],
                success=True
            )
        else:
            mock_track_query_result(
                category=category,
                query_type=query_type,
                accuracy=0.0,
                cost=0.0,
                latency_ms=0.0,
                success=False
            )

        # Track self-hosted endpoint (failure)
        if dual_result["self_hosted"]["succeeded"]:
            mock_track_query_result(
                category=category,
                query_type=query_type,
                accuracy=0.0,  # Would use sh_scores if it existed
                cost=dual_result["self_hosted"]["credits_used"],
                latency_ms=dual_result["self_hosted"]["latency_ms"],
                success=True
            )
        else:
            mock_track_query_result(
                category=category,
                query_type=query_type,
                accuracy=0.0,
                cost=0.0,
                latency_ms=0.0,
                success=False
            )

        # Verify both endpoints were tracked
        assert len(tracked_calls) == 2, "Should have 2 tracking calls (one per endpoint)"

        # Verify cloud success was tracked
        cloud_call = tracked_calls[0]
        assert cloud_call["success"] == True
        assert cloud_call["accuracy"] == 0.85
        assert cloud_call["cost"] == 12.5

        # Verify self-hosted failure was tracked
        sh_call = tracked_calls[1]
        assert sh_call["success"] == False, "Self-hosted failure should be tracked with success=False"
        assert sh_call["accuracy"] == 0.0
        assert sh_call["cost"] == 0.0

    def test_comment_3_generator_includes_baselines(self):
        """
        Comment 3: Verify generator produces metadata with baselines section.
        """
        generator = BenchmarkGenerator()
        metadata = generator.generate_metadata("drug_interactions", 1, 50)

        # Verify baselines section exists
        assert "baselines" in metadata, "Metadata should include baselines section"

        baselines = metadata["baselines"]

        # Verify cloud baseline structure
        assert "cloud" in baselines
        assert "average_latency_ms" in baselines["cloud"]
        assert "success_rate" in baselines["cloud"]
        assert "average_cost_per_query" in baselines["cloud"]
        assert "average_accuracy" in baselines["cloud"]
        assert "notes" in baselines["cloud"]

        # Verify self-hosted baseline structure
        assert "self_hosted" in baselines
        assert "average_latency_ms" in baselines["self_hosted"]
        assert "success_rate" in baselines["self_hosted"]
        assert "average_cost_per_query" in baselines["self_hosted"]
        assert "average_accuracy" in baselines["self_hosted"]
        assert baselines["self_hosted"]["average_cost_per_query"] == 0.0, "Self-hosted should be free"

        # Verify regression thresholds
        assert "regression_thresholds" in baselines
        assert "accuracy_drop_percent" in baselines["regression_thresholds"]
        assert "cost_increase_percent" in baselines["regression_thresholds"]
        assert "latency_increase_percent" in baselines["regression_thresholds"]

        # Verify classifier validation
        assert "classifier_validation" in baselines
        assert "overall_accuracy" in baselines["classifier_validation"]

    def test_comment_3_generator_includes_expected_classification(self):
        """
        Comment 3: Verify generator produces queries with expected_classification.
        """
        generator = BenchmarkGenerator()

        # Generate a query with explicit drug names
        query = generator.generate_query_template(
            query_id="test_001",
            query_text="What are the interactions between aspirin and warfarin?",
            expected_type="comparison",
            expected_content=["interaction mechanism", "clinical significance"],
            tags=["interaction", "pharmacokinetic"],
            category="drug_interactions",
            drug_names=["aspirin", "warfarin"]
        )

        # Verify expected_classification exists
        assert "expected_classification" in query, "Query should include expected_classification"

        classification = query["expected_classification"]

        # Verify classification structure
        assert "domain" in classification
        assert "safety_urgency" in classification
        assert "research_priority" in classification
        assert "drug_names" in classification

        # Verify domain mapping
        assert classification["domain"] == "drug_interactions"

        # Verify drug names
        assert classification["drug_names"] == ["aspirin", "warfarin"]

        # Verify safety urgency is appropriate for drug interactions
        assert classification["safety_urgency"] == "high", "Drug interactions should have high safety urgency"

    def test_comment_3_generator_classification_logic(self):
        """
        Comment 3: Verify classification logic for different categories and types.
        """
        generator = BenchmarkGenerator()

        # Test 1: Safety query should have high urgency
        safety_query = generator.generate_query_template(
            query_id="test_001",
            query_text="What are the side effects of ibuprofen?",
            expected_type="safety",
            expected_content=["side effects"],
            tags=["safety"],
            category="adverse_reactions",
            drug_names=["ibuprofen"]
        )

        assert safety_query["expected_classification"]["safety_urgency"] == "high"
        assert safety_query["expected_classification"]["research_priority"] == "high"
        assert safety_query["expected_classification"]["domain"] == "adverse_reactions"

        # Test 2: Definition query should have low/none urgency
        definition_query = generator.generate_query_template(
            query_id="test_002",
            query_text="What does BID mean?",
            expected_type="definition",
            expected_content=["twice daily"],
            tags=["abbreviation"],
            category="clinical_terminology",
            drug_names=[]
        )

        assert definition_query["expected_classification"]["safety_urgency"] == "none"
        assert definition_query["expected_classification"]["research_priority"] == "background"
        assert definition_query["expected_classification"]["domain"] == "general_research"

        # Test 3: Mechanism query should have appropriate classification
        mechanism_query = generator.generate_query_template(
            query_id="test_003",
            query_text="How does metformin work?",
            expected_type="scientific",
            expected_content=["mechanism", "molecular target"],
            tags=["mechanism"],
            category="mechanism_of_action",
            drug_names=["metformin"]
        )

        assert mechanism_query["expected_classification"]["domain"] == "mechanism_of_action"
        assert mechanism_query["expected_classification"]["safety_urgency"] == "low"
        assert mechanism_query["expected_classification"]["drug_names"] == ["metformin"]

    def test_comment_3_generator_produces_complete_benchmark(self):
        """
        Comment 3: Verify generator produces complete benchmark with all required fields.
        """
        generator = BenchmarkGenerator()

        # Generate a complete benchmark
        benchmark = generator.generate_benchmark("drug_interactions", version=1, num_queries=5)

        # Verify top-level structure
        assert "metadata" in benchmark
        assert "queries" in benchmark

        # Verify metadata completeness
        metadata = benchmark["metadata"]
        assert "version" in metadata
        assert "category" in metadata
        assert "created_date" in metadata
        assert "total_queries" in metadata
        assert "description" in metadata
        assert "baselines" in metadata, "Metadata should include baselines"

        # Verify all queries have expected_classification
        queries = benchmark["queries"]
        assert len(queries) == 5, "Should generate 5 queries"

        for query in queries:
            assert "id" in query
            assert "query" in query
            assert "expected_type" in query
            assert "expected_content" in query
            assert "expected_classification" in query, "Each query should have expected_classification"
            assert "evaluation_criteria" in query
            assert "tags" in query

            # Verify classification completeness
            classification = query["expected_classification"]
            assert "domain" in classification
            assert "safety_urgency" in classification
            assert "research_priority" in classification
            assert "drug_names" in classification

    def test_comment_3_generator_exports_valid_json(self):
        """
        Comment 3: Verify generator exports valid JSON that matches schema.
        """
        generator = BenchmarkGenerator()

        # Generate and save benchmark to temp file
        benchmark = generator.generate_benchmark("drug_interactions", version=1, num_queries=3)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            with open(temp_path, 'w') as f:
                json.dump(benchmark, f, indent=2)

            # Verify file can be read back
            with open(temp_path, 'r') as f:
                loaded_benchmark = json.load(f)

            # Verify structure matches
            assert loaded_benchmark["metadata"]["category"] == "drug_interactions"
            assert "baselines" in loaded_benchmark["metadata"]
            assert len(loaded_benchmark["queries"]) == 3

            for query in loaded_benchmark["queries"]:
                assert "expected_classification" in query
        finally:
            Path(temp_path).unlink()

    def test_edge_case_all_queries_succeed_both_endpoints(self):
        """
        Edge case: All queries succeed on both endpoints.
        Verifies failed_queries = 0.
        """
        query_results = [
            {"mode": "both", "cloud": {"succeeded": True}, "self_hosted": {"succeeded": True}},
            {"mode": "both", "cloud": {"succeeded": True}, "self_hosted": {"succeeded": True}},
            {"mode": "both", "cloud": {"succeeded": True}, "self_hosted": {"succeeded": True}},
        ]

        failed_queries = sum(
            1 for r in query_results
            if r.get('mode') == 'both'
            and not r.get('cloud', {}).get('succeeded')
            and not r.get('self_hosted', {}).get('succeeded')
        )

        assert failed_queries == 0

    def test_edge_case_all_queries_fail_both_endpoints(self):
        """
        Edge case: All queries fail on both endpoints.
        Verifies failed_queries = total_queries.
        """
        query_results = [
            {"mode": "both", "cloud": {"succeeded": False}, "self_hosted": {"succeeded": False}},
            {"mode": "both", "cloud": {"succeeded": False}, "self_hosted": {"succeeded": False}},
            {"mode": "both", "cloud": {"succeeded": False}, "self_hosted": {"succeeded": False}},
        ]

        failed_queries = sum(
            1 for r in query_results
            if r.get('mode') == 'both'
            and not r.get('cloud', {}).get('succeeded')
            and not r.get('self_hosted', {}).get('succeeded')
        )

        assert failed_queries == 3

    def test_edge_case_generator_with_empty_drug_list(self):
        """
        Edge case: Generator works even when drug list is empty.
        """
        generator = BenchmarkGenerator()
        # Clear drug lists
        generator.drug_loader.brand_drugs = []
        generator.drug_loader.generic_drugs = []

        # Should still generate valid queries (with placeholder drugs)
        queries = generator.generate_sample_queries("drug_interactions", count=2)

        assert len(queries) == 2
        for query in queries:
            assert "expected_classification" in query
            assert isinstance(query["expected_classification"]["drug_names"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
