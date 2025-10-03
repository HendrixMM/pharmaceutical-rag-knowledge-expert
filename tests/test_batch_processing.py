"""
Batch Processing Optimization Test Suite

Comprehensive testing of the pharmaceutical batch processing system with:
- Free tier optimization through intelligent batching
- Pharmaceutical query prioritization
- Cost-efficient request scheduling
- Burst capacity management
- Real-time queue optimization

Tests validate batch processing for maximum free tier utilization.
"""
import asyncio
import time
from datetime import timedelta

import pytest

# Import modules under test
try:
    from src.optimization.batch_processor import (
        BatchRequest,
        BatchResponse,
        PharmaceuticalBatchProcessor,
        ProcessingPriority,
    )
    from src.optimization.queue_manager import QueueStrategy, RequestQueue
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from src.optimization.batch_processor import (
        BatchRequest,
        BatchResponse,
        PharmaceuticalBatchProcessor,
        ProcessingPriority,
    )
    from src.optimization.queue_manager import QueueStrategy, RequestQueue


class TestPharmaceuticalBatchProcessor:
    """Test suite for pharmaceutical batch processing optimization."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with batch processing configuration."""
        self.batch_config = {
            "max_batch_size": 25,  # NVIDIA Build typical batch limit
            "min_batch_size": 5,
            "batch_timeout_seconds": 30,
            "pharmaceutical_priorities": {
                "drug_safety_queries": 3,  # Highest priority
                "drug_interactions": 3,  # Highest priority
                "clinical_research": 2,  # Medium priority
                "general_pharma": 1,  # Normal priority
            },
            "cost_optimization": {
                "target_daily_usage": 333,  # Free tier daily allocation
                "burst_allowance": 0.5,  # 50% burst capacity
                "efficiency_threshold": 0.8,  # 80% efficiency target
            },
        }

        yield

    def test_batch_processor_initialization(self):
        """Test pharmaceutical batch processor initialization."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        assert processor is not None
        assert hasattr(processor, "pharmaceutical_optimized")
        assert processor.pharmaceutical_optimized == True
        assert hasattr(processor, "batch_queue")
        assert hasattr(processor, "priority_weights")

        # Should load pharmaceutical priority weights
        priorities = processor.priority_weights
        assert priorities["drug_safety_queries"] == 3
        assert priorities["clinical_research"] == 2

    def test_pharmaceutical_request_prioritization(self):
        """Test pharmaceutical request prioritization in batches."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Create diverse pharmaceutical requests
        requests = [
            BatchRequest(
                request_id="safety_1",
                query_type="drug_safety_queries",
                content="warfarin bleeding risk assessment",
                priority=ProcessingPriority.HIGH,
                metadata={"drug": "warfarin", "risk_category": "bleeding"},
            ),
            BatchRequest(
                request_id="general_1",
                query_type="general_pharma",
                content="basic pharmacology question",
                priority=ProcessingPriority.NORMAL,
                metadata={"category": "education"},
            ),
            BatchRequest(
                request_id="research_1",
                query_type="clinical_research",
                content="phase III diabetes trial data",
                priority=ProcessingPriority.MEDIUM,
                metadata={"phase": "iii", "indication": "diabetes"},
            ),
            BatchRequest(
                request_id="interaction_1",
                query_type="drug_interactions",
                content="ACE inhibitor potassium interaction",
                priority=ProcessingPriority.HIGH,
                metadata={"drug_class": "ace_inhibitor", "interaction_type": "electrolyte"},
            ),
        ]

        # Sort by pharmaceutical priorities
        sorted_requests = processor.prioritize_requests(requests)

        # High priority pharmaceutical queries should come first
        assert sorted_requests[0].query_type in ["drug_safety_queries", "drug_interactions"]
        assert sorted_requests[1].query_type in ["drug_safety_queries", "drug_interactions"]
        assert sorted_requests[2].query_type == "clinical_research"
        assert sorted_requests[3].query_type == "general_pharma"

    def test_optimal_batch_sizing(self):
        """Test optimal batch size calculation for cost efficiency."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Test various scenarios for optimal batch sizing

        # High-priority safety queries (smaller batches for faster processing)
        safety_requests = [
            BatchRequest(f"safety_{i}", "drug_safety_queries", f"safety query {i}", ProcessingPriority.HIGH)
            for i in range(15)
        ]

        optimal_size = processor.calculate_optimal_batch_size(
            requests=safety_requests,
            current_load=0.3,  # 30% current system load
            time_constraint=60,  # 60 second constraint
        )

        # Should use smaller batches for urgent safety queries
        assert optimal_size <= 10
        assert optimal_size >= self.batch_config["min_batch_size"]

        # General research queries (larger batches for efficiency)
        research_requests = [
            BatchRequest(f"research_{i}", "clinical_research", f"research query {i}", ProcessingPriority.MEDIUM)
            for i in range(30)
        ]

        optimal_size = processor.calculate_optimal_batch_size(
            requests=research_requests,
            current_load=0.1,  # 10% current system load
            time_constraint=300,  # 5 minute constraint
        )

        # Should use larger batches for research queries
        assert optimal_size >= 15
        assert optimal_size <= self.batch_config["max_batch_size"]

    def test_cost_efficient_batching_strategy(self):
        """Test cost-efficient batching strategy for free tier optimization."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Simulate daily usage tracking
        current_daily_usage = 250  # 75% of 333 daily limit used
        remaining_capacity = 83  # 25% remaining

        # Create batch requests
        pending_requests = [
            BatchRequest(f"req_{i}", "clinical_research", f"query {i}", ProcessingPriority.MEDIUM) for i in range(20)
        ]

        # Calculate cost-efficient strategy
        strategy = processor.calculate_cost_efficient_strategy(
            pending_requests=pending_requests,
            current_usage=current_daily_usage,
            daily_limit=333,
            time_remaining_today=timedelta(hours=6),
        )

        assert "recommended_batch_sizes" in strategy
        assert "processing_schedule" in strategy
        assert "cost_projection" in strategy

        # Should optimize for remaining capacity
        total_projected_usage = sum(strategy["recommended_batch_sizes"])
        assert total_projected_usage <= remaining_capacity

        # Should provide processing schedule
        schedule = strategy["processing_schedule"]
        assert len(schedule) > 0
        assert all("batch_size" in item and "delay_minutes" in item for item in schedule)

    @pytest.mark.asyncio
    async def test_real_time_batch_processing(self):
        """Test real-time batch processing with pharmaceutical prioritization."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Mock the actual processing function
        async def mock_process_batch(batch_requests):
            # Simulate processing delay based on batch size
            await asyncio.sleep(0.1 * len(batch_requests))
            return BatchResponse(
                batch_id=f"batch_{time.time()}",
                processed_count=len(batch_requests),
                success_count=len(batch_requests),
                processing_time_ms=100 * len(batch_requests),
                cost_credits=len(batch_requests) * 2,
            )

        processor.process_batch_async = mock_process_batch

        # Add diverse requests to the queue
        test_requests = [
            # Urgent drug safety queries
            BatchRequest("urgent_1", "drug_safety_queries", "warfarin toxicity assessment", ProcessingPriority.HIGH),
            BatchRequest("urgent_2", "drug_interactions", "insulin drug interactions", ProcessingPriority.HIGH),
            # Regular research queries
            BatchRequest("research_1", "clinical_research", "diabetes treatment outcomes", ProcessingPriority.MEDIUM),
            BatchRequest("research_2", "clinical_research", "cardiovascular risk factors", ProcessingPriority.MEDIUM),
            BatchRequest("research_3", "clinical_research", "hypertension management", ProcessingPriority.MEDIUM),
            # General pharmaceutical queries
            BatchRequest("general_1", "general_pharma", "basic pharmacology", ProcessingPriority.NORMAL),
            BatchRequest("general_2", "general_pharma", "drug classification", ProcessingPriority.NORMAL),
        ]

        # Process requests through the batch system
        for request in test_requests:
            processor.add_to_queue(request)

        # Execute batch processing
        batch_results = await processor.process_queued_batches()

        assert len(batch_results) > 0

        # Verify pharmaceutical prioritization in processing order
        first_batch = batch_results[0]
        assert first_batch.processed_count > 0

        # High-priority pharmaceutical queries should be processed first
        processed_ids = []
        for result in batch_results:
            if hasattr(result, "request_ids"):
                processed_ids.extend(result.request_ids)

        # Urgent requests should appear early in processing order
        urgent_positions = []
        for i, req_id in enumerate(processed_ids):
            if "urgent" in req_id:
                urgent_positions.append(i)

        if urgent_positions:
            avg_urgent_position = sum(urgent_positions) / len(urgent_positions)
            assert avg_urgent_position < len(processed_ids) / 2  # Should be in first half

    def test_queue_management_optimization(self):
        """Test queue management and optimization strategies."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Test queue strategies
        processor.queue_manager

        # Test FIFO strategy
        fifo_queue = RequestQueue(strategy=QueueStrategy.FIFO)
        requests = [
            BatchRequest("req_1", "general_pharma", "query 1", ProcessingPriority.NORMAL),
            BatchRequest("req_2", "drug_safety_queries", "query 2", ProcessingPriority.HIGH),
            BatchRequest("req_3", "clinical_research", "query 3", ProcessingPriority.MEDIUM),
        ]

        for req in requests:
            fifo_queue.add(req)

        fifo_order = [req.request_id for req in fifo_queue.get_batch(3)]
        assert fifo_order == ["req_1", "req_2", "req_3"]

        # Test PRIORITY strategy
        priority_queue = RequestQueue(strategy=QueueStrategy.PRIORITY)
        for req in requests:
            priority_queue.add(req)

        priority_order = [req.request_id for req in priority_queue.get_batch(3)]
        # High priority should come first
        assert priority_order[0] == "req_2"  # drug_safety_queries (HIGH)
        assert priority_order[1] == "req_3"  # clinical_research (MEDIUM)
        assert priority_order[2] == "req_1"  # general_pharma (NORMAL)

    def test_batch_performance_optimization(self):
        """Test batch performance optimization metrics."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Simulate batch processing results
        batch_results = [
            BatchResponse(
                batch_id="batch_1",
                processed_count=10,
                success_count=10,
                processing_time_ms=2000,  # 2 seconds
                cost_credits=20,
            ),
            BatchResponse(
                batch_id="batch_2",
                processed_count=15,
                success_count=14,  # One failure
                processing_time_ms=3500,  # 3.5 seconds
                cost_credits=30,
            ),
            BatchResponse(
                batch_id="batch_3",
                processed_count=8,
                success_count=8,
                processing_time_ms=1200,  # 1.2 seconds
                cost_credits=16,
            ),
        ]

        # Calculate performance metrics
        performance = processor.calculate_performance_metrics(batch_results)

        assert "average_batch_size" in performance
        assert "success_rate" in performance
        assert "average_processing_time_ms" in performance
        assert "cost_efficiency" in performance
        assert "requests_per_second" in performance

        # Verify calculations
        expected_avg_size = (10 + 15 + 8) / 3
        assert abs(performance["average_batch_size"] - expected_avg_size) < 0.1

        expected_success_rate = (10 + 14 + 8) / (10 + 15 + 8)
        assert abs(performance["success_rate"] - expected_success_rate) < 0.01

        # Cost efficiency should be credits per successful request
        expected_cost_efficiency = (20 + 30 + 16) / (10 + 14 + 8)
        assert abs(performance["cost_efficiency"] - expected_cost_efficiency) < 0.1

    def test_pharmaceutical_batch_analytics(self):
        """Test pharmaceutical-specific batch analytics."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Create batches with pharmaceutical metadata
        batch_history = []

        # Safety-focused batch
        safety_batch = BatchResponse(
            batch_id="safety_batch_1",
            processed_count=5,
            success_count=5,
            processing_time_ms=1500,
            cost_credits=15,
            metadata={
                "query_types": {"drug_safety_queries": 3, "drug_interactions": 2},
                "pharmaceutical_categories": ["cardiovascular", "endocrine"],
                "priority_distribution": {"high": 5, "medium": 0, "normal": 0},
            },
        )
        batch_history.append(safety_batch)

        # Research-focused batch
        research_batch = BatchResponse(
            batch_id="research_batch_1",
            processed_count=20,
            success_count=19,
            processing_time_ms=4000,
            cost_credits=40,
            metadata={
                "query_types": {"clinical_research": 15, "general_pharma": 5},
                "pharmaceutical_categories": ["oncology", "neurology", "cardiology"],
                "priority_distribution": {"high": 0, "medium": 15, "normal": 5},
            },
        )
        batch_history.append(research_batch)

        # Generate pharmaceutical analytics
        analytics = processor.generate_pharmaceutical_analytics(batch_history)

        assert "total_pharmaceutical_queries" in analytics
        assert "query_type_distribution" in analytics
        assert "pharmaceutical_category_breakdown" in analytics
        assert "priority_effectiveness" in analytics
        assert "cost_per_pharmaceutical_category" in analytics

        # Verify pharmaceutical-specific insights
        query_dist = analytics["query_type_distribution"]
        assert query_dist["drug_safety_queries"] == 3
        assert query_dist["clinical_research"] == 15

        category_breakdown = analytics["pharmaceutical_category_breakdown"]
        assert "cardiovascular" in category_breakdown
        assert "oncology" in category_breakdown

    def test_free_tier_optimization_strategies(self):
        """Test free tier optimization strategies for batch processing."""
        processor = PharmaceuticalBatchProcessor(config=self.batch_config, pharmaceutical_optimized=True)

        # Test daily optimization planning
        daily_plan = processor.optimize_for_free_tier(
            pending_requests_count=500,  # More requests than daily limit
            current_usage=100,  # Already used 100 credits
            daily_limit=333,  # Free tier daily limit
            hours_remaining=12,  # 12 hours left in day
        )

        assert "processing_strategy" in daily_plan
        assert "batch_schedule" in daily_plan
        assert "priority_allocation" in daily_plan
        assert "overflow_handling" in daily_plan

        # Should prioritize pharmaceutical queries within limit
        strategy = daily_plan["processing_strategy"]
        assert strategy["total_credits_planned"] <= (333 - 100)  # Within remaining limit

        priority_allocation = daily_plan["priority_allocation"]
        assert "drug_safety_queries" in priority_allocation
        assert priority_allocation["drug_safety_queries"] > priority_allocation.get("general_pharma", 0)

        # Test overflow handling for requests beyond daily limit
        overflow = daily_plan["overflow_handling"]
        assert "queue_for_tomorrow" in overflow
        assert "alternative_processing" in overflow


class TestIntegratedBatchOptimization:
    """Integration tests for complete batch optimization workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_pharmaceutical_batch_workflow(self):
        """Test complete pharmaceutical batch processing workflow."""

        config = {
            "max_batch_size": 20,
            "batch_timeout_seconds": 30,
            "pharmaceutical_priorities": {"drug_safety_queries": 3, "clinical_research": 2, "general_pharma": 1},
            "cost_optimization": {"target_daily_usage": 333, "efficiency_threshold": 0.8},
        }

        processor = PharmaceuticalBatchProcessor(config=config, pharmaceutical_optimized=True)

        # Mock the actual API processing
        async def mock_api_call(requests):
            await asyncio.sleep(0.1)  # Simulate API latency
            return {
                "results": [{"request_id": req.request_id, "success": True} for req in requests],
                "processing_time": 100,
                "cost": len(requests) * 2,
            }

        processor.api_call = mock_api_call

        # Create realistic pharmaceutical research workflow
        pharmaceutical_requests = [
            # Urgent drug safety queries
            BatchRequest(
                "safety_warfarin",
                "drug_safety_queries",
                "Warfarin interaction with NSAIDs safety assessment",
                ProcessingPriority.HIGH,
                metadata={"drug": "warfarin", "urgency": "high"},
            ),
            BatchRequest(
                "safety_metformin",
                "drug_safety_queries",
                "Metformin contraindications in kidney disease",
                ProcessingPriority.HIGH,
                metadata={"drug": "metformin", "contraindication": "renal"},
            ),
            # Clinical research queries
            BatchRequest(
                "research_diabetes_1",
                "clinical_research",
                "Phase III diabetes trial efficacy endpoints",
                ProcessingPriority.MEDIUM,
                metadata={"phase": "iii", "indication": "diabetes"},
            ),
            BatchRequest(
                "research_diabetes_2",
                "clinical_research",
                "Diabetes medication adherence studies systematic review",
                ProcessingPriority.MEDIUM,
                metadata={"study_type": "systematic_review", "indication": "diabetes"},
            ),
            BatchRequest(
                "research_cardio",
                "clinical_research",
                "Cardiovascular outcomes in hypertension treatment",
                ProcessingPriority.MEDIUM,
                metadata={"indication": "hypertension", "outcome": "cardiovascular"},
            ),
            # General pharmaceutical queries
            BatchRequest(
                "general_pk",
                "general_pharma",
                "Basic pharmacokinetics principles explanation",
                ProcessingPriority.NORMAL,
                metadata={"category": "education", "topic": "pharmacokinetics"},
            ),
            BatchRequest(
                "general_classification",
                "general_pharma",
                "Drug classification system overview",
                ProcessingPriority.NORMAL,
                metadata={"category": "education", "topic": "classification"},
            ),
        ]

        # Add all requests to the processor
        for request in pharmaceutical_requests:
            processor.add_to_queue(request)

        # Execute the complete batch processing workflow
        start_time = time.time()
        processing_results = await processor.process_all_queued_requests()
        end_time = time.time()

        # Validate workflow results
        assert len(processing_results) > 0

        total_processed = sum(result.processed_count for result in processing_results)
        total_successful = sum(result.success_count for result in processing_results)
        total_cost = sum(result.cost_credits for result in processing_results)

        # All requests should be processed
        assert total_processed == len(pharmaceutical_requests)

        # Success rate should be high
        success_rate = total_successful / total_processed if total_processed > 0 else 0
        assert success_rate >= 0.9  # 90%+ success rate

        # Cost should be reasonable
        assert total_cost <= len(pharmaceutical_requests) * 3  # Maximum 3 credits per request

        # Processing should be efficient
        processing_time = end_time - start_time
        assert processing_time < 10  # Should complete within 10 seconds

        # Generate final analytics
        workflow_analytics = processor.generate_workflow_analytics(processing_results)

        assert "pharmaceutical_efficiency" in workflow_analytics
        assert "cost_optimization_score" in workflow_analytics
        assert "priority_processing_accuracy" in workflow_analytics

        print("✅ End-to-end pharmaceutical batch workflow successful")
        print(f"   Requests processed: {total_processed}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Total cost: {total_cost} credits")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Pharmaceutical efficiency: {workflow_analytics.get('pharmaceutical_efficiency', 'N/A')}")

    def test_batch_cost_optimization_validation(self):
        """Test batch processing cost optimization validation."""
        processor = PharmaceuticalBatchProcessor(pharmaceutical_optimized=True)

        # Simulate month-long batch processing data
        monthly_batches = []
        daily_limits = 333

        for day in range(30):
            # Vary daily usage to simulate realistic patterns
            daily_usage = min(daily_limits, int(daily_limits * (0.6 + 0.4 * (day % 7) / 6)))

            # Create batch result for the day
            daily_batch = BatchResponse(
                batch_id=f"day_{day}_batch",
                processed_count=daily_usage // 2,  # Assume 2 credits per request
                success_count=int((daily_usage // 2) * 0.95),  # 95% success rate
                processing_time_ms=daily_usage * 50,  # 50ms per credit
                cost_credits=daily_usage,
            )
            monthly_batches.append(daily_batch)

        # Validate cost optimization over the month
        cost_analysis = processor.validate_cost_optimization(monthly_batches)

        assert "total_cost" in cost_analysis
        assert "average_daily_cost" in cost_analysis
        assert "free_tier_utilization" in cost_analysis
        assert "optimization_score" in cost_analysis

        # Should stay within free tier (10K per month)
        assert cost_analysis["total_cost"] <= 10000

        # Free tier utilization should be high but not excessive
        utilization = cost_analysis["free_tier_utilization"]
        assert 0.8 <= utilization <= 1.0  # 80-100% utilization

        # Optimization score should be good
        optimization_score = cost_analysis["optimization_score"]
        assert optimization_score >= 0.7  # 70%+ optimization score
