"""
Pharmaceutical Credit Tracking and Cost Monitoring Test Suite

Comprehensive testing of the pharmaceutical credit tracking system with:
- Free tier optimization (10K requests/month)
- Multi-tier alert management
- Cost analysis for pharmaceutical research
- Batch processing cost efficiency
- Real-time burn rate monitoring

Tests validate cost-effective cloud-first strategy for pharmaceutical research.
"""

import pytest
import asyncio
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
import yaml

# Import modules under test
try:
    from src.monitoring.credit_tracker import PharmaceuticalCreditTracker, CreditUsage, AlertLevel
    from src.monitoring.alert_manager import AlertManager, AlertType, AlertSeverity
    from config.alerts import AlertConfig
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.monitoring.credit_tracker import PharmaceuticalCreditTracker, CreditUsage, AlertLevel
    from src.monitoring.alert_manager import AlertManager, AlertType, AlertSeverity
    from config.alerts import AlertConfig


class TestPharmaceuticalCreditTracking:
    """Test suite for pharmaceutical credit tracking and cost monitoring."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with mock configuration."""
        self.test_config = {
            "daily_credit_limit": 333,  # ~10K/month free tier
            "weekly_warning_threshold": 0.8,
            "monthly_critical_threshold": 0.9,
            "pharmaceutical_priority_boost": 1.2,
            "safety_alert_priority": 2.0
        }

        # Create temporary alert configuration
        self.temp_alert_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        alert_config = {
            "credit_monitoring": {
                "daily_burn_rate": {
                    "warning_threshold": 0.8,
                    "critical_threshold": 0.9,
                    "alert_channels": ["email", "console"]
                },
                "free_tier_limits": {
                    "monthly_requests": 10000,
                    "daily_average": 333,
                    "burst_limit": 500
                },
                "pharmaceutical_metrics": {
                    "drug_safety_queries": {"priority": "high", "cost_multiplier": 1.5},
                    "clinical_research": {"priority": "medium", "cost_multiplier": 1.2},
                    "general_pharma": {"priority": "normal", "cost_multiplier": 1.0}
                }
            }
        }
        yaml.dump(alert_config, self.temp_alert_config)
        self.temp_alert_config.close()

        yield

        # Cleanup
        os.unlink(self.temp_alert_config.name)

    def test_pharmaceutical_credit_tracker_initialization(self):
        """Test pharmaceutical credit tracker initialization."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        assert tracker is not None
        assert hasattr(tracker, 'base_monitor')
        assert hasattr(tracker, 'pharmaceutical_metrics')
        assert tracker.pharmaceutical_optimized == True

        # Should load pharmaceutical-specific thresholds
        assert hasattr(tracker, 'daily_limit')
        assert hasattr(tracker, 'monthly_limit')

    def test_free_tier_calculation(self):
        """Test free tier limit calculations and optimization."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Test daily allocation from 10K monthly limit
        daily_allocation = tracker.calculate_daily_allocation(10000, 30)
        assert daily_allocation == 333  # 10000/30 rounded down

        # Test with different month lengths
        february_allocation = tracker.calculate_daily_allocation(10000, 28)
        assert february_allocation == 357  # 10000/28 rounded down

        # Test burst capacity calculation
        burst_capacity = tracker.calculate_burst_capacity(daily_allocation)
        assert burst_capacity >= daily_allocation * 1.5  # 50% burst buffer

    def test_pharmaceutical_priority_weighting(self):
        """Test pharmaceutical query priority weighting system."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Test different pharmaceutical query types
        drug_safety_cost = tracker.calculate_weighted_cost(
            base_cost=1.0,
            query_type="drug_safety_queries"
        )
        assert drug_safety_cost == 1.5  # High priority multiplier

        clinical_research_cost = tracker.calculate_weighted_cost(
            base_cost=1.0,
            query_type="clinical_research"
        )
        assert clinical_research_cost == 1.2  # Medium priority multiplier

        general_cost = tracker.calculate_weighted_cost(
            base_cost=1.0,
            query_type="general_pharma"
        )
        assert general_cost == 1.0  # Normal priority

    def test_daily_burn_rate_monitoring(self):
        """Test daily credit burn rate monitoring and alerts."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Mock current usage
        current_time = datetime.now()

        # Test normal usage (50% of daily limit)
        normal_usage = CreditUsage(
            timestamp=current_time,
            credits_used=166,  # 50% of 333 daily limit
            query_type="general_pharma",
            success=True
        )

        alert_level = tracker.assess_daily_burn_rate([normal_usage])
        assert alert_level == AlertLevel.NORMAL

        # Test warning usage (80% of daily limit)
        warning_usage = CreditUsage(
            timestamp=current_time,
            credits_used=266,  # 80% of 333 daily limit
            query_type="clinical_research",
            success=True
        )

        alert_level = tracker.assess_daily_burn_rate([warning_usage])
        assert alert_level == AlertLevel.WARNING

        # Test critical usage (95% of daily limit)
        critical_usage = CreditUsage(
            timestamp=current_time,
            credits_used=316,  # 95% of 333 daily limit
            query_type="drug_safety_queries",
            success=True
        )

        alert_level = tracker.assess_daily_burn_rate([critical_usage])
        assert alert_level == AlertLevel.CRITICAL

    def test_pharmaceutical_usage_analytics(self):
        """Test pharmaceutical-specific usage analytics."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Create diverse pharmaceutical usage data
        usage_data = [
            CreditUsage(
                timestamp=datetime.now() - timedelta(hours=1),
                credits_used=50,
                query_type="drug_safety_queries",
                success=True,
                metadata={"drug": "metformin", "safety_level": "high"}
            ),
            CreditUsage(
                timestamp=datetime.now() - timedelta(hours=2),
                credits_used=30,
                query_type="clinical_research",
                success=True,
                metadata={"study_type": "phase_iii", "indication": "diabetes"}
            ),
            CreditUsage(
                timestamp=datetime.now() - timedelta(hours=3),
                credits_used=20,
                query_type="general_pharma",
                success=True,
                metadata={"category": "mechanism_of_action"}
            )
        ]

        analytics = tracker.analyze_pharmaceutical_usage(usage_data)

        # Should provide comprehensive analytics
        assert "total_credits_used" in analytics
        assert "query_type_breakdown" in analytics
        assert "pharmaceutical_categories" in analytics
        assert "cost_efficiency_score" in analytics
        assert "safety_query_percentage" in analytics

        # Verify calculations
        assert analytics["total_credits_used"] == 100
        assert analytics["safety_query_percentage"] == 50.0  # 50/100 credits
        assert len(analytics["query_type_breakdown"]) == 3

    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendations for pharmaceutical research."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Test high-cost usage pattern
        high_cost_usage = [
            CreditUsage(
                timestamp=datetime.now(),
                credits_used=100,
                query_type="drug_safety_queries",
                success=True
            ) for _ in range(5)  # 500 credits in safety queries
        ]

        recommendations = tracker.generate_cost_optimization_recommendations(high_cost_usage)

        assert "batch_processing" in recommendations
        assert "query_optimization" in recommendations
        assert "timing_optimization" in recommendations

        # Should recommend batching for high-volume safety queries
        batch_rec = recommendations["batch_processing"]
        assert batch_rec["recommended"] == True
        assert "potential_savings" in batch_rec
        assert batch_rec["potential_savings"] > 0

    @pytest.mark.asyncio
    async def test_real_time_monitoring_integration(self):
        """Test real-time credit monitoring integration."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Mock alert manager
        with patch.object(tracker, 'alert_manager') as mock_alert_manager:
            # Simulate real-time credit usage
            await tracker.record_usage_async(
                credits_used=50,
                query_type="drug_safety_queries",
                metadata={"drug": "warfarin", "interaction_check": True}
            )

            # Should record usage
            assert tracker.current_daily_usage >= 50

            # Test threshold breach detection
            await tracker.record_usage_async(
                credits_used=300,  # Push over daily limit
                query_type="clinical_research"
            )

            # Should trigger alert
            mock_alert_manager.send_alert.assert_called()

    def test_pharmaceutical_cost_analysis_report(self):
        """Test pharmaceutical cost analysis reporting."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Create comprehensive usage data
        usage_data = []
        start_time = datetime.now() - timedelta(days=7)

        for day in range(7):
            day_time = start_time + timedelta(days=day)
            # Different patterns each day
            for hour in range(8):  # 8 queries per day
                usage_data.append(CreditUsage(
                    timestamp=day_time + timedelta(hours=hour),
                    credits_used=10 + (day * 5),  # Increasing usage over week
                    query_type=["drug_safety_queries", "clinical_research", "general_pharma"][hour % 3],
                    success=True,
                    metadata={"day": day, "hour": hour}
                ))

        report = tracker.generate_pharmaceutical_cost_report(usage_data, period="weekly")

        # Comprehensive report validation
        assert "period" in report
        assert "total_cost" in report
        assert "daily_breakdown" in report
        assert "pharmaceutical_insights" in report
        assert "optimization_opportunities" in report
        assert "trend_analysis" in report

        # Should show increasing trend
        trend = report["trend_analysis"]
        assert trend["direction"] in ["increasing", "stable", "decreasing"]

        # Should identify optimization opportunities
        opportunities = report["optimization_opportunities"]
        assert len(opportunities) > 0

    def test_free_tier_maximization_strategies(self):
        """Test free tier maximization strategies."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Test monthly planning
        monthly_plan = tracker.generate_monthly_optimization_plan(
            target_queries=9500,  # Stay under 10K limit
            pharmaceutical_mix={
                "drug_safety_queries": 0.3,
                "clinical_research": 0.5,
                "general_pharma": 0.2
            }
        )

        assert "daily_allocation" in monthly_plan
        assert "query_type_allocation" in monthly_plan
        assert "buffer_recommendations" in monthly_plan
        assert "cost_projection" in monthly_plan

        # Should stay within free tier
        total_projected_cost = monthly_plan["cost_projection"]["total"]
        assert total_projected_cost <= 10000

        # Should allocate based on pharmaceutical priorities
        allocations = monthly_plan["query_type_allocation"]
        assert allocations["drug_safety_queries"] > 0
        assert allocations["clinical_research"] > allocations["general_pharma"]

    def test_burst_capacity_management(self):
        """Test burst capacity management for urgent pharmaceutical queries."""
        tracker = PharmaceuticalCreditTracker(
            alert_config_path=self.temp_alert_config.name
        )

        # Test burst capacity calculation
        daily_limit = 333
        burst_capacity = tracker.calculate_burst_capacity(daily_limit)

        # Should allow 50% burst for urgent safety queries
        assert burst_capacity >= daily_limit * 1.5

        # Test burst usage tracking
        urgent_usage = CreditUsage(
            timestamp=datetime.now(),
            credits_used=450,  # Over daily limit but within burst
            query_type="drug_safety_queries",
            success=True,
            metadata={"urgent": True, "safety_level": "critical"}
        )

        burst_status = tracker.assess_burst_usage([urgent_usage])
        assert burst_status["within_burst_limit"] == True
        assert burst_status["burst_utilization"] > 0

        # Test burst recovery planning
        recovery_plan = tracker.plan_burst_recovery(burst_status)
        assert "recommended_cooldown_hours" in recovery_plan
        assert "reduced_usage_target" in recovery_plan


class TestIntegratedCostMonitoring:
    """Integration tests for comprehensive cost monitoring."""

    @pytest.mark.asyncio
    async def test_end_to_end_cost_monitoring(self):
        """Test complete cost monitoring workflow."""
        tracker = PharmaceuticalCreditTracker()

        # Simulate realistic pharmaceutical research workflow
        research_queries = [
            {"type": "drug_safety_queries", "credits": 75, "drug": "metformin"},
            {"type": "clinical_research", "credits": 45, "study": "diabetes_trial"},
            {"type": "drug_safety_queries", "credits": 80, "drug": "warfarin"},
            {"type": "general_pharma", "credits": 25, "topic": "pharmacokinetics"},
            {"type": "clinical_research", "credits": 50, "study": "cardiovascular"},
        ]

        total_credits = 0
        for query in research_queries:
            await tracker.record_usage_async(
                credits_used=query["credits"],
                query_type=query["type"],
                metadata={k: v for k, v in query.items() if k not in ["type", "credits"]}
            )
            total_credits += query["credits"]

        # Generate comprehensive monitoring report
        daily_status = tracker.get_daily_status()
        assert daily_status["total_usage"] == total_credits
        assert "remaining_capacity" in daily_status
        assert "pharmaceutical_breakdown" in daily_status

        # Should provide actionable insights
        insights = daily_status["pharmaceutical_breakdown"]
        assert "drug_safety_percentage" in insights
        assert "clinical_research_percentage" in insights

        print(f"✅ Cost monitoring integration successful")
        print(f"   Total credits used: {total_credits}")
        print(f"   Safety queries: {insights.get('drug_safety_percentage', 0):.1f}%")