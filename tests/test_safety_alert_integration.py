"""
Safety Alert Integration Test Suite

Comprehensive testing of the pharmaceutical safety alert integration system with:
- Real-time drug safety monitoring
- Drug interaction detection and alerts
- Clinical contraindication warnings
- Integration with pharmaceutical query processing
- Automated escalation for critical safety issues

Tests validate safety alert system for pharmaceutical research protection.
"""
import json
import os
import tempfile
import time
from datetime import datetime, timedelta

import pytest

# Import modules under test
try:
    from src.pharmaceutical.contraindication_analyzer import ContraindicationRisk
    from src.pharmaceutical.drug_interaction_checker import InteractionSeverity
    from src.pharmaceutical.safety_alert_integration import (
        AlertTrigger,
        ContraindicationAlert,
        DrugSafetyAlert,
        InteractionAlert,
        PharmaceuticalSafetyMonitor,
        SafetyLevel,
        SafetyResponse,
    )
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from src.pharmaceutical.contraindication_analyzer import ContraindicationRisk
    from src.pharmaceutical.drug_interaction_checker import InteractionSeverity
    from src.pharmaceutical.safety_alert_integration import (
        AlertTrigger,
        ContraindicationAlert,
        DrugSafetyAlert,
        InteractionAlert,
        PharmaceuticalSafetyMonitor,
        SafetyLevel,
        SafetyResponse,
    )


class TestPharmaceuticalSafetyMonitor:
    """Test suite for pharmaceutical safety monitoring system."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with safety monitoring configuration."""
        self.safety_config = {
            "drug_interactions": {
                "severity_thresholds": {"critical": 0.9, "major": 0.7, "moderate": 0.5, "minor": 0.3},
                "high_risk_combinations": [
                    ["warfarin", "aspirin"],
                    ["metformin", "contrast_dye"],
                    ["ace_inhibitors", "potassium_supplements"],
                    ["digoxin", "amiodarone"],
                ],
            },
            "contraindications": {
                "absolute_contraindications": {
                    "metformin": ["severe_kidney_disease", "diabetic_ketoacidosis"],
                    "warfarin": ["active_bleeding", "pregnancy"],
                    "ace_inhibitors": ["pregnancy", "hyperkalemia", "bilateral_renal_artery_stenosis"],
                },
                "relative_contraindications": {
                    "metformin": ["mild_kidney_disease", "heart_failure"],
                    "statins": ["liver_disease", "myopathy_history"],
                },
            },
            "safety_monitoring": {
                "real_time_enabled": True,
                "alert_escalation_threshold": 3,
                "safety_check_mandatory_for": ["drug_safety_queries", "drug_interactions"],
                "automatic_contraindication_check": True,
            },
        }

        # Create temporary configuration file
        self.temp_config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(self.safety_config, self.temp_config_file)
        self.temp_config_file.close()

        yield

        # Cleanup
        os.unlink(self.temp_config_file.name)

    def test_safety_monitor_initialization(self):
        """Test pharmaceutical safety monitor initialization."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        assert monitor is not None
        assert hasattr(monitor, "pharmaceutical_optimized")
        assert monitor.pharmaceutical_optimized == True
        assert hasattr(monitor, "interaction_checker")
        assert hasattr(monitor, "contraindication_analyzer")
        assert hasattr(monitor, "alert_manager")

        # Should load safety configuration
        assert hasattr(monitor, "high_risk_combinations")
        assert hasattr(monitor, "contraindication_rules")

    def test_drug_interaction_detection(self):
        """Test drug interaction detection and alert generation."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test high-risk drug interaction detection
        high_risk_scenarios = [
            {
                "query": "Can I take warfarin and aspirin together for cardiovascular protection?",
                "expected_drugs": ["warfarin", "aspirin"],
                "expected_severity": InteractionSeverity.CRITICAL,
            },
            {
                "query": "Is it safe to use metformin before contrast dye imaging?",
                "expected_drugs": ["metformin", "contrast_dye"],
                "expected_severity": InteractionSeverity.MAJOR,
            },
            {
                "query": "ACE inhibitor with potassium supplement interaction risk",
                "expected_drugs": ["ace_inhibitors", "potassium_supplements"],
                "expected_severity": InteractionSeverity.MAJOR,
            },
        ]

        for scenario in high_risk_scenarios:
            safety_check = monitor.check_drug_interactions(scenario["query"])

            assert safety_check is not None
            assert isinstance(safety_check, InteractionAlert)
            assert safety_check.severity >= scenario["expected_severity"]
            assert safety_check.requires_immediate_attention == True

            # Should identify the interacting drugs
            detected_drugs = safety_check.interacting_drugs
            for expected_drug in scenario["expected_drugs"]:
                assert any(expected_drug.lower() in drug.lower() for drug in detected_drugs)

    def test_contraindication_analysis(self):
        """Test contraindication analysis and warning system."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test absolute contraindication detection
        absolute_contraindication_cases = [
            {
                "query": "Can I prescribe metformin to a patient with severe kidney disease?",
                "drug": "metformin",
                "condition": "severe_kidney_disease",
                "expected_risk": ContraindicationRisk.ABSOLUTE,
            },
            {
                "query": "Warfarin use during pregnancy safety considerations",
                "drug": "warfarin",
                "condition": "pregnancy",
                "expected_risk": ContraindicationRisk.ABSOLUTE,
            },
            {
                "query": "ACE inhibitors in pregnancy - is this safe?",
                "drug": "ace_inhibitors",
                "condition": "pregnancy",
                "expected_risk": ContraindicationRisk.ABSOLUTE,
            },
        ]

        for case in absolute_contraindication_cases:
            contraindication_check = monitor.analyze_contraindications(case["query"])

            assert contraindication_check is not None
            assert isinstance(contraindication_check, ContraindicationAlert)
            assert contraindication_check.risk_level >= case["expected_risk"]
            assert contraindication_check.requires_immediate_attention == True

            # Should identify the drug and contraindicated condition
            assert case["drug"] in contraindication_check.drug_name.lower()
            assert case["condition"] in contraindication_check.contraindicated_conditions

    def test_real_time_safety_monitoring(self):
        """Test real-time safety monitoring during query processing."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Mock alert manager for testing
        mock_alerts_sent = []

        def mock_send_alert(alert):
            mock_alerts_sent.append(alert)
            return True

        monitor.alert_manager.send_alert = mock_send_alert

        # Process pharmaceutical queries with safety monitoring
        safety_critical_queries = [
            "Patient on warfarin experiencing unusual bleeding - what should I check?",
            "Metformin patient scheduled for CT scan with contrast tomorrow",
            "Elderly patient on digoxin and amiodarone showing signs of toxicity",
            "Pregnant woman asking about ACE inhibitor safety",
        ]

        safety_responses = []
        for query in safety_critical_queries:
            response = monitor.process_query_with_safety_monitoring(query)
            safety_responses.append(response)

        # Validate safety monitoring results
        assert len(safety_responses) == len(safety_critical_queries)

        # Should trigger multiple safety alerts
        assert len(mock_alerts_sent) >= 2  # At least 2 critical scenarios

        # Each response should include safety assessment
        for response in safety_responses:
            assert isinstance(response, SafetyResponse)
            assert hasattr(response, "safety_level")
            assert hasattr(response, "alerts_triggered")
            assert hasattr(response, "recommended_actions")

        # Critical queries should have high safety levels
        high_safety_responses = [r for r in safety_responses if r.safety_level >= SafetyLevel.HIGH]
        assert len(high_safety_responses) >= 2

    def test_safety_alert_escalation(self):
        """Test safety alert escalation for critical pharmaceutical issues."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Track escalations
        escalations_triggered = []

        def mock_escalate(alert, escalation_level):
            escalations_triggered.append({"alert": alert, "level": escalation_level, "timestamp": datetime.now()})

        monitor.escalate_safety_alert = mock_escalate

        # Simulate critical safety scenarios that should trigger escalation
        critical_scenarios = [
            "Patient experiencing warfarin overdose symptoms - immediate help needed",
            "Severe allergic reaction to newly prescribed medication - emergency",
            "Multiple drug interactions causing patient hospitalization",
        ]

        for scenario in critical_scenarios:
            safety_check = monitor.comprehensive_safety_check(scenario)

            # Should identify as critical
            assert safety_check.safety_level == SafetyLevel.CRITICAL
            assert safety_check.requires_escalation == True

        # Should trigger escalations for critical cases
        assert len(escalations_triggered) >= len(critical_scenarios)

        # Validate escalation details
        for escalation in escalations_triggered:
            assert escalation["level"] in ["immediate", "urgent", "priority"]
            assert isinstance(escalation["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_integrated_safety_workflow(self):
        """Test integrated safety workflow with pharmaceutical query processing."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Complex pharmaceutical scenario requiring multiple safety checks
        complex_scenario = {
            "patient_context": {
                "age": 78,
                "conditions": ["diabetes", "heart_failure", "kidney_disease"],
                "current_medications": ["metformin", "digoxin", "furosemide"],
                "allergies": ["sulfa"],
            },
            "query": "What are the safety considerations for adding warfarin anticoagulation therapy to this patient's regimen?",
            "new_medication": "warfarin",
        }

        # Process through integrated safety workflow
        workflow_result = await monitor.process_integrated_safety_workflow(
            query=complex_scenario["query"],
            patient_context=complex_scenario["patient_context"],
            proposed_medication=complex_scenario["new_medication"],
        )

        # Should perform comprehensive safety analysis
        assert workflow_result is not None
        assert "interaction_analysis" in workflow_result
        assert "contraindication_analysis" in workflow_result
        assert "patient_specific_risks" in workflow_result
        assert "safety_recommendations" in workflow_result

        # Should identify multiple safety concerns
        interaction_analysis = workflow_result["interaction_analysis"]
        assert len(interaction_analysis["potential_interactions"]) > 0

        contraindication_analysis = workflow_result["contraindication_analysis"]
        assert len(contraindication_analysis["identified_risks"]) > 0

        # Should provide actionable safety recommendations
        safety_recommendations = workflow_result["safety_recommendations"]
        assert "monitoring_required" in safety_recommendations
        assert "dose_adjustments" in safety_recommendations
        assert "alternative_options" in safety_recommendations

    def test_drug_safety_database_integration(self):
        """Test integration with drug safety databases and knowledge sources."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test safety database queries for comprehensive information
        safety_queries = [
            {"drug": "warfarin", "safety_aspects": ["interactions", "contraindications", "monitoring", "reversal"]},
            {"drug": "metformin", "safety_aspects": ["contraindications", "adverse_effects", "special_populations"]},
            {"drug": "digoxin", "safety_aspects": ["interactions", "toxicity", "monitoring", "dosing"]},
        ]

        for query in safety_queries:
            safety_profile = monitor.get_comprehensive_safety_profile(query["drug"])

            assert safety_profile is not None
            assert "drug_name" in safety_profile
            assert safety_profile["drug_name"].lower() == query["drug"]

            # Should include requested safety aspects
            for aspect in query["safety_aspects"]:
                assert aspect in safety_profile or f"{aspect}_data" in safety_profile

            # Should include evidence-based information
            assert "evidence_level" in safety_profile
            assert "last_updated" in safety_profile
            assert "references" in safety_profile

    def test_safety_alert_suppression_and_filtering(self):
        """Test safety alert suppression and filtering to prevent alert fatigue."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Test duplicate alert suppression
        duplicate_query = "warfarin aspirin interaction risk assessment"

        # First alert should be sent
        first_alert = monitor.check_drug_interactions(duplicate_query)
        assert first_alert is not None
        should_send_first = monitor.should_send_alert(first_alert)
        assert should_send_first == True

        # Duplicate alert should be suppressed
        second_alert = monitor.check_drug_interactions(duplicate_query)
        should_send_second = monitor.should_send_alert(second_alert)
        assert should_send_second == False

        # Test severity-based filtering
        low_severity_alert = DrugSafetyAlert(
            drug_name="vitamin_d",
            safety_concern="mild_gastrointestinal_upset",
            severity=SafetyLevel.LOW,
            alert_trigger=AlertTrigger.ROUTINE_MONITORING,
        )

        # Low severity alerts might be filtered in high-volume scenarios
        should_send_low = monitor.should_send_alert(low_severity_alert, volume_threshold=10)
        # Behavior depends on current alert volume

        # Critical alerts should never be suppressed
        critical_alert = DrugSafetyAlert(
            drug_name="warfarin",
            safety_concern="major_bleeding_risk",
            severity=SafetyLevel.CRITICAL,
            alert_trigger=AlertTrigger.IMMEDIATE_ATTENTION,
        )

        should_send_critical = monitor.should_send_alert(critical_alert, volume_threshold=100)
        assert should_send_critical == True  # Critical always sent

    def test_pharmaceutical_safety_analytics(self):
        """Test pharmaceutical safety analytics and reporting."""
        monitor = PharmaceuticalSafetyMonitor(config_path=self.temp_config_file.name, pharmaceutical_optimized=True)

        # Simulate safety monitoring over time period
        safety_events = []

        # Drug interaction events
        for i in range(15):
            event = {
                "timestamp": datetime.now() - timedelta(hours=i),
                "type": "drug_interaction",
                "severity": ["minor", "moderate", "major", "critical"][i % 4],
                "drugs": ["warfarin_aspirin", "metformin_contrast", "digoxin_amiodarone"][i % 3],
                "outcome": "detected" if i % 3 != 0 else "prevented",
            }
            safety_events.append(event)

        # Contraindication events
        for i in range(10):
            event = {
                "timestamp": datetime.now() - timedelta(hours=i * 2),
                "type": "contraindication",
                "severity": ["moderate", "major", "critical"][i % 3],
                "drug": ["metformin", "warfarin", "ace_inhibitor"][i % 3],
                "condition": ["kidney_disease", "pregnancy", "hyperkalemia"][i % 3],
                "outcome": "prevented",
            }
            safety_events.append(event)

        # Generate safety analytics
        analytics = monitor.generate_safety_analytics(safety_events, period="24_hours")

        assert "total_safety_events" in analytics
        assert analytics["total_safety_events"] == len(safety_events)

        assert "event_type_distribution" in analytics
        event_dist = analytics["event_type_distribution"]
        assert "drug_interaction" in event_dist
        assert "contraindication" in event_dist

        assert "severity_breakdown" in analytics
        severity_breakdown = analytics["severity_breakdown"]
        assert "critical" in severity_breakdown
        assert "major" in severity_breakdown

        assert "prevention_rate" in analytics
        # Should calculate how many adverse events were prevented

        assert "top_safety_concerns" in analytics
        top_concerns = analytics["top_safety_concerns"]
        assert len(top_concerns) > 0


class TestIntegratedSafetyWorkflows:
    """Integration tests for complete pharmaceutical safety workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_safety_monitoring_workflow(self):
        """Test complete pharmaceutical safety monitoring workflow."""

        # Create comprehensive safety configuration
        safety_config = {
            "drug_interactions": {
                "severity_thresholds": {"critical": 0.9, "major": 0.7, "moderate": 0.5},
                "high_risk_combinations": [["warfarin", "aspirin"], ["digoxin", "amiodarone"]],
            },
            "contraindications": {
                "absolute_contraindications": {
                    "metformin": ["severe_kidney_disease"],
                    "warfarin": ["active_bleeding", "pregnancy"],
                }
            },
            "safety_monitoring": {
                "real_time_enabled": True,
                "alert_escalation_threshold": 2,
                "automatic_contraindication_check": True,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
            json.dump(safety_config, config_file)
            config_path = config_file.name

        try:
            monitor = PharmaceuticalSafetyMonitor(config_path=config_path, pharmaceutical_optimized=True)

            # Track all safety events
            safety_events = []

            def track_safety_event(event):
                safety_events.append({"timestamp": datetime.now(), "event": event, "type": type(event).__name__})

            monitor.on_safety_event = track_safety_event

            # Comprehensive pharmaceutical safety scenario
            complex_safety_scenarios = [
                {
                    "query": "Elderly patient on warfarin and digoxin wants to start aspirin for heart protection",
                    "expected_alerts": ["drug_interaction", "multiple_anticoagulants"],
                },
                {
                    "query": "Pregnant woman with diabetes asking about metformin safety",
                    "expected_alerts": ["pregnancy_contraindication"],
                },
                {
                    "query": "Patient with severe kidney disease prescribed metformin for diabetes",
                    "expected_alerts": ["absolute_contraindication", "renal_function"],
                },
                {
                    "query": "Digoxin toxicity symptoms in patient also taking amiodarone",
                    "expected_alerts": ["drug_interaction", "toxicity_risk", "immediate_attention"],
                },
            ]

            workflow_results = []

            for scenario in complex_safety_scenarios:
                start_time = time.time()

                # Process query through complete safety workflow
                safety_response = await monitor.comprehensive_safety_analysis(
                    query=scenario["query"],
                    enable_real_time_monitoring=True,
                    perform_interaction_check=True,
                    perform_contraindication_check=True,
                    generate_recommendations=True,
                )

                end_time = time.time()
                processing_time = end_time - start_time

                workflow_results.append(
                    {"scenario": scenario, "response": safety_response, "processing_time": processing_time}
                )

            # Validate comprehensive safety workflow

            # All scenarios should generate safety responses
            assert len(workflow_results) == len(complex_safety_scenarios)

            # Each response should include comprehensive safety analysis
            for result in workflow_results:
                response = result["response"]
                assert "safety_level" in response
                assert "detected_issues" in response
                assert "recommendations" in response
                assert "escalation_required" in response

                # High-risk scenarios should trigger appropriate alerts
                if "warfarin" in result["scenario"]["query"] or "pregnancy" in result["scenario"]["query"]:
                    assert response["safety_level"] >= SafetyLevel.HIGH

            # Should detect expected safety issues
            all_detected_issues = []
            for result in workflow_results:
                all_detected_issues.extend(result["response"]["detected_issues"])

            # Should identify drug interactions
            interaction_detected = any("interaction" in issue.lower() for issue in all_detected_issues)
            assert interaction_detected == True

            # Should identify contraindications
            contraindication_detected = any("contraindication" in issue.lower() for issue in all_detected_issues)
            assert contraindication_detected == True

            # Performance should be reasonable
            avg_processing_time = sum(r["processing_time"] for r in workflow_results) / len(workflow_results)
            assert avg_processing_time < 2.0  # Under 2 seconds per comprehensive analysis

            # Generate final safety report
            safety_report = monitor.generate_comprehensive_safety_report(workflow_results)

            assert "total_scenarios_analyzed" in safety_report
            assert "safety_issues_detected" in safety_report
            assert "alert_distribution" in safety_report
            assert "response_time_metrics" in safety_report
            assert "pharmaceutical_insights" in safety_report

            print("✅ End-to-end safety monitoring workflow successful")
            print(f"   Scenarios analyzed: {len(workflow_results)}")
            print(f"   Safety issues detected: {len(all_detected_issues)}")
            print(f"   Average processing time: {avg_processing_time:.2f} seconds")
            print(f"   Safety events tracked: {len(safety_events)}")

        finally:
            os.unlink(config_path)

    def test_safety_monitoring_cost_integration(self):
        """Test safety monitoring integration with cost optimization systems."""
        monitor = PharmaceuticalSafetyMonitor(pharmaceutical_optimized=True)

        # Test cost-aware safety monitoring
        safety_scenarios_with_cost = [
            {
                "query": "emergency warfarin reversal protocol",
                "urgency": "critical",
                "expected_cost_tier": "premium",
                "expected_processing_priority": 5,
            },
            {
                "query": "routine drug interaction check for new prescription",
                "urgency": "standard",
                "expected_cost_tier": "standard",
                "expected_processing_priority": 2,
            },
            {
                "query": "general medication safety information",
                "urgency": "low",
                "expected_cost_tier": "low",
                "expected_processing_priority": 1,
            },
        ]

        cost_optimization_results = []

        for scenario in safety_scenarios_with_cost:
            cost_analysis = monitor.analyze_safety_query_cost(query=scenario["query"], urgency=scenario["urgency"])

            cost_optimization_results.append({"scenario": scenario, "cost_analysis": cost_analysis})

        # Validate cost-aware safety processing

        for result in cost_optimization_results:
            scenario = result["scenario"]
            analysis = result["cost_analysis"]

            # Should match expected cost tier
            assert analysis["cost_tier"] == scenario["expected_cost_tier"]

            # Should prioritize based on safety urgency
            assert analysis["processing_priority"] >= scenario["expected_processing_priority"]

            # Critical safety queries should bypass cost optimization
            if scenario["urgency"] == "critical":
                assert analysis["bypass_cost_optimization"] == True
                assert analysis["immediate_processing"] == True
