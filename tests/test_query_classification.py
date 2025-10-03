"""
Query Classification System Test Suite

Comprehensive testing of the pharmaceutical query classification system with:
- Multi-level pharmaceutical domain classification
- Drug safety priority escalation
- Clinical research categorization
- Cost-aware query routing
- Real-time classification accuracy

Tests validate intelligent query classification for pharmaceutical research optimization.
"""
import time

import pytest

# Import modules under test
try:
    from src.pharmaceutical.query_classifier import (
        ClassificationConfidence,
        PharmaceuticalDomain,
        PharmaceuticalQueryClassifier,
        ResearchPriority,
        SafetyUrgency,
    )
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from src.pharmaceutical.query_classifier import (
        ClassificationConfidence,
        PharmaceuticalDomain,
        PharmaceuticalQueryClassifier,
        ResearchPriority,
        SafetyUrgency,
    )


class TestPharmaceuticalQueryClassifier:
    """Test suite for pharmaceutical query classification system."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with classification models and ontology."""
        self.classifier_config = {
            "pharmaceutical_domains": {
                "drug_safety": {
                    "keywords": ["safety", "adverse", "toxicity", "contraindication", "warning"],
                    "priority_weight": 3.0,
                    "urgency_threshold": 0.7,
                },
                "drug_interactions": {
                    "keywords": ["interaction", "contraindication", "combination", "concurrent"],
                    "priority_weight": 3.0,
                    "urgency_threshold": 0.8,
                },
                "clinical_research": {
                    "keywords": ["trial", "study", "efficacy", "phase", "clinical"],
                    "priority_weight": 2.0,
                    "urgency_threshold": 0.5,
                },
                "pharmacokinetics": {
                    "keywords": ["absorption", "distribution", "metabolism", "excretion", "clearance"],
                    "priority_weight": 1.5,
                    "urgency_threshold": 0.4,
                },
                "pharmacodynamics": {
                    "keywords": ["mechanism", "action", "receptor", "target", "pathway"],
                    "priority_weight": 1.5,
                    "urgency_threshold": 0.4,
                },
                "general_pharma": {
                    "keywords": ["drug", "medication", "pharmaceutical", "therapy"],
                    "priority_weight": 1.0,
                    "urgency_threshold": 0.3,
                },
            }
        }

        yield

    def test_classifier_initialization(self):
        """Test pharmaceutical query classifier initialization."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        assert classifier is not None
        assert hasattr(classifier, "pharmaceutical_optimized")
        assert classifier.pharmaceutical_optimized == True
        assert hasattr(classifier, "domain_keywords")
        assert hasattr(classifier, "safety_analyzer")

        # Should load domain-specific configurations
        assert "drug_safety" in classifier.domain_keywords
        assert "clinical_research" in classifier.domain_keywords

    def test_drug_safety_query_classification(self):
        """Test classification of drug safety queries with proper urgency assessment."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # High-urgency drug safety queries
        critical_safety_queries = [
            "What are the contraindications for warfarin in patients with active bleeding?",
            "Metformin toxicity symptoms in kidney disease patients",
            "Serious adverse reactions to ACE inhibitors in elderly patients",
            "Drug safety alert: new black box warning for diabetes medication",
        ]

        for query in critical_safety_queries:
            classification = classifier.classify_query(query)

            assert classification.primary_domain == PharmaceuticalDomain.DRUG_SAFETY
            assert classification.safety_urgency >= SafetyUrgency.HIGH
            assert classification.confidence >= ClassificationConfidence.HIGH
            assert classification.requires_immediate_attention == True

            # Should identify specific safety elements
            assert "safety_keywords" in classification.metadata
            assert len(classification.metadata["safety_keywords"]) > 0

    def test_drug_interaction_classification(self):
        """Test classification of drug interaction queries."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        interaction_queries = [
            "Drug interactions between warfarin and NSAIDs",
            "Can I take metformin with ACE inhibitors?",
            "Contraindications for concurrent use of statins and fibrates",
            "Dangerous drug combinations with MAO inhibitors",
        ]

        for query in interaction_queries:
            classification = classifier.classify_query(query)

            assert classification.primary_domain == PharmaceuticalDomain.DRUG_INTERACTIONS
            assert classification.safety_urgency >= SafetyUrgency.MEDIUM
            assert classification.requires_safety_check == True

            # Should identify interacting drugs
            metadata = classification.metadata
            assert "potential_drugs" in metadata
            assert len(metadata["potential_drugs"]) >= 1

    def test_clinical_research_classification(self):
        """Test classification of clinical research queries."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        research_queries = [
            "Phase III clinical trial results for diabetes medications",
            "Systematic review of cardiovascular outcomes with ACE inhibitors",
            "Meta-analysis of statin efficacy in primary prevention",
            "Randomized controlled trial design for hypertension treatment",
        ]

        for query in research_queries:
            classification = classifier.classify_query(query)

            assert classification.primary_domain == PharmaceuticalDomain.CLINICAL_RESEARCH
            assert classification.research_priority >= ResearchPriority.MEDIUM
            assert classification.safety_urgency <= SafetyUrgency.MEDIUM  # Research less urgent than safety

            # Should identify research elements
            metadata = classification.metadata
            assert "research_type" in metadata
            assert "study_phase" in metadata or "research_method" in metadata

    def test_pharmacokinetic_classification(self):
        """Test classification of pharmacokinetic queries."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        pk_queries = [
            "Metformin absorption in gastrointestinal disorders",
            "Warfarin metabolism through CYP2C9 pathway",
            "Renal clearance of ACE inhibitors in kidney disease",
            "First-pass metabolism effects on drug bioavailability",
        ]

        for query in pk_queries:
            classification = classifier.classify_query(query)

            assert classification.primary_domain == PharmaceuticalDomain.PHARMACOKINETICS
            assert classification.research_priority >= ResearchPriority.LOW
            assert classification.technical_complexity >= 0.6

            # Should identify PK-specific elements
            metadata = classification.metadata
            assert "pk_processes" in metadata
            pk_processes = metadata["pk_processes"]
            assert any(process in ["absorption", "distribution", "metabolism", "excretion"] for process in pk_processes)

    def test_multi_domain_query_classification(self):
        """Test classification of queries spanning multiple pharmaceutical domains."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # Complex query combining safety, interactions, and pharmacokinetics
        complex_query = (
            "What are the safety considerations for warfarin drug interactions "
            "in patients with impaired metabolism through CYP2C9 pathway, "
            "and what clinical trial evidence supports dose adjustments?"
        )

        classification = classifier.classify_query(complex_query)

        # Should identify primary domain (likely drug safety due to high priority)
        assert classification.primary_domain == PharmaceuticalDomain.DRUG_SAFETY

        # Should identify secondary domains
        assert hasattr(classification, "secondary_domains")
        secondary = classification.secondary_domains
        assert PharmaceuticalDomain.DRUG_INTERACTIONS in secondary
        assert PharmaceuticalDomain.PHARMACOKINETICS in secondary

        # Should have high confidence due to multiple domain indicators
        assert classification.confidence >= ClassificationConfidence.HIGH

        # Should require safety attention due to drug safety component
        assert classification.requires_immediate_attention == True

    def test_confidence_scoring_accuracy(self):
        """Test accuracy of confidence scoring for different query types."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # High confidence queries (clear domain indicators)
        high_confidence_queries = [
            ("warfarin bleeding contraindications", PharmaceuticalDomain.DRUG_SAFETY),
            ("phase III diabetes trial results", PharmaceuticalDomain.CLINICAL_RESEARCH),
            ("metformin renal clearance", PharmaceuticalDomain.PHARMACOKINETICS),
        ]

        for query_text, expected_domain in high_confidence_queries:
            classification = classifier.classify_query(query_text)

            assert classification.primary_domain == expected_domain
            assert classification.confidence >= ClassificationConfidence.HIGH
            assert classification.confidence_score >= 0.8

        # Medium confidence queries (some ambiguity)
        medium_confidence_queries = [
            "diabetes medication side effects",  # Could be safety or general
            "blood pressure drug effectiveness",  # Could be research or clinical
        ]

        for query_text in medium_confidence_queries:
            classification = classifier.classify_query(query_text)

            assert ClassificationConfidence.MEDIUM <= classification.confidence <= ClassificationConfidence.HIGH
            assert 0.5 <= classification.confidence_score <= 0.8

        # Low confidence queries (ambiguous or general)
        low_confidence_queries = ["tell me about medicine", "pharmaceutical industry overview", "what is a drug?"]

        for query_text in low_confidence_queries:
            classification = classifier.classify_query(query_text)

            assert classification.confidence <= ClassificationConfidence.MEDIUM
            assert classification.confidence_score <= 0.6

    def test_safety_urgency_escalation(self):
        """Test safety urgency escalation logic."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # Critical urgency scenarios
        critical_scenarios = [
            "patient experiencing warfarin overdose symptoms",
            "immediate contraindications for emergency medication",
            "severe allergic reaction to prescribed drug",
            "toxic drug interaction causing hospitalization",
        ]

        for scenario in critical_scenarios:
            classification = classifier.classify_query(scenario)

            assert classification.safety_urgency == SafetyUrgency.CRITICAL
            assert classification.requires_immediate_attention == True
            assert classification.escalation_required == True

        # Medium urgency scenarios
        medium_scenarios = [
            "potential drug interactions with new prescription",
            "side effects monitoring for chronic medication",
            "dose adjustment considerations for elderly patient",
        ]

        for scenario in medium_scenarios:
            classification = classifier.classify_query(scenario)

            assert classification.safety_urgency >= SafetyUrgency.MEDIUM
            assert classification.requires_safety_check == True

    def test_cost_aware_classification(self):
        """Test cost-aware classification for query routing."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # High-cost queries (complex, multi-domain)
        high_cost_queries = [
            "comprehensive drug interaction analysis for polypharmacy patient with cardiovascular, diabetes, and kidney disease",
            "systematic review of all clinical trials for novel diabetes medications with cardiovascular outcomes",
            "detailed pharmacokinetic modeling for personalized dosing in special populations",
        ]

        for query in high_cost_queries:
            classification = classifier.classify_query(query)

            assert hasattr(classification, "estimated_cost_tier")
            assert classification.estimated_cost_tier in ["high", "premium"]
            assert classification.processing_complexity >= 0.8

        # Low-cost queries (simple, single domain)
        low_cost_queries = ["what is metformin?", "basic mechanism of ACE inhibitors", "common side effects of statins"]

        for query in low_cost_queries:
            classification = classifier.classify_query(query)

            assert classification.estimated_cost_tier in ["low", "standard"]
            assert classification.processing_complexity <= 0.5

    @pytest.mark.asyncio
    async def test_real_time_classification_performance(self):
        """Test real-time classification performance and accuracy."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # Test batch classification performance
        test_queries = [
            "warfarin bleeding risk assessment",
            "metformin contraindications in kidney disease",
            "phase III diabetes clinical trial outcomes",
            "ACE inhibitor mechanism of action",
            "drug interactions with NSAIDs",
            "cardiovascular safety of diabetes medications",
            "pharmacokinetics of statins in elderly",
            "adverse reactions to beta blockers",
            "clinical efficacy of hypertension treatments",
            "drug metabolism through liver enzymes",
        ]

        # Measure classification performance
        start_time = time.time()
        classifications = []

        for query in test_queries:
            classification = await classifier.classify_query_async(query)
            classifications.append(classification)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance validation
        assert len(classifications) == len(test_queries)
        assert total_time < 5.0  # Should classify 10 queries in under 5 seconds

        # Accuracy validation
        high_confidence_count = sum(1 for c in classifications if c.confidence >= ClassificationConfidence.HIGH)
        assert high_confidence_count >= len(test_queries) * 0.7  # 70%+ high confidence

        # Domain distribution validation
        domain_counts = {}
        for classification in classifications:
            domain = classification.primary_domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Should have reasonable domain distribution
        assert len(domain_counts) >= 3  # At least 3 different domains identified

    def test_pharmaceutical_ontology_integration(self):
        """Test integration with pharmaceutical ontology for enhanced classification."""
        classifier = PharmaceuticalQueryClassifier(config=self.classifier_config, pharmaceutical_optimized=True)

        # Test ontology-enhanced classification
        ontology_queries = [
            "ACE inhibitor induced angioedema in African American patients",  # Specific drug class + ethnic consideration
            "SGLT-2 inhibitor cardiovascular benefits in heart failure patients",  # Novel drug class + indication
            "CYP2D6 polymorphism effects on antidepressant metabolism",  # Pharmacogenomics
        ]

        for query in ontology_queries:
            classification = classifier.classify_query(query)

            # Should use ontology for enhanced understanding
            assert hasattr(classification, "ontology_matches")
            ontology_matches = classification.ontology_matches

            assert "drug_classes" in ontology_matches or "specific_drugs" in ontology_matches
            assert "indications" in ontology_matches or "conditions" in ontology_matches

            # Should have higher confidence with ontology support
            assert classification.confidence >= ClassificationConfidence.MEDIUM


class TestIntegratedQueryClassification:
    """Integration tests for complete query classification workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_classification_workflow(self):
        """Test complete query classification workflow for pharmaceutical research."""

        classifier = PharmaceuticalQueryClassifier(pharmaceutical_optimized=True)

        # Realistic pharmaceutical research workflow
        research_workflow = [
            {
                "query": "What are the contraindications for metformin in patients with chronic kidney disease?",
                "expected_domain": PharmaceuticalDomain.DRUG_SAFETY,
                "expected_urgency": SafetyUrgency.HIGH,
            },
            {
                "query": "Phase III clinical trial results for SGLT-2 inhibitors in cardiovascular outcomes",
                "expected_domain": PharmaceuticalDomain.CLINICAL_RESEARCH,
                "expected_urgency": SafetyUrgency.LOW,
            },
            {
                "query": "Drug interactions between warfarin and commonly prescribed antibiotics",
                "expected_domain": PharmaceuticalDomain.DRUG_INTERACTIONS,
                "expected_urgency": SafetyUrgency.HIGH,
            },
            {
                "query": "Pharmacokinetics of ACE inhibitors in elderly patients with multiple comorbidities",
                "expected_domain": PharmaceuticalDomain.PHARMACOKINETICS,
                "expected_urgency": SafetyUrgency.MEDIUM,
            },
        ]

        classification_results = []
        processing_metrics = {
            "total_time": 0,
            "accuracy_score": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }

        start_time = time.time()

        for item in research_workflow:
            query_start = time.time()
            classification = await classifier.classify_query_async(item["query"])
            query_end = time.time()

            classification_results.append(
                {
                    "query": item["query"],
                    "classification": classification,
                    "expected": item,
                    "processing_time": query_end - query_start,
                }
            )

            # Check accuracy
            if classification.primary_domain == item["expected_domain"]:
                processing_metrics["accuracy_score"] += 1

            # Track confidence distribution
            if classification.confidence >= ClassificationConfidence.HIGH:
                processing_metrics["confidence_distribution"]["high"] += 1
            elif classification.confidence >= ClassificationConfidence.MEDIUM:
                processing_metrics["confidence_distribution"]["medium"] += 1
            else:
                processing_metrics["confidence_distribution"]["low"] += 1

        end_time = time.time()
        processing_metrics["total_time"] = end_time - start_time

        # Validate workflow results
        assert len(classification_results) == len(research_workflow)

        # Accuracy should be high
        accuracy_rate = processing_metrics["accuracy_score"] / len(research_workflow)
        assert accuracy_rate >= 0.75  # 75%+ accuracy

        # Performance should be good
        avg_processing_time = processing_metrics["total_time"] / len(research_workflow)
        assert avg_processing_time < 1.0  # Under 1 second per query

        # Confidence should be reasonable
        high_confidence_rate = processing_metrics["confidence_distribution"]["high"] / len(research_workflow)
        assert high_confidence_rate >= 0.5  # 50%+ high confidence

        # Generate detailed workflow analytics
        workflow_analytics = classifier.generate_workflow_analytics(classification_results)

        assert "classification_accuracy" in workflow_analytics
        assert "domain_distribution" in workflow_analytics
        assert "safety_prioritization" in workflow_analytics
        assert "processing_efficiency" in workflow_analytics

        print("✅ End-to-end query classification workflow successful")
        print(f"   Classification accuracy: {accuracy_rate:.1%}")
        print(f"   Average processing time: {avg_processing_time:.3f} seconds")
        print(f"   High confidence rate: {high_confidence_rate:.1%}")
        print(f"   Domain distribution: {workflow_analytics['domain_distribution']}")

    def test_classification_cost_optimization_integration(self):
        """Test classification integration with cost optimization systems."""
        classifier = PharmaceuticalQueryClassifier(pharmaceutical_optimized=True)

        # Test cost-aware query routing
        query_batches = {
            "high_priority_safety": [
                "emergency warfarin overdose treatment protocol",
                "immediate contraindications for prescribed medication",
            ],
            "standard_research": [
                "diabetes medication efficacy systematic review",
                "hypertension treatment guidelines meta-analysis",
            ],
            "general_information": ["basic pharmacology principles explanation", "drug classification system overview"],
        }

        routing_results = {}

        for batch_name, queries in query_batches.items():
            batch_classifications = []

            for query in queries:
                classification = classifier.classify_query(query)
                batch_classifications.append(classification)

            routing_results[batch_name] = {
                "classifications": batch_classifications,
                "avg_cost_tier": classifier.calculate_average_cost_tier(batch_classifications),
                "processing_priority": classifier.calculate_processing_priority(batch_classifications),
                "resource_allocation": classifier.recommend_resource_allocation(batch_classifications),
            }

        # Validate cost-aware routing

        # High priority safety should have highest cost tier and priority
        safety_results = routing_results["high_priority_safety"]
        assert safety_results["avg_cost_tier"] in ["high", "premium"]
        assert safety_results["processing_priority"] >= 3.0

        # General information should have lowest cost tier
        general_results = routing_results["general_information"]
        assert general_results["avg_cost_tier"] in ["low", "standard"]
        assert general_results["processing_priority"] <= 2.0

        # Resource allocation should be reasonable
        for batch_name, results in routing_results.items():
            allocation = results["resource_allocation"]
            assert "recommended_batch_size" in allocation
            assert "processing_order" in allocation
            assert "cost_optimization_score" in allocation
