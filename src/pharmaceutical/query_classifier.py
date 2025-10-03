"""
Pharmaceutical Query Classification and Prioritization System

Advanced classification system for pharmaceutical research queries with
intelligent prioritization, drug safety detection, and research workflow optimization.

Features:
- Multi-level pharmaceutical query classification
- Drug safety priority escalation
- Clinical research categorization
- Regulatory compliance context detection
- Cost-aware priority assignment

Integrates with the cloud-first architecture to ensure critical pharmaceutical
queries receive optimal routing and resource allocation.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PharmaceuticalDomain(Enum):
    """Primary pharmaceutical research domains."""

    DRUG_SAFETY = "drug_safety"
    CLINICAL_TRIALS = "clinical_trials"
    PHARMACOKINETICS = "pharmacokinetics"
    DRUG_INTERACTIONS = "drug_interactions"
    MECHANISM_OF_ACTION = "mechanism_of_action"
    DOSAGE_GUIDELINES = "dosage_guidelines"
    ADVERSE_REACTIONS = "adverse_reactions"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    THERAPEUTIC_EFFICACY = "therapeutic_efficacy"
    PHARMACOGENOMICS = "pharmacogenomics"
    GENERAL_RESEARCH = "general_research"


class SafetyUrgency(IntEnum):
    """Safety urgency levels for pharmaceutical queries."""

    CRITICAL = 1  # Immediate safety concerns
    HIGH = 2  # Important safety information
    MODERATE = 3  # General safety guidance
    LOW = 4  # Background safety information
    NONE = 5  # No safety implications


class ResearchPriority(IntEnum):
    """Research priority levels."""

    EMERGENCY = 1  # Emergency/critical patient care
    HIGH = 2  # Active clinical research
    NORMAL = 3  # Standard research queries
    BACKGROUND = 4  # Literature review, general research
    EXPLORATORY = 5  # Discovery and exploration


@dataclass
class PharmaceuticalContext:
    """Comprehensive pharmaceutical context for queries."""

    domain: PharmaceuticalDomain
    safety_urgency: SafetyUrgency
    research_priority: ResearchPriority
    drug_names: List[str] = field(default_factory=list)
    therapeutic_areas: List[str] = field(default_factory=list)
    regulatory_context: Optional[str] = None
    patient_population: Optional[str] = None
    clinical_phase: Optional[str] = None
    confidence_score: float = 0.0


class PharmaceuticalQueryClassifier:
    """
    Advanced pharmaceutical query classifier with safety prioritization.

    Provides sophisticated analysis of pharmaceutical research queries with
    automatic classification, priority assignment, and safety escalation.
    """

    def __init__(self):
        """Initialize pharmaceutical query classifier with domain knowledge."""

        # Safety-critical keywords (highest priority)
        self.safety_critical_keywords = {
            "immediate_danger": [
                "overdose",
                "toxic",
                "poisoning",
                "fatal",
                "death",
                "emergency",
                "life-threatening",
                "critical",
                "urgent",
                "immediate",
            ],
            "adverse_reactions": [
                "adverse",
                "side effect",
                "reaction",
                "toxicity",
                "harm",
                "dangerous",
                "warning",
                "contraindicated",
                "avoid",
            ],
            "drug_interactions": [
                "interaction",
                "combined with",
                "together with",
                "concurrent",
                "contraindication",
                "incompatible",
                "drug-drug",
            ],
            "safety_monitoring": [
                "monitoring",
                "blood test",
                "liver function",
                "kidney function",
                "cardiac",
                "respiratory",
                "neurological",
            ],
        }

        # Clinical research keywords
        self.clinical_research_keywords = {
            "clinical_trials": [
                "clinical trial",
                "study",
                "phase i",
                "phase ii",
                "phase iii",
                "randomized",
                "placebo",
                "double-blind",
                "efficacy",
                "endpoint",
            ],
            "regulatory": [
                "fda approved",
                "regulatory",
                "indication",
                "label",
                "prescribing",
                "compliance",
                "guideline",
                "protocol",
            ],
            "therapeutic_areas": [
                "oncology",
                "cardiology",
                "neurology",
                "psychiatry",
                "endocrinology",
                "infectious disease",
                "dermatology",
                "gastroenterology",
            ],
        }

        # Pharmacological keywords
        self.pharmacology_keywords = {
            "pharmacokinetics": [
                "absorption",
                "distribution",
                "metabolism",
                "excretion",
                "adme",
                "bioavailability",
                "clearance",
                "half-life",
                "pharmacokinetics",
            ],
            "mechanism_of_action": [
                "mechanism",
                "action",
                "pathway",
                "target",
                "receptor",
                "enzyme",
                "protein",
                "molecular",
                "cellular",
            ],
            "dosing": ["dose", "dosage", "dosing", "administration", "frequency", "regimen", "schedule", "titration"],
        }

        # Drug name patterns (common pharmaceutical naming patterns)
        self.drug_name_patterns = [
            r"\b\w+mycin\b",  # Antibiotics ending in -mycin
            r"\b\w+cillin\b",  # Penicillin derivatives
            r"\b\w+prazole\b",  # Proton pump inhibitors
            r"\b\w+sartan\b",  # ARBs
            r"\b\w+pril\b",  # ACE inhibitors
            r"\b\w+olol\b",  # Beta blockers
            r"\b\w+statin\b",  # Statins
            r"\b\w+tidine\b",  # H2 blockers
            r"\b\w+zole\b",  # Antifungals
            r"\b\w+mab\b",  # Monoclonal antibodies
        ]

        # Patient population keywords
        self.patient_populations = {
            "vulnerable": ["pediatric", "elderly", "geriatric", "pregnant", "nursing"],
            "comorbidity": ["diabetes", "hypertension", "kidney", "liver", "cardiac"],
            "special": ["renal impairment", "hepatic impairment", "dialysis"],
        }

        logger.info("PharmaceuticalQueryClassifier initialized with comprehensive domain knowledge")

    def classify_query(self, query_text: str) -> PharmaceuticalContext:
        """
        Comprehensive pharmaceutical query classification.

        Args:
            query_text: The pharmaceutical research query

        Returns:
            PharmaceuticalContext with complete classification
        """
        query_lower = query_text.lower()

        # Initialize classification results
        domain = PharmaceuticalDomain.GENERAL_RESEARCH
        safety_urgency = SafetyUrgency.NONE
        research_priority = ResearchPriority.NORMAL
        drug_names = []
        therapeutic_areas = []
        regulatory_context = None
        patient_population = None
        clinical_phase = None

        # Safety classification (highest priority)
        safety_urgency, safety_domain = self._classify_safety_urgency(query_lower)
        if safety_domain:
            domain = safety_domain

        # Domain classification
        if domain == PharmaceuticalDomain.GENERAL_RESEARCH:
            domain = self._classify_pharmaceutical_domain(query_lower)

        # Research priority assignment
        research_priority = self._assign_research_priority(query_lower, domain, safety_urgency)

        # Extract drug names
        drug_names = self._extract_drug_names(query_text)

        # Extract therapeutic areas
        therapeutic_areas = self._extract_therapeutic_areas(query_lower)

        # Extract regulatory context
        regulatory_context = self._extract_regulatory_context(query_lower)

        # Extract patient population
        patient_population = self._extract_patient_population(query_lower)

        # Extract clinical phase information
        clinical_phase = self._extract_clinical_phase(query_lower)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(query_text, domain, safety_urgency, research_priority)

        return PharmaceuticalContext(
            domain=domain,
            safety_urgency=safety_urgency,
            research_priority=research_priority,
            drug_names=drug_names,
            therapeutic_areas=therapeutic_areas,
            regulatory_context=regulatory_context,
            patient_population=patient_population,
            clinical_phase=clinical_phase,
            confidence_score=confidence_score,
        )

    def _classify_safety_urgency(self, query_lower: str) -> Tuple[SafetyUrgency, Optional[PharmaceuticalDomain]]:
        """Classify safety urgency and return associated domain if applicable."""

        # Critical safety - immediate danger
        if any(keyword in query_lower for keyword in self.safety_critical_keywords["immediate_danger"]):
            return SafetyUrgency.CRITICAL, PharmaceuticalDomain.DRUG_SAFETY

        # High urgency - adverse reactions
        if any(keyword in query_lower for keyword in self.safety_critical_keywords["adverse_reactions"]):
            return SafetyUrgency.HIGH, PharmaceuticalDomain.ADVERSE_REACTIONS

        # High urgency - drug interactions
        if any(keyword in query_lower for keyword in self.safety_critical_keywords["drug_interactions"]):
            return SafetyUrgency.HIGH, PharmaceuticalDomain.DRUG_INTERACTIONS

        # Moderate urgency - safety monitoring
        if any(keyword in query_lower for keyword in self.safety_critical_keywords["safety_monitoring"]):
            return SafetyUrgency.MODERATE, PharmaceuticalDomain.DRUG_SAFETY

        return SafetyUrgency.NONE, None

    def _classify_pharmaceutical_domain(self, query_lower: str) -> PharmaceuticalDomain:
        """Classify the primary pharmaceutical research domain."""

        # Clinical trials
        if any(keyword in query_lower for keyword in self.clinical_research_keywords["clinical_trials"]):
            return PharmaceuticalDomain.CLINICAL_TRIALS

        # Regulatory compliance
        if any(keyword in query_lower for keyword in self.clinical_research_keywords["regulatory"]):
            return PharmaceuticalDomain.REGULATORY_COMPLIANCE

        # Pharmacokinetics
        if any(keyword in query_lower for keyword in self.pharmacology_keywords["pharmacokinetics"]):
            return PharmaceuticalDomain.PHARMACOKINETICS

        # Mechanism of action
        if any(keyword in query_lower for keyword in self.pharmacology_keywords["mechanism_of_action"]):
            return PharmaceuticalDomain.MECHANISM_OF_ACTION

        # Dosage guidelines
        if any(keyword in query_lower for keyword in self.pharmacology_keywords["dosing"]):
            return PharmaceuticalDomain.DOSAGE_GUIDELINES

        # Check for therapeutic efficacy keywords
        efficacy_keywords = ["efficacy", "effectiveness", "therapeutic", "treatment", "outcome"]
        if any(keyword in query_lower for keyword in efficacy_keywords):
            return PharmaceuticalDomain.THERAPEUTIC_EFFICACY

        # Check for pharmacogenomics
        pharmacogenomics_keywords = ["genetic", "genomic", "polymorphism", "pharmacogenomic"]
        if any(keyword in query_lower for keyword in pharmacogenomics_keywords):
            return PharmaceuticalDomain.PHARMACOGENOMICS

        return PharmaceuticalDomain.GENERAL_RESEARCH

    def _assign_research_priority(
        self, query_lower: str, domain: PharmaceuticalDomain, safety_urgency: SafetyUrgency
    ) -> ResearchPriority:
        """Assign research priority based on domain and safety urgency."""

        # Safety urgency overrides other considerations
        if safety_urgency == SafetyUrgency.CRITICAL:
            return ResearchPriority.EMERGENCY
        elif safety_urgency == SafetyUrgency.HIGH:
            return ResearchPriority.HIGH

        # Domain-based priority assignment
        high_priority_domains = {
            PharmaceuticalDomain.DRUG_SAFETY,
            PharmaceuticalDomain.ADVERSE_REACTIONS,
            PharmaceuticalDomain.DRUG_INTERACTIONS,
            PharmaceuticalDomain.CLINICAL_TRIALS,
        }

        if domain in high_priority_domains:
            return ResearchPriority.HIGH

        # Check for urgent keywords
        urgent_keywords = ["urgent", "immediate", "critical", "emergency", "stat"]
        if any(keyword in query_lower for keyword in urgent_keywords):
            return ResearchPriority.EMERGENCY

        # Check for active research keywords
        active_keywords = ["current", "ongoing", "active", "new", "recent"]
        if any(keyword in query_lower for keyword in active_keywords):
            return ResearchPriority.HIGH

        # Check for background research keywords
        background_keywords = ["review", "overview", "general", "background", "literature"]
        if any(keyword in query_lower for keyword in background_keywords):
            return ResearchPriority.BACKGROUND

        # Check for exploratory keywords
        exploratory_keywords = ["explore", "investigate", "discover", "potential", "novel"]
        if any(keyword in query_lower for keyword in exploratory_keywords):
            return ResearchPriority.EXPLORATORY

        return ResearchPriority.NORMAL

    def _extract_drug_names(self, query_text: str) -> List[str]:
        """Extract potential drug names from query text."""
        drug_names = []

        # Check for drug name patterns
        for pattern in self.drug_name_patterns:
            matches = re.findall(pattern, query_text, re.IGNORECASE)
            drug_names.extend(matches)

        # Common pharmaceutical names (not exhaustive, but covers major categories)
        common_drugs = [
            # Cardiovascular
            "metoprolol",
            "atenolol",
            "propranolol",
            "lisinopril",
            "enalapril",
            "losartan",
            "valsartan",
            "amlodipine",
            "nifedipine",
            "atorvastatin",
            "simvastatin",
            "warfarin",
            "heparin",
            "aspirin",
            # Diabetes
            "metformin",
            "insulin",
            "glipizide",
            "glyburide",
            "pioglitazone",
            # Antibiotics
            "amoxicillin",
            "penicillin",
            "azithromycin",
            "ciprofloxacin",
            "doxycycline",
            "vancomycin",
            # Pain/Inflammation
            "ibuprofen",
            "naproxen",
            "acetaminophen",
            "tramadol",
            "morphine",
            # Psychiatric
            "sertraline",
            "fluoxetine",
            "escitalopram",
            "alprazolam",
            "lorazepam",
            # Gastrointestinal
            "omeprazole",
            "lansoprazole",
            "ranitidine",
            "famotidine",
        ]

        query_lower = query_text.lower()
        for drug in common_drugs:
            if drug in query_lower:
                drug_names.append(drug.capitalize())

        return list(set(drug_names))  # Remove duplicates

    def _extract_therapeutic_areas(self, query_lower: str) -> List[str]:
        """Extract therapeutic areas from query."""
        therapeutic_areas = []

        area_keywords = {
            "cardiology": ["cardiac", "heart", "cardiovascular", "hypertension", "blood pressure"],
            "oncology": ["cancer", "tumor", "oncology", "chemotherapy", "radiation"],
            "neurology": ["neurological", "brain", "nervous", "alzheimer", "parkinson"],
            "endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "glucose"],
            "psychiatry": ["depression", "anxiety", "psychiatric", "mental health"],
            "infectious_disease": ["infection", "bacterial", "viral", "antibiotic"],
            "gastroenterology": ["gastrointestinal", "stomach", "liver", "digestive"],
            "pulmonology": ["respiratory", "lung", "asthma", "copd"],
            "nephrology": ["kidney", "renal", "dialysis"],
            "dermatology": ["skin", "dermatological", "topical"],
        }

        for area, keywords in area_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                therapeutic_areas.append(area)

        return therapeutic_areas

    def _extract_regulatory_context(self, query_lower: str) -> Optional[str]:
        """Extract regulatory context information."""

        regulatory_patterns = {
            "fda": ["fda", "food and drug administration"],
            "ema": ["ema", "european medicines agency"],
            "ich": ["ich", "international council for harmonisation"],
            "guideline": ["guideline", "guidance", "protocol"],
            "approval": ["approval", "approved", "indication"],
            "label": ["label", "labeling", "prescribing information"],
        }

        for context, keywords in regulatory_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return context

        return None

    def _extract_patient_population(self, query_lower: str) -> Optional[str]:
        """Extract patient population information."""

        for population_type, keywords in self.patient_populations.items():
            if any(keyword in query_lower for keyword in keywords):
                # Return the specific keyword found
                for keyword in keywords:
                    if keyword in query_lower:
                        return keyword

        return None

    def _extract_clinical_phase(self, query_lower: str) -> Optional[str]:
        """Extract clinical trial phase information."""

        phase_patterns = [
            "phase i",
            "phase 1",
            "phase ii",
            "phase 2",
            "phase iii",
            "phase 3",
            "phase iv",
            "phase 4",
            "preclinical",
            "post-marketing",
        ]

        for phase in phase_patterns:
            if phase in query_lower:
                return phase.replace(" ", "_")

        return None

    def _calculate_confidence_score(
        self,
        query_text: str,
        domain: PharmaceuticalDomain,
        safety_urgency: SafetyUrgency,
        research_priority: ResearchPriority,
    ) -> float:
        """Calculate confidence score for the classification."""

        base_score = 0.5  # Base confidence

        # Higher confidence for safety-critical queries
        if safety_urgency in [SafetyUrgency.CRITICAL, SafetyUrgency.HIGH]:
            base_score += 0.3

        # Higher confidence for specific domains
        specific_domains = {
            PharmaceuticalDomain.DRUG_SAFETY,
            PharmaceuticalDomain.DRUG_INTERACTIONS,
            PharmaceuticalDomain.ADVERSE_REACTIONS,
        }
        if domain in specific_domains:
            base_score += 0.2

        # Adjust based on query length and specificity
        word_count = len(query_text.split())
        if word_count > 10:  # More specific queries
            base_score += 0.1
        elif word_count < 5:  # Very short queries
            base_score -= 0.1

        # Check for pharmaceutical terminology density
        pharma_terms = 0
        all_keywords = []
        for category in [self.safety_critical_keywords, self.clinical_research_keywords, self.pharmacology_keywords]:
            for keyword_list in category.values():
                all_keywords.extend(keyword_list)

        query_lower = query_text.lower()
        pharma_terms = sum(1 for term in all_keywords if term in query_lower)

        # Normalize by query length
        term_density = pharma_terms / max(word_count, 1)
        base_score += min(0.2, term_density * 0.5)

        return min(1.0, max(0.1, base_score))

    def get_priority_routing_config(self, context: PharmaceuticalContext) -> Dict[str, Any]:
        """
        Get routing configuration based on pharmaceutical context.

        Args:
            context: Pharmaceutical context from classification

        Returns:
            Routing configuration for cloud-first optimization
        """
        config = {
            "cloud_first_priority": True,
            "batch_eligible": True,
            "cost_tier_preference": "free_tier",
            "timeout_seconds": 30,
            "retry_attempts": 3,
        }

        # Critical safety queries get priority routing
        if context.safety_urgency == SafetyUrgency.CRITICAL:
            config.update(
                {
                    "cloud_first_priority": True,
                    "batch_eligible": False,  # Process immediately
                    "cost_tier_preference": "any",  # Cost not a factor
                    "timeout_seconds": 15,  # Faster timeout
                    "retry_attempts": 5,  # More retries
                }
            )

        # High priority research gets enhanced routing
        elif context.research_priority in [ResearchPriority.EMERGENCY, ResearchPriority.HIGH]:
            config.update(
                {"cloud_first_priority": True, "batch_eligible": False, "timeout_seconds": 20, "retry_attempts": 4}
            )

        # Background queries can be batched and deprioritized
        elif context.research_priority in [ResearchPriority.BACKGROUND, ResearchPriority.EXPLORATORY]:
            config.update(
                {
                    "batch_eligible": True,
                    "cost_tier_preference": "free_tier_only",
                    "timeout_seconds": 60,
                    "retry_attempts": 2,
                }
            )

        return config

    def generate_classification_report(self, query_text: str) -> Dict[str, Any]:
        """
        Generate comprehensive classification report.

        Args:
            query_text: Query to classify and analyze

        Returns:
            Detailed classification report
        """
        context = self.classify_query(query_text)
        routing_config = self.get_priority_routing_config(context)

        return {
            "query_text": query_text,
            "classification_timestamp": datetime.now().isoformat(),
            "pharmaceutical_context": {
                "domain": context.domain.value,
                "safety_urgency": context.safety_urgency.name,
                "research_priority": context.research_priority.name,
                "drug_names": context.drug_names,
                "therapeutic_areas": context.therapeutic_areas,
                "regulatory_context": context.regulatory_context,
                "patient_population": context.patient_population,
                "clinical_phase": context.clinical_phase,
                "confidence_score": round(context.confidence_score, 3),
            },
            "routing_configuration": routing_config,
            "recommendations": self._generate_recommendations(context),
        }

    def _generate_recommendations(self, context: PharmaceuticalContext) -> List[str]:
        """Generate recommendations based on classification."""
        recommendations = []

        if context.safety_urgency == SafetyUrgency.CRITICAL:
            recommendations.append("URGENT: Process immediately - patient safety implications")
            recommendations.append("Route to cloud endpoint with highest reliability")

        if context.domain == PharmaceuticalDomain.DRUG_INTERACTIONS:
            recommendations.append("Cross-reference with drug interaction databases")
            recommendations.append("Include contraindication warnings in response")

        if context.patient_population:
            recommendations.append(f"Consider {context.patient_population}-specific dosing guidelines")

        if context.confidence_score < 0.6:
            recommendations.append("Low confidence classification - consider manual review")

        if context.research_priority in [ResearchPriority.BACKGROUND, ResearchPriority.EXPLORATORY]:
            recommendations.append("Suitable for batch processing to optimize costs")

        return recommendations


# Convenience functions for pharmaceutical classification
def classify_pharmaceutical_query(query_text: str) -> PharmaceuticalContext:
    """
    Quick pharmaceutical query classification.

    Args:
        query_text: Query to classify

    Returns:
        Pharmaceutical context with classification results
    """
    classifier = PharmaceuticalQueryClassifier()
    return classifier.classify_query(query_text)


def get_pharmaceutical_routing_config(query_text: str) -> Dict[str, Any]:
    """
    Get routing configuration for pharmaceutical query.

    Args:
        query_text: Query to analyze

    Returns:
        Routing configuration for cloud-first optimization
    """
    classifier = PharmaceuticalQueryClassifier()
    context = classifier.classify_query(query_text)
    return classifier.get_priority_routing_config(context)


if __name__ == "__main__":
    # Test pharmaceutical classification
    classifier = PharmaceuticalQueryClassifier()

    test_queries = [
        "What are the contraindications for metformin in elderly patients with kidney disease?",
        "Explain the mechanism of action of ACE inhibitors in hypertension treatment",
        "Drug interactions between warfarin and NSAIDs - urgent patient safety concern",
        "Phase III clinical trial results for new oncology drug efficacy",
        "General overview of cardiovascular medication classes",
    ]

    for query in test_queries:
        report = classifier.generate_classification_report(query)
        print(f"\nQuery: {query}")
        print(f"Domain: {report['pharmaceutical_context']['domain']}")
        print(f"Safety Urgency: {report['pharmaceutical_context']['safety_urgency']}")
        print(f"Research Priority: {report['pharmaceutical_context']['research_priority']}")
        print(f"Confidence: {report['pharmaceutical_context']['confidence_score']}")
        print(f"Routing: {report['routing_configuration']}")
        print("---")
