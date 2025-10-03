"""
Pharmaceutical Domain Embedding Optimizer

Advanced optimization system for pharmaceutical content embeddings that enhances
semantic representation quality for medical and regulatory documents.

Features:
1. Medical terminology normalization and standardization
2. Drug name canonicalization and synonym resolution
3. Regulatory content prioritization and classification
4. Clinical trial data structure optimization
5. Safety-critical information highlighting
6. Pharmaceutical knowledge graph integration
7. Compliance and audit trail maintenance

<<use_mcp microsoft-learn>>
"""
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PharmaceuticalContentType(Enum):
    """Types of pharmaceutical content for specialized optimization."""

    CLINICAL_TRIAL = "clinical_trial"
    DRUG_LABEL = "drug_label"
    REGULATORY_DOCUMENT = "regulatory_document"
    PATENT = "patent"
    RESEARCH_PAPER = "research_paper"
    SAFETY_DATA = "safety_data"
    PRESCRIBING_INFORMATION = "prescribing_information"
    ADVERSE_EVENT_REPORT = "adverse_event_report"
    PHARMACOVIGILANCE = "pharmacovigilance"
    SUBMISSION_DOCUMENT = "submission_document"


class SafetyCriticalityLevel(Enum):
    """Safety criticality levels for pharmaceutical content."""

    CRITICAL = "critical"  # Black box warnings, contraindications
    HIGH = "high"  # Serious adverse events, drug interactions
    MEDIUM = "medium"  # Common side effects, precautions
    LOW = "low"  # General information, background


@dataclass
class PharmaceuticalTermMapping:
    """Mapping for pharmaceutical term normalization."""

    canonical_term: str
    synonyms: List[str]
    category: str
    regulatory_status: str = "approved"
    safety_level: SafetyCriticalityLevel = SafetyCriticalityLevel.LOW


@dataclass
class OptimizationResult:
    """Result of pharmaceutical content optimization."""

    optimized_text: str
    original_text: str
    optimizations_applied: List[str]
    pharmaceutical_terms_found: List[str]
    safety_signals_detected: List[str]
    regulatory_references: List[str]
    content_type: PharmaceuticalContentType
    safety_criticality: SafetyCriticalityLevel
    optimization_confidence: float


class PharmaceuticalEmbeddingOptimizer:
    """
    Advanced pharmaceutical domain optimizer for embedding enhancement.

    Provides intelligent content preprocessing, medical terminology normalization,
    and regulatory compliance optimization for pharmaceutical embeddings.
    """

    def __init__(self):
        """Initialize the pharmaceutical embedding optimizer."""

        # Drug name normalization database
        self.drug_mappings = self._initialize_drug_mappings()

        # Medical terminology standardization
        self.medical_terms = self._initialize_medical_terminology()

        # Regulatory frameworks and standards
        self.regulatory_frameworks = self._initialize_regulatory_frameworks()

        # Safety signal detection patterns
        self.safety_patterns = self._initialize_safety_patterns()

        # Content type detection patterns
        self.content_type_patterns = self._initialize_content_type_patterns()

        # Pharmaceutical abbreviation expansion
        self.abbreviation_mappings = self._initialize_abbreviation_mappings()

        # Optimization statistics
        self.optimization_stats = {
            "total_optimizations": 0,
            "drug_normalizations": 0,
            "terminology_standardizations": 0,
            "safety_signals_detected": 0,
            "regulatory_enhancements": 0,
        }

        logger.info("Initialized Pharmaceutical Embedding Optimizer")

    def optimize_pharmaceutical_content(
        self,
        texts: List[str],
        content_type: Optional[PharmaceuticalContentType] = None,
        enable_drug_normalization: bool = True,
        enable_terminology_standardization: bool = True,
        enable_safety_detection: bool = True,
        enable_regulatory_enhancement: bool = True,
    ) -> List[OptimizationResult]:
        """
        Optimize pharmaceutical content for improved embedding quality.

        Args:
            texts: List of pharmaceutical texts to optimize
            content_type: Optional content type for specialized optimization
            enable_drug_normalization: Enable drug name normalization
            enable_terminology_standardization: Enable medical term standardization
            enable_safety_detection: Enable safety signal detection
            enable_regulatory_enhancement: Enable regulatory content enhancement

        Returns:
            List of optimization results
        """
        results = []

        for text in texts:
            result = self._optimize_single_text(
                text=text,
                content_type=content_type,
                enable_drug_normalization=enable_drug_normalization,
                enable_terminology_standardization=enable_terminology_standardization,
                enable_safety_detection=enable_safety_detection,
                enable_regulatory_enhancement=enable_regulatory_enhancement,
            )
            results.append(result)

        self.optimization_stats["total_optimizations"] += len(texts)
        return results

    def _optimize_single_text(
        self,
        text: str,
        content_type: Optional[PharmaceuticalContentType],
        enable_drug_normalization: bool,
        enable_terminology_standardization: bool,
        enable_safety_detection: bool,
        enable_regulatory_enhancement: bool,
    ) -> OptimizationResult:
        """Optimize a single pharmaceutical text."""

        original_text = text
        optimized_text = text
        optimizations_applied = []
        pharmaceutical_terms_found = []
        safety_signals_detected = []
        regulatory_references = []

        # Detect content type if not provided
        if not content_type:
            content_type = self._detect_content_type(text)

        # Apply drug name normalization
        if enable_drug_normalization:
            optimized_text, drug_terms = self._normalize_drug_names(optimized_text)
            if drug_terms:
                optimizations_applied.append("drug_normalization")
                pharmaceutical_terms_found.extend(drug_terms)
                self.optimization_stats["drug_normalizations"] += 1

        # Apply medical terminology standardization
        if enable_terminology_standardization:
            optimized_text, medical_terms = self._standardize_medical_terminology(optimized_text)
            if medical_terms:
                optimizations_applied.append("terminology_standardization")
                pharmaceutical_terms_found.extend(medical_terms)
                self.optimization_stats["terminology_standardizations"] += 1

        # Expand pharmaceutical abbreviations
        optimized_text, abbreviations = self._expand_abbreviations(optimized_text)
        if abbreviations:
            optimizations_applied.append("abbreviation_expansion")

        # Detect safety signals
        if enable_safety_detection:
            safety_signals = self._detect_safety_signals(optimized_text)
            if safety_signals:
                safety_signals_detected.extend(safety_signals)
                optimizations_applied.append("safety_signal_detection")
                self.optimization_stats["safety_signals_detected"] += len(safety_signals)

        # Enhance regulatory content
        if enable_regulatory_enhancement:
            optimized_text, reg_refs = self._enhance_regulatory_content(optimized_text, content_type)
            if reg_refs:
                regulatory_references.extend(reg_refs)
                optimizations_applied.append("regulatory_enhancement")
                self.optimization_stats["regulatory_enhancements"] += 1

        # Apply content-type specific optimizations
        optimized_text = self._apply_content_type_optimizations(optimized_text, content_type)
        if optimized_text != text:
            optimizations_applied.append(f"{content_type.value}_specific_optimization")

        # Determine safety criticality
        safety_criticality = self._assess_safety_criticality(safety_signals_detected, content_type)

        # Calculate optimization confidence
        optimization_confidence = self._calculate_optimization_confidence(
            len(optimizations_applied), len(pharmaceutical_terms_found), len(safety_signals_detected), content_type
        )

        return OptimizationResult(
            optimized_text=optimized_text,
            original_text=original_text,
            optimizations_applied=optimizations_applied,
            pharmaceutical_terms_found=pharmaceutical_terms_found,
            safety_signals_detected=safety_signals_detected,
            regulatory_references=regulatory_references,
            content_type=content_type,
            safety_criticality=safety_criticality,
            optimization_confidence=optimization_confidence,
        )

    def _initialize_drug_mappings(self) -> Dict[str, PharmaceuticalTermMapping]:
        """Initialize drug name mappings for normalization."""
        return {
            # Common drug normalizations
            "acetaminophen": PharmaceuticalTermMapping(
                canonical_term="acetaminophen",
                synonyms=["paracetamol", "APAP", "N-acetyl-p-aminophenol"],
                category="analgesic",
                regulatory_status="approved",
            ),
            "ibuprofen": PharmaceuticalTermMapping(
                canonical_term="ibuprofen",
                synonyms=["advil", "motrin", "brufen"],
                category="NSAID",
                regulatory_status="approved",
            ),
            "aspirin": PharmaceuticalTermMapping(
                canonical_term="aspirin",
                synonyms=["acetylsalicylic acid", "ASA"],
                category="NSAID",
                regulatory_status="approved",
                safety_level=SafetyCriticalityLevel.MEDIUM,
            ),
            "warfarin": PharmaceuticalTermMapping(
                canonical_term="warfarin",
                synonyms=["coumadin", "jantoven"],
                category="anticoagulant",
                regulatory_status="approved",
                safety_level=SafetyCriticalityLevel.HIGH,
            ),
            "metformin": PharmaceuticalTermMapping(
                canonical_term="metformin",
                synonyms=["glucophage", "fortamet", "glumetza"],
                category="antidiabetic",
                regulatory_status="approved",
            ),
        }

    def _initialize_medical_terminology(self) -> Dict[str, str]:
        """Initialize medical terminology standardization mappings."""
        return {
            # Standardize common medical terms
            "adverse drug reaction": "adverse drug reaction (ADR)",
            "side effect": "adverse event",
            "drug interaction": "drug-drug interaction (DDI)",
            "contraindication": "contraindication",
            "black box warning": "boxed warning",
            "myocardial infarction": "myocardial infarction (MI)",
            "congestive heart failure": "heart failure (HF)",
            "chronic kidney disease": "chronic kidney disease (CKD)",
            "acute kidney injury": "acute kidney injury (AKI)",
            "gastrointestinal": "gastrointestinal (GI)",
            "cardiovascular": "cardiovascular (CV)",
            "central nervous system": "central nervous system (CNS)",
            "pharmacokinetics": "pharmacokinetics (PK)",
            "pharmacodynamics": "pharmacodynamics (PD)",
        }

    def _initialize_regulatory_frameworks(self) -> Dict[str, List[str]]:
        """Initialize regulatory framework references."""
        return {
            "FDA": [
                "Food and Drug Administration",
                "21 CFR",
                "Code of Federal Regulations",
                "FDA guidance",
                "CDER",
                "CBER",
                "CDRH",
            ],
            "EMA": [
                "European Medicines Agency",
                "European Union",
                "EMA guideline",
                "CHMP",
                "Committee for Medicinal Products for Human Use",
            ],
            "ICH": [
                "International Council for Harmonisation",
                "ICH guideline",
                "Good Clinical Practice",
                "GCP",
                "Good Manufacturing Practice",
                "GMP",
            ],
        }

    def _initialize_safety_patterns(self) -> Dict[str, List[str]]:
        """Initialize safety signal detection patterns."""
        return {
            "critical_warnings": [
                "black box warning",
                "boxed warning",
                "contraindicated",
                "fatal",
                "life-threatening",
                "death",
                "mortality",
            ],
            "serious_adverse_events": [
                "serious adverse event",
                "SAE",
                "hospitalization",
                "disability",
                "birth defect",
                "malignancy",
                "overdose",
            ],
            "drug_interactions": [
                "drug interaction",
                "contraindicated with",
                "avoid concurrent use",
                "increased risk when combined",
                "CYP450 interaction",
            ],
            "pregnancy_safety": [
                "pregnancy category",
                "teratogenic",
                "embryotoxic",
                "lactation",
                "breastfeeding",
                "reproductive toxicity",
            ],
        }

    def _initialize_content_type_patterns(self) -> Dict[PharmaceuticalContentType, List[str]]:
        """Initialize content type detection patterns."""
        return {
            PharmaceuticalContentType.CLINICAL_TRIAL: [
                "clinical trial",
                "randomized controlled",
                "phase I",
                "phase II",
                "phase III",
                "primary endpoint",
                "secondary endpoint",
                "efficacy",
                "safety",
                "clinical study",
            ],
            PharmaceuticalContentType.DRUG_LABEL: [
                "prescribing information",
                "package insert",
                "drug label",
                "contraindications",
                "warnings and precautions",
                "dosage and administration",
                "adverse reactions",
            ],
            PharmaceuticalContentType.REGULATORY_DOCUMENT: [
                "FDA approval",
                "regulatory submission",
                "NDA",
                "BLA",
                "ANDA",
                "regulatory review",
                "compliance",
                "inspection",
                "Form 483",
            ],
            PharmaceuticalContentType.SAFETY_DATA: [
                "adverse event",
                "safety signal",
                "pharmacovigilance",
                "FAERS",
                "periodic safety update",
                "risk evaluation",
                "REMS",
                "safety profile",
            ],
        }

    def _initialize_abbreviation_mappings(self) -> Dict[str, str]:
        """Initialize pharmaceutical abbreviation expansions."""
        return {
            "AE": "adverse event",
            "SAE": "serious adverse event",
            "ADR": "adverse drug reaction",
            "DDI": "drug-drug interaction",
            "PK": "pharmacokinetics",
            "PD": "pharmacodynamics",
            "QD": "once daily",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
            "PRN": "as needed",
            "PO": "by mouth",
            "IV": "intravenous",
            "IM": "intramuscular",
            "SQ": "subcutaneous",
            "FDA": "Food and Drug Administration",
            "EMA": "European Medicines Agency",
            "ICH": "International Council for Harmonisation",
            "GCP": "Good Clinical Practice",
            "GMP": "Good Manufacturing Practice",
            "CFR": "Code of Federal Regulations",
        }

    def _detect_content_type(self, text: str) -> PharmaceuticalContentType:
        """Detect the pharmaceutical content type from text."""
        text_lower = text.lower()
        scores = defaultdict(int)

        for content_type, patterns in self.content_type_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    scores[content_type] += 1

        if scores:
            return max(scores.keys(), key=lambda x: scores[x])
        else:
            return PharmaceuticalContentType.RESEARCH_PAPER  # Default

    def _normalize_drug_names(self, text: str) -> Tuple[str, List[str]]:
        """Normalize drug names to canonical forms."""
        normalized_text = text
        found_drugs = []

        for canonical, mapping in self.drug_mappings.items():
            for synonym in mapping.synonyms:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(synonym), re.IGNORECASE)
                if pattern.search(normalized_text):
                    normalized_text = pattern.sub(f"{canonical} ({synonym})", normalized_text)
                    found_drugs.append(canonical)

        return normalized_text, found_drugs

    def _standardize_medical_terminology(self, text: str) -> Tuple[str, List[str]]:
        """Standardize medical terminology."""
        standardized_text = text
        found_terms = []

        for term, standard in self.medical_terms.items():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(standardized_text):
                standardized_text = pattern.sub(standard, standardized_text)
                found_terms.append(standard)

        return standardized_text, found_terms

    def _expand_abbreviations(self, text: str) -> Tuple[str, List[str]]:
        """Expand pharmaceutical abbreviations."""
        expanded_text = text
        expanded_abbreviations = []

        for abbrev, expansion in self.abbreviation_mappings.items():
            # Match abbreviation as whole word
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                expanded_text = re.sub(pattern, f"{abbrev} ({expansion})", expanded_text, flags=re.IGNORECASE)
                expanded_abbreviations.append(abbrev)

        return expanded_text, expanded_abbreviations

    def _detect_safety_signals(self, text: str) -> List[str]:
        """Detect safety signals in pharmaceutical text."""
        safety_signals = []
        text_lower = text.lower()

        for category, patterns in self.safety_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    safety_signals.append(f"{category}:{pattern}")

        return safety_signals

    def _enhance_regulatory_content(self, text: str, content_type: PharmaceuticalContentType) -> Tuple[str, List[str]]:
        """Enhance regulatory content with proper context."""
        enhanced_text = text
        regulatory_refs = []

        for framework, terms in self.regulatory_frameworks.items():
            for term in terms:
                if term.lower() in text.lower():
                    regulatory_refs.append(f"{framework}:{term}")

        # Add regulatory context for specific content types
        if content_type == PharmaceuticalContentType.REGULATORY_DOCUMENT:
            if "FDA" in text and "regulatory compliance" not in text.lower():
                enhanced_text += " [Regulatory Compliance Context: FDA submission document]"

        return enhanced_text, regulatory_refs

    def _apply_content_type_optimizations(self, text: str, content_type: PharmaceuticalContentType) -> str:
        """Apply content-type specific optimizations."""
        optimized_text = text

        if content_type == PharmaceuticalContentType.CLINICAL_TRIAL:
            # Enhance clinical trial data with endpoint emphasis
            if "primary endpoint" in text.lower() and "[PRIMARY]" not in text:
                optimized_text = optimized_text.replace("primary endpoint", "[PRIMARY] primary endpoint")

        elif content_type == PharmaceuticalContentType.DRUG_LABEL:
            # Emphasize safety-critical sections
            if "contraindications" in text.lower() and "[CONTRAINDICATION]" not in text:
                optimized_text = optimized_text.replace("contraindications", "[CONTRAINDICATION] contraindications")

        elif content_type == PharmaceuticalContentType.SAFETY_DATA:
            # Highlight safety signals
            if "serious adverse event" in text.lower() and "[SAE]" not in text:
                optimized_text = optimized_text.replace("serious adverse event", "[SAE] serious adverse event")

        return optimized_text

    def _assess_safety_criticality(
        self, safety_signals: List[str], content_type: PharmaceuticalContentType
    ) -> SafetyCriticalityLevel:
        """Assess the safety criticality level of content."""
        critical_indicators = ["critical_warnings", "fatal", "death"]
        high_indicators = ["serious_adverse_events", "drug_interactions", "SAE"]

        if any(signal.startswith(tuple(critical_indicators)) for signal in safety_signals):
            return SafetyCriticalityLevel.CRITICAL

        if any(signal.startswith(tuple(high_indicators)) for signal in safety_signals):
            return SafetyCriticalityLevel.HIGH

        if safety_signals:
            return SafetyCriticalityLevel.MEDIUM

        return SafetyCriticalityLevel.LOW

    def _calculate_optimization_confidence(
        self,
        optimizations_count: int,
        pharmaceutical_terms_count: int,
        safety_signals_count: int,
        content_type: PharmaceuticalContentType,
    ) -> float:
        """Calculate confidence score for optimization quality."""
        base_score = 0.5  # Base confidence

        # Boost for number of optimizations applied
        optimization_boost = min(optimizations_count * 0.1, 0.3)

        # Boost for pharmaceutical terms found
        terms_boost = min(pharmaceutical_terms_count * 0.05, 0.2)

        # Boost for safety signals detected
        safety_boost = min(safety_signals_count * 0.03, 0.15)

        # Content type specific confidence adjustment
        content_type_boost = {
            PharmaceuticalContentType.DRUG_LABEL: 0.1,
            PharmaceuticalContentType.CLINICAL_TRIAL: 0.08,
            PharmaceuticalContentType.SAFETY_DATA: 0.12,
            PharmaceuticalContentType.REGULATORY_DOCUMENT: 0.09,
        }.get(content_type, 0.05)

        total_confidence = min(base_score + optimization_boost + terms_boost + safety_boost + content_type_boost, 1.0)

        return total_confidence

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        return {
            "optimization_stats": self.optimization_stats.copy(),
            "available_drug_mappings": len(self.drug_mappings),
            "available_medical_terms": len(self.medical_terms),
            "regulatory_frameworks": list(self.regulatory_frameworks.keys()),
            "safety_pattern_categories": list(self.safety_patterns.keys()),
            "supported_content_types": [ct.value for ct in PharmaceuticalContentType],
        }

    def add_custom_drug_mapping(
        self,
        canonical_name: str,
        synonyms: List[str],
        category: str,
        safety_level: SafetyCriticalityLevel = SafetyCriticalityLevel.LOW,
    ):
        """Add custom drug mapping for specialized pharmaceutical content."""
        self.drug_mappings[canonical_name] = PharmaceuticalTermMapping(
            canonical_term=canonical_name, synonyms=synonyms, category=category, safety_level=safety_level
        )
        logger.info(f"Added custom drug mapping: {canonical_name}")

    def add_custom_medical_term(self, term: str, standardized_form: str):
        """Add custom medical terminology standardization."""
        self.medical_terms[term] = standardized_form
        logger.info(f"Added custom medical term: {term} -> {standardized_form}")


# Global instance for easy access
pharmaceutical_optimizer = PharmaceuticalEmbeddingOptimizer()


def optimize_pharmaceutical_texts(
    texts: List[str], content_type: Optional[str] = None, enable_all_optimizations: bool = True
) -> List[OptimizationResult]:
    """
    Convenience function to optimize pharmaceutical texts for embeddings.

    Args:
        texts: List of pharmaceutical texts to optimize
        content_type: Optional content type string
        enable_all_optimizations: Enable all available optimizations

    Returns:
        List of optimization results
    """
    # Convert string content type to enum
    content_type_enum = None
    if content_type:
        try:
            content_type_enum = PharmaceuticalContentType(content_type)
        except ValueError:
            logger.warning(f"Unknown content type: {content_type}")

    return pharmaceutical_optimizer.optimize_pharmaceutical_content(
        texts=texts,
        content_type=content_type_enum,
        enable_drug_normalization=enable_all_optimizations,
        enable_terminology_standardization=enable_all_optimizations,
        enable_safety_detection=enable_all_optimizations,
        enable_regulatory_enhancement=enable_all_optimizations,
    )
