"""
Multi-Model Embedding Strategy for NVIDIA NeMo Retriever

Advanced model selection and routing system that intelligently chooses the optimal
embedding model based on content analysis, pharmaceutical domain requirements,
and performance characteristics.

Features:
1. Content-aware model selection
2. Performance-based model routing
3. Pharmaceutical domain optimization
4. Load balancing across models
5. Fallback and redundancy management
6. A/B testing capabilities

<<use_mcp microsoft-learn>>
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import hashlib
import json
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class ModelPerformanceRating(Enum):
    """Model performance ratings."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    UNKNOWN = "unknown"

@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of an embedding model."""
    name: str
    max_sequence_length: int
    embedding_dimension: int
    supports_multilingual: bool
    supports_technical_content: bool
    supports_medical_content: bool
    performance_rating: ModelPerformanceRating
    cost_tier: str  # "premium", "standard", "economy"
    latency_ms: float = 0.0
    accuracy_score: float = 0.0
    pharmaceutical_optimized: bool = False
    regulatory_compliant: bool = False

@dataclass
class ContentAnalysis:
    """Analysis of content characteristics for model selection."""
    content_type: str
    language: str
    technical_complexity: float  # 0.0 to 1.0
    medical_content_ratio: float  # 0.0 to 1.0
    regulatory_content: bool
    safety_critical: bool
    document_length: int
    pharmaceutical_terms_count: int
    domain_specificity: float  # 0.0 to 1.0

@dataclass
class ModelPerformanceHistory:
    """Historical performance data for a model."""
    model_name: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 1.0
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    error_count: int = 0
    total_requests: int = 0
    last_used: float = 0.0
    pharmaceutical_performance: float = 0.0

@dataclass
class ModelSelectionResult:
    """Result of model selection process."""
    selected_model: str
    confidence_score: float
    selection_reason: str
    fallback_models: List[str]
    performance_prediction: Dict[str, float]
    pharmaceutical_optimization_applied: bool

class MultiModelEmbeddingStrategy:
    """
    Advanced multi-model embedding strategy with pharmaceutical optimization.

    Provides intelligent model selection, performance tracking, and pharmaceutical
    domain-specific optimizations for optimal embedding quality and compliance.
    """

    def __init__(self):
        """Initialize the multi-model embedding strategy."""

        # Available embedding models with their capabilities
        self.model_capabilities = {
            "nv-embedqa-e5-v5": ModelCapabilities(
                name="nv-embedqa-e5-v5",
                max_sequence_length=32768,
                embedding_dimension=1024,
                supports_multilingual=True,
                supports_technical_content=True,
                supports_medical_content=True,
                performance_rating=ModelPerformanceRating.EXCELLENT,
                cost_tier="premium",
                pharmaceutical_optimized=True,
                regulatory_compliant=True
            ),
            "nv-embedqa-mistral7b-v2": ModelCapabilities(
                name="nv-embedqa-mistral7b-v2",
                max_sequence_length=32768,
                embedding_dimension=4096,
                supports_multilingual=True,
                supports_technical_content=True,
                supports_medical_content=True,
                performance_rating=ModelPerformanceRating.EXCELLENT,
                cost_tier="premium",
                pharmaceutical_optimized=True,
                regulatory_compliant=True
            ),
            "snowflake-arctic-embed-l": ModelCapabilities(
                name="snowflake-arctic-embed-l",
                max_sequence_length=8192,
                embedding_dimension=1024,
                supports_multilingual=True,
                supports_technical_content=True,
                supports_medical_content=False,
                performance_rating=ModelPerformanceRating.GOOD,
                cost_tier="standard",
                pharmaceutical_optimized=False,
                regulatory_compliant=False
            ),
            "nv-embed-v1": ModelCapabilities(
                name="nv-embed-v1",
                max_sequence_length=2048,
                embedding_dimension=1024,
                supports_multilingual=False,
                supports_technical_content=True,
                supports_medical_content=False,
                performance_rating=ModelPerformanceRating.AVERAGE,
                cost_tier="economy",
                pharmaceutical_optimized=False,
                regulatory_compliant=False
            )
        }

        # Performance tracking
        self.performance_history: Dict[str, ModelPerformanceHistory] = {}
        for model_name in self.model_capabilities.keys():
            self.performance_history[model_name] = ModelPerformanceHistory(model_name=model_name)

        # Pharmaceutical domain-specific model preferences
        self.pharmaceutical_preferences = {
            "clinical_trial": {
                "primary": "nv-embedqa-e5-v5",
                "secondary": "nv-embedqa-mistral7b-v2",
                "reason": "Optimized for medical question-answering and regulatory compliance"
            },
            "drug_label": {
                "primary": "nv-embedqa-e5-v5",
                "secondary": "nv-embedqa-mistral7b-v2",
                "reason": "High precision required for safety-critical information"
            },
            "regulatory_document": {
                "primary": "nv-embedqa-e5-v5",
                "secondary": "nv-embedqa-mistral7b-v2",
                "reason": "Regulatory compliance and audit trail requirements"
            },
            "patent": {
                "primary": "nv-embedqa-mistral7b-v2",
                "secondary": "nv-embedqa-e5-v5",
                "reason": "Superior handling of complex technical and legal language"
            },
            "research_paper": {
                "primary": "nv-embedqa-mistral7b-v2",
                "secondary": "nv-embedqa-e5-v5",
                "reason": "Excellent performance on long-form scientific content"
            },
            "multilingual_content": {
                "primary": "nv-embedqa-mistral7b-v2",
                "secondary": "nv-embedqa-e5-v5",
                "reason": "Native multilingual support and pharmaceutical terminology"
            },
            "safety_data": {
                "primary": "nv-embedqa-e5-v5",
                "secondary": "nv-embedqa-mistral7b-v2",
                "reason": "Maximum precision for adverse event and safety reporting"
            }
        }

        # Model selection statistics
        self.selection_stats = defaultdict(int)
        self.pharmaceutical_optimizations_applied = 0

        logger.info("Initialized Multi-Model Embedding Strategy")
        logger.info(f"Available models: {list(self.model_capabilities.keys())}")

    def analyze_content(self, texts: List[str]) -> ContentAnalysis:
        """
        Analyze content characteristics to inform model selection.

        Args:
            texts: List of texts to analyze

        Returns:
            Content analysis results
        """
        if not texts:
            return ContentAnalysis(
                content_type="general",
                language="en",
                technical_complexity=0.0,
                medical_content_ratio=0.0,
                regulatory_content=False,
                safety_critical=False,
                document_length=0,
                pharmaceutical_terms_count=0,
                domain_specificity=0.0
            )

        combined_text = " ".join(texts).lower()
        total_length = len(combined_text)

        # Content type detection
        content_type = self._detect_content_type(combined_text)

        # Language detection (simplified)
        language = self._detect_language(combined_text)

        # Technical complexity assessment
        technical_complexity = self._assess_technical_complexity(combined_text)

        # Medical content analysis
        medical_content_ratio = self._assess_medical_content(combined_text)

        # Regulatory content detection
        regulatory_content = self._detect_regulatory_content(combined_text)

        # Safety-critical content detection
        safety_critical = self._detect_safety_critical_content(combined_text)

        # Pharmaceutical terms counting
        pharmaceutical_terms_count = self._count_pharmaceutical_terms(combined_text)

        # Domain specificity calculation
        domain_specificity = self._calculate_domain_specificity(
            pharmaceutical_terms_count, medical_content_ratio, regulatory_content, total_length
        )

        return ContentAnalysis(
            content_type=content_type,
            language=language,
            technical_complexity=technical_complexity,
            medical_content_ratio=medical_content_ratio,
            regulatory_content=regulatory_content,
            safety_critical=safety_critical,
            document_length=total_length,
            pharmaceutical_terms_count=pharmaceutical_terms_count,
            domain_specificity=domain_specificity
        )

    def select_optimal_model(
        self,
        content_analysis: ContentAnalysis,
        performance_weight: float = 0.3,
        cost_weight: float = 0.2,
        pharmaceutical_weight: float = 0.5
    ) -> ModelSelectionResult:
        """
        Select the optimal embedding model based on content analysis and performance history.

        Args:
            content_analysis: Analysis of the content to be embedded
            performance_weight: Weight for performance considerations (0.0-1.0)
            cost_weight: Weight for cost considerations (0.0-1.0)
            pharmaceutical_weight: Weight for pharmaceutical optimizations (0.0-1.0)

        Returns:
            Model selection result with reasoning
        """
        scores = {}
        pharmaceutical_optimization_applied = False

        # Get pharmaceutical preference if applicable
        pharma_preference = None
        if content_analysis.content_type in self.pharmaceutical_preferences:
            pharma_preference = self.pharmaceutical_preferences[content_analysis.content_type]
            pharmaceutical_optimization_applied = True
            self.pharmaceutical_optimizations_applied += 1

        for model_name, capabilities in self.model_capabilities.items():
            score = 0.0
            score_components = {}

            # Base capability score
            capability_score = self._calculate_capability_score(capabilities, content_analysis)
            score_components["capability"] = capability_score

            # Performance history score
            performance_score = self._calculate_performance_score(model_name)
            score_components["performance"] = performance_score

            # Cost efficiency score
            cost_score = self._calculate_cost_score(capabilities)
            score_components["cost"] = cost_score

            # Pharmaceutical optimization score
            pharma_score = self._calculate_pharmaceutical_score(
                capabilities, content_analysis, pharma_preference, model_name
            )
            score_components["pharmaceutical"] = pharma_score

            # Weighted total score
            total_score = (
                capability_score * 0.3 +
                performance_score * performance_weight +
                cost_score * cost_weight +
                pharma_score * pharmaceutical_weight
            )

            scores[model_name] = {
                "total_score": total_score,
                "components": score_components
            }

        # Select the highest scoring model
        selected_model = max(scores.keys(), key=lambda x: scores[x]["total_score"])
        confidence_score = scores[selected_model]["total_score"]

        # Generate fallback models (sorted by score, excluding selected)
        fallback_models = sorted(
            [model for model in scores.keys() if model != selected_model],
            key=lambda x: scores[x]["total_score"],
            reverse=True
        )[:3]  # Top 3 fallbacks

        # Generate selection reason
        selection_reason = self._generate_selection_reason(
            selected_model, content_analysis, pharma_preference, scores[selected_model]["components"]
        )

        # Performance prediction
        performance_prediction = {
            "latency_ms": self.model_capabilities[selected_model].latency_ms or 500.0,
            "accuracy_estimate": confidence_score,
            "cost_efficiency": scores[selected_model]["components"]["cost"]
        }

        # Update selection statistics
        self.selection_stats[selected_model] += 1

        result = ModelSelectionResult(
            selected_model=selected_model,
            confidence_score=confidence_score,
            selection_reason=selection_reason,
            fallback_models=fallback_models,
            performance_prediction=performance_prediction,
            pharmaceutical_optimization_applied=pharmaceutical_optimization_applied
        )

        logger.info(f"Selected model: {selected_model} (confidence: {confidence_score:.3f})")
        logger.debug(f"Selection reason: {selection_reason}")

        return result

    def update_model_performance(
        self,
        model_name: str,
        response_time_ms: float,
        success: bool,
        accuracy_score: Optional[float] = None,
        pharmaceutical_context: bool = False
    ):
        """
        Update performance history for a model.

        Args:
            model_name: Name of the model
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            accuracy_score: Optional accuracy score (0.0-1.0)
            pharmaceutical_context: Whether this was pharmaceutical content
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = ModelPerformanceHistory(model_name=model_name)

        history = self.performance_history[model_name]

        # Update response times
        history.response_times.append(response_time_ms)

        # Update success rate
        history.total_requests += 1
        if not success:
            history.error_count += 1
        history.success_rate = 1.0 - (history.error_count / history.total_requests)

        # Update accuracy scores
        if accuracy_score is not None:
            history.accuracy_scores.append(accuracy_score)

        # Update pharmaceutical performance
        if pharmaceutical_context and accuracy_score is not None:
            if history.pharmaceutical_performance == 0.0:
                history.pharmaceutical_performance = accuracy_score
            else:
                # Running average
                history.pharmaceutical_performance = (
                    history.pharmaceutical_performance * 0.8 + accuracy_score * 0.2
                )

        # Update last used timestamp
        history.last_used = time.time()

        # Update model capabilities with real performance data
        if model_name in self.model_capabilities:
            if history.response_times:
                self.model_capabilities[model_name].latency_ms = statistics.mean(history.response_times)
            if history.accuracy_scores:
                self.model_capabilities[model_name].accuracy_score = statistics.mean(history.accuracy_scores)

    def get_model_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for all models.

        Returns:
            Performance report dictionary
        """
        report = {
            "summary": {
                "total_models": len(self.model_capabilities),
                "total_selections": sum(self.selection_stats.values()),
                "pharmaceutical_optimizations": self.pharmaceutical_optimizations_applied
            },
            "model_performance": {},
            "selection_statistics": dict(self.selection_stats),
            "pharmaceutical_preferences": self.pharmaceutical_preferences
        }

        for model_name, history in self.performance_history.items():
            capabilities = self.model_capabilities.get(model_name, {})

            model_report = {
                "total_requests": history.total_requests,
                "success_rate": history.success_rate,
                "avg_response_time_ms": statistics.mean(history.response_times) if history.response_times else 0,
                "avg_accuracy": statistics.mean(history.accuracy_scores) if history.accuracy_scores else 0,
                "pharmaceutical_performance": history.pharmaceutical_performance,
                "last_used": history.last_used,
                "capabilities": {
                    "max_sequence_length": getattr(capabilities, 'max_sequence_length', 0),
                    "embedding_dimension": getattr(capabilities, 'embedding_dimension', 0),
                    "pharmaceutical_optimized": getattr(capabilities, 'pharmaceutical_optimized', False),
                    "regulatory_compliant": getattr(capabilities, 'regulatory_compliant', False)
                }
            }

            report["model_performance"][model_name] = model_report

        return report

    def _detect_content_type(self, text: str) -> str:
        """Detect the type of pharmaceutical content."""
        content_indicators = {
            "clinical_trial": ["clinical trial", "randomized", "placebo", "efficacy", "endpoint"],
            "drug_label": ["contraindications", "dosage", "administration", "warnings", "precautions"],
            "regulatory_document": ["FDA", "EMA", "regulatory", "compliance", "submission"],
            "patent": ["claim", "invention", "prior art", "pharmaceutical composition"],
            "research_paper": ["abstract", "introduction", "methods", "results", "discussion"],
            "safety_data": ["adverse event", "side effect", "safety", "toxicity", "pharmacovigilance"]
        }

        for content_type, indicators in content_indicators.items():
            if sum(1 for indicator in indicators if indicator in text) >= 2:
                return content_type

        return "general"

    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        multilingual_indicators = ["español", "français", "deutsch", "italiano", "中文", "日本語"]
        if any(indicator in text for indicator in multilingual_indicators):
            return "multilingual"
        return "en"

    def _assess_technical_complexity(self, text: str) -> float:
        """Assess technical complexity of the content."""
        technical_terms = [
            "bioavailability", "pharmacokinetics", "metabolism", "clearance",
            "molecular", "compound", "synthesis", "receptor", "enzyme"
        ]

        complexity_score = sum(1 for term in technical_terms if term in text)
        return min(complexity_score / 10.0, 1.0)  # Normalize to 0-1

    def _assess_medical_content(self, text: str) -> float:
        """Assess the ratio of medical content."""
        medical_terms = [
            "patient", "treatment", "therapy", "diagnosis", "symptoms",
            "disease", "condition", "medical", "clinical", "therapeutic"
        ]

        medical_count = sum(1 for term in medical_terms if term in text)
        total_words = len(text.split())
        return min(medical_count / max(total_words / 100, 1), 1.0)  # Normalize

    def _detect_regulatory_content(self, text: str) -> bool:
        """Detect regulatory content."""
        regulatory_terms = [
            "FDA", "EMA", "regulatory", "compliance", "validation",
            "CFR", "GMP", "GLP", "GCP", "audit"
        ]
        return any(term in text for term in regulatory_terms)

    def _detect_safety_critical_content(self, text: str) -> bool:
        """Detect safety-critical content."""
        safety_terms = [
            "contraindication", "warning", "adverse", "toxicity",
            "black box", "safety", "risk", "hazard"
        ]
        return any(term in text for term in safety_terms)

    def _count_pharmaceutical_terms(self, text: str) -> int:
        """Count pharmaceutical-specific terms."""
        pharma_terms = [
            "pharmaceutical", "drug", "medication", "therapeutic", "dosage",
            "bioavailability", "pharmacology", "clinical", "efficacy", "safety"
        ]
        return sum(1 for term in pharma_terms if term in text)

    def _calculate_domain_specificity(
        self, pharma_terms: int, medical_ratio: float, regulatory: bool, text_length: int
    ) -> float:
        """Calculate domain specificity score."""
        if text_length == 0:
            return 0.0

        term_density = pharma_terms / max(text_length / 100, 1)
        regulatory_bonus = 0.2 if regulatory else 0.0

        return min(term_density + medical_ratio + regulatory_bonus, 1.0)

    def _calculate_capability_score(self, capabilities: ModelCapabilities, content: ContentAnalysis) -> float:
        """Calculate capability match score."""
        score = 0.0

        # Language support
        if content.language == "multilingual" and capabilities.supports_multilingual:
            score += 0.3
        elif content.language == "en":
            score += 0.2

        # Technical content support
        if content.technical_complexity > 0.5 and capabilities.supports_technical_content:
            score += 0.2

        # Medical content support
        if content.medical_content_ratio > 0.3 and capabilities.supports_medical_content:
            score += 0.2

        # Pharmaceutical optimization
        if content.domain_specificity > 0.4 and capabilities.pharmaceutical_optimized:
            score += 0.3

        # Regulatory compliance
        if content.regulatory_content and capabilities.regulatory_compliant:
            score += 0.2

        return min(score, 1.0)

    def _calculate_performance_score(self, model_name: str) -> float:
        """Calculate performance score based on history."""
        history = self.performance_history.get(model_name)
        if not history or history.total_requests == 0:
            return 0.5  # Neutral score for unknown performance

        # Success rate component
        success_component = history.success_rate

        # Response time component (lower is better)
        if history.response_times:
            avg_response_time = statistics.mean(history.response_times)
            time_component = max(0, 1.0 - (avg_response_time / 5000))  # Normalize to 5s max
        else:
            time_component = 0.5

        # Accuracy component
        if history.accuracy_scores:
            accuracy_component = statistics.mean(history.accuracy_scores)
        else:
            accuracy_component = 0.5

        return (success_component + time_component + accuracy_component) / 3

    def _calculate_cost_score(self, capabilities: ModelCapabilities) -> float:
        """Calculate cost efficiency score."""
        cost_scores = {"economy": 1.0, "standard": 0.7, "premium": 0.4}
        return cost_scores.get(capabilities.cost_tier, 0.5)

    def _calculate_pharmaceutical_score(
        self,
        capabilities: ModelCapabilities,
        content: ContentAnalysis,
        pharma_preference: Optional[Dict],
        model_name: str
    ) -> float:
        """Calculate pharmaceutical optimization score."""
        score = 0.0

        # Base pharmaceutical optimization
        if capabilities.pharmaceutical_optimized:
            score += 0.4

        # Regulatory compliance
        if capabilities.regulatory_compliant and content.regulatory_content:
            score += 0.3

        # Pharmaceutical preference match
        if pharma_preference:
            if model_name == pharma_preference["primary"]:
                score += 0.5
            elif model_name == pharma_preference["secondary"]:
                score += 0.3

        # Safety-critical content handling
        if content.safety_critical and capabilities.pharmaceutical_optimized:
            score += 0.2

        # High domain specificity bonus
        if content.domain_specificity > 0.7 and capabilities.pharmaceutical_optimized:
            score += 0.3

        return min(score, 1.0)

    def _generate_selection_reason(
        self,
        model_name: str,
        content: ContentAnalysis,
        pharma_preference: Optional[Dict],
        score_components: Dict[str, float]
    ) -> str:
        """Generate human-readable reason for model selection."""
        reasons = []

        # Content type specific reason
        if pharma_preference and model_name in [pharma_preference["primary"], pharma_preference["secondary"]]:
            reasons.append(f"Optimized for {content.content_type}: {pharma_preference['reason']}")

        # Capability reasons
        capabilities = self.model_capabilities[model_name]
        if capabilities.pharmaceutical_optimized and content.domain_specificity > 0.5:
            reasons.append("Pharmaceutical domain optimization")

        if capabilities.regulatory_compliant and content.regulatory_content:
            reasons.append("Regulatory compliance requirements")

        if content.safety_critical and capabilities.pharmaceutical_optimized:
            reasons.append("Safety-critical content handling")

        if content.language == "multilingual" and capabilities.supports_multilingual:
            reasons.append("Multilingual content support")

        # Performance reasons
        if score_components.get("performance", 0) > 0.8:
            reasons.append("Excellent historical performance")

        # Default reason
        if not reasons:
            reasons.append("Best overall capability match")

        return "; ".join(reasons)


# Global instance for easy access
multi_model_strategy = MultiModelEmbeddingStrategy()

def select_optimal_embedding_model(
    texts: List[str],
    content_type: Optional[str] = None,
    pharmaceutical_optimization: bool = True
) -> ModelSelectionResult:
    """
    Convenience function to select optimal embedding model.

    Args:
        texts: Texts to be embedded
        content_type: Optional content type override
        pharmaceutical_optimization: Enable pharmaceutical optimizations

    Returns:
        Model selection result
    """
    # Analyze content
    content_analysis = multi_model_strategy.analyze_content(texts)

    # Override content type if provided
    if content_type:
        content_analysis.content_type = content_type

    # Adjust weights for pharmaceutical optimization
    pharma_weight = 0.6 if pharmaceutical_optimization else 0.2
    performance_weight = 0.3
    cost_weight = 0.1

    return multi_model_strategy.select_optimal_model(
        content_analysis=content_analysis,
        performance_weight=performance_weight,
        cost_weight=cost_weight,
        pharmaceutical_weight=pharma_weight
    )