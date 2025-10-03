"""
NVIDIA NeMo Reranking Service

Advanced reranking service using NVIDIA NeMo Retriever reranking models
optimized for pharmaceutical and medical content retrieval.

Key Features:
- NV-RerankQA-Mistral4B-v3 model integration
- Pharmaceutical domain-specific reranking optimization
- Cross-modal reranking (text, tables, charts)
- Hybrid local+PubMed content reranking
- Confidence scoring and explanation generation
- Batch processing with performance optimization
- Integration with existing RAG pipeline

<<use_mcp microsoft-learn>>

Based on latest NVIDIA NeMo Retriever reranking patterns and best practices.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from langchain_core.documents import Document

from .nemo_retriever_client import create_nemo_client
from .nemo_retriever_client import NeMoRetrieverClient

logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """Result from reranking operation."""

    success: bool
    reranked_documents: Optional[List[Document]] = None
    scores: Optional[List[float]] = None
    explanations: Optional[List[str]] = None
    model_used: Optional[str] = None
    processing_time_ms: float = 0.0
    original_count: int = 0
    returned_count: int = 0
    error: Optional[str] = None


@dataclass
class RerankingConfig:
    """Configuration for reranking operations."""

    model: str = "nv-rerankqa-mistral4b-v3"
    top_k: Optional[int] = None  # Return all by default
    min_score_threshold: float = 0.0
    max_documents: int = 1000
    batch_size: int = 50
    enable_explanations: bool = False
    pharmaceutical_optimization: bool = True
    cross_modal_weighting: bool = True
    preserve_original_order_on_tie: bool = True


@dataclass
class PharmaceuticalRerankingContext:
    """Context for pharmaceutical-specific reranking optimizations."""

    query_type: str = "general"  # "safety", "efficacy", "dosing", "mechanism", "regulatory"
    content_priorities: Dict[str, float] = field(default_factory=dict)
    safety_critical: bool = False
    requires_recent_data: bool = False
    regulatory_context: bool = False


class NeMoRerankingService:
    """
    Advanced reranking service using NVIDIA NeMo Retriever reranking models.

    Provides intelligent document reranking with pharmaceutical domain optimization,
    cross-modal content handling, and confidence scoring.
    """

    # Pharmaceutical query type patterns for optimization
    PHARMACEUTICAL_QUERY_PATTERNS = {
        "safety": [
            "adverse",
            "side effect",
            "contraindication",
            "warning",
            "safety",
            "toxicity",
            "overdose",
            "interaction",
            "risk",
            "caution",
        ],
        "efficacy": [
            "efficacy",
            "effectiveness",
            "response",
            "outcome",
            "benefit",
            "improvement",
            "success",
            "cure",
            "treatment",
            "therapeutic",
        ],
        "dosing": [
            "dosage",
            "dose",
            "administration",
            "frequency",
            "schedule",
            "titration",
            "adjustment",
            "concentration",
            "amount",
            "regimen",
        ],
        "mechanism": [
            "mechanism",
            "action",
            "pathway",
            "target",
            "receptor",
            "pharmacology",
            "binding",
            "metabolism",
            "clearance",
            "kinetics",
        ],
        "regulatory": [
            "approval",
            "fda",
            "ema",
            "indication",
            "labeling",
            "guideline",
            "regulation",
            "compliance",
            "submission",
            "review",
        ],
    }

    # Content type weights for cross-modal reranking
    CONTENT_TYPE_WEIGHTS = {
        "text": 1.0,
        "table": 1.2,  # Tables often contain critical pharmaceutical data
        "chart": 1.1,  # Charts provide visual insights
        "image": 0.9,  # Images less reliable for text-based queries
        "formula": 1.3,  # Chemical formulas highly relevant
        "clinical_data": 1.4,  # Clinical data prioritized for medical queries
        "regulatory": 1.5,  # Regulatory info highly important for compliance
        "safety": 1.6,  # Safety information gets highest priority
    }

    # Source authority weights for pharmaceutical content
    SOURCE_AUTHORITY_WEIGHTS = {
        "pubmed": 1.3,  # PubMed articles are authoritative
        "fda_label": 1.5,  # FDA drug labels are highly authoritative
        "clinical_trial": 1.4,  # Clinical trial data is reliable
        "patent": 0.9,  # Patents less authoritative for clinical use
        "textbook": 1.1,  # Medical textbooks are reliable
        "guideline": 1.4,  # Clinical guidelines are important
        "local_doc": 1.0,  # Local documents baseline
        "unknown": 1.0,  # Unknown sources baseline
    }

    def __init__(self, nemo_client: Optional[NeMoRetrieverClient] = None, config: Optional[RerankingConfig] = None):
        """
        Initialize NeMo Reranking Service.

        Args:
            nemo_client: Pre-configured NeMo client (creates new if None)
            config: Reranking configuration
        """
        self.nemo_client = nemo_client
        self.config = config or RerankingConfig()

        # Performance metrics
        self.metrics = {
            "total_reranking_operations": 0,
            "total_documents_reranked": 0,
            "total_processing_time_ms": 0.0,
            "pharmaceutical_optimizations": 0,
            "cross_modal_operations": 0,
            "average_score_improvement": 0.0,
        }

    async def _ensure_nemo_client(self) -> NeMoRetrieverClient:
        """Ensure NeMo client is available."""
        if not self.nemo_client:
            self.nemo_client = await create_nemo_client()
        return self.nemo_client

    def _analyze_pharmaceutical_query(self, query: str) -> PharmaceuticalRerankingContext:
        """
        Analyze query to determine pharmaceutical context and optimization strategy.

        Args:
            query: Search query

        Returns:
            PharmaceuticalRerankingContext with optimization recommendations
        """
        query_lower = query.lower()

        # Determine query type
        query_type = "general"
        max_matches = 0

        for q_type, patterns in self.PHARMACEUTICAL_QUERY_PATTERNS.items():
            matches = sum(1 for pattern in patterns if pattern in query_lower)
            if matches > max_matches:
                max_matches = matches
                query_type = q_type

        # Determine content priorities based on query type
        content_priorities = {}
        if query_type == "safety":
            content_priorities = {"safety": 1.8, "regulatory": 1.6, "clinical_data": 1.5, "text": 1.0}
        elif query_type == "efficacy":
            content_priorities = {"clinical_data": 1.6, "table": 1.4, "chart": 1.3, "text": 1.0}
        elif query_type == "dosing":
            content_priorities = {"table": 1.5, "regulatory": 1.4, "text": 1.0}
        elif query_type == "mechanism":
            content_priorities = {"formula": 1.5, "chart": 1.3, "text": 1.0}
        else:
            content_priorities = self.CONTENT_TYPE_WEIGHTS.copy()

        # Determine critical flags
        safety_critical = any(
            term in query_lower
            for term in [
                "safety",
                "adverse",
                "toxic",
                "warning",
                "contraindication",
                "black box",
                "death",
                "serious",
                "severe",
            ]
        )

        requires_recent_data = any(
            term in query_lower for term in ["recent", "latest", "new", "current", "updated", "2023", "2024"]
        )

        regulatory_context = any(
            term in query_lower for term in ["fda", "ema", "approval", "regulation", "guideline", "compliance"]
        )

        return PharmaceuticalRerankingContext(
            query_type=query_type,
            content_priorities=content_priorities,
            safety_critical=safety_critical,
            requires_recent_data=requires_recent_data,
            regulatory_context=regulatory_context,
        )

    def _calculate_source_authority_score(self, document: Document) -> float:
        """Calculate authority score based on document source."""
        metadata = document.metadata

        # Determine source type
        source_type = metadata.get("source_type", "unknown")

        # Check for specific source indicators
        if "pubmed" in str(metadata.get("source", "")).lower():
            source_type = "pubmed"
        elif "fda" in str(metadata.get("source", "")).lower():
            source_type = "fda_label"
        elif metadata.get("clinical_trial_id"):
            source_type = "clinical_trial"
        elif "patent" in str(metadata.get("source", "")).lower():
            source_type = "patent"

        return self.SOURCE_AUTHORITY_WEIGHTS.get(source_type, 1.0)

    def _calculate_content_type_score(
        self, document: Document, pharma_context: PharmaceuticalRerankingContext
    ) -> float:
        """Calculate content type score based on pharmaceutical context."""
        metadata = document.metadata
        content_type = metadata.get("element_type", "text")

        # Map element types to our content categories
        type_mapping = {
            "Table": "table",
            "chart": "chart",
            "image": "image",
            "text": "text",
            "Title": "text",
            "NarrativeText": "text",
        }

        mapped_type = type_mapping.get(content_type, "text")

        # Check for pharmaceutical-specific content types
        content = document.page_content.lower()
        if any(term in content for term in ["adverse", "safety", "contraindication"]):
            mapped_type = "safety"
        elif any(term in content for term in ["clinical trial", "efficacy", "outcome"]):
            mapped_type = "clinical_data"
        elif any(term in content for term in ["fda", "approval", "regulation"]):
            mapped_type = "regulatory"
        elif any(term in content for term in ["chemical formula", "molecular", "structure"]):
            mapped_type = "formula"

        # Get weight from pharmaceutical context or default
        return pharma_context.content_priorities.get(mapped_type, self.CONTENT_TYPE_WEIGHTS.get(mapped_type, 1.0))

    def _calculate_recency_score(self, document: Document, requires_recent: bool) -> float:
        """Calculate recency score for documents."""
        if not requires_recent:
            return 1.0

        metadata = document.metadata

        # Try to extract publication date
        pub_date = metadata.get("publication_date") or metadata.get("date")

        if pub_date:
            try:
                # Simple year extraction (could be enhanced)
                year = int(str(pub_date)[:4])
                current_year = 2024  # Could use datetime.now().year

                if year >= current_year - 1:  # Very recent
                    return 1.3
                elif year >= current_year - 3:  # Recent
                    return 1.1
                elif year >= current_year - 5:  # Moderately recent
                    return 1.0
                else:  # Older
                    return 0.9
            except (ValueError, TypeError):
                pass

        # No date information, neutral score
        return 1.0

    async def rerank_documents(
        self, query: str, documents: List[Document], return_top_k: Optional[int] = None
    ) -> RerankingResult:
        """
        Rerank documents based on relevance to query using NeMo reranking models.

        Args:
            query: Search query
            documents: List of documents to rerank
            return_top_k: Number of top documents to return (None for all)

        Returns:
            RerankingResult with reranked documents and scores
        """
        start_time = time.time()
        original_count = len(documents)

        try:
            if not documents:
                return RerankingResult(
                    success=True,
                    reranked_documents=[],
                    scores=[],
                    original_count=0,
                    returned_count=0,
                    model_used=self.config.model,
                    processing_time_ms=0.0,
                )

            # Limit documents if too many
            if len(documents) > self.config.max_documents:
                documents = documents[: self.config.max_documents]
                logger.warning(f"Truncated documents from {original_count} to {len(documents)}")

            # Analyze pharmaceutical context
            pharma_context = None
            if self.config.pharmaceutical_optimization:
                pharma_context = self._analyze_pharmaceutical_query(query)
                self.metrics["pharmaceutical_optimizations"] += 1

            # Get base reranking scores from NeMo
            nemo_result = await self._rerank_with_nemo(query, documents)

            if not nemo_result.success:
                return RerankingResult(
                    success=False,
                    error=nemo_result.error,
                    model_used=self.config.model,
                    original_count=original_count,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            base_scores = nemo_result.scores

            # Apply pharmaceutical and cross-modal optimizations
            enhanced_scores = self._apply_pharmaceutical_optimizations(documents, base_scores, pharma_context)

            # Sort documents by enhanced scores
            scored_docs = list(zip(documents, enhanced_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Apply score threshold filtering
            if self.config.min_score_threshold > 0:
                scored_docs = [(doc, score) for doc, score in scored_docs if score >= self.config.min_score_threshold]

            # Apply top-k filtering
            top_k = return_top_k or self.config.top_k
            if top_k:
                scored_docs = scored_docs[:top_k]

            reranked_documents = [doc for doc, _ in scored_docs]
            final_scores = [score for _, score in scored_docs]

            # Generate explanations if requested
            explanations = None
            if self.config.enable_explanations:
                explanations = self._generate_explanations(query, reranked_documents, final_scores, pharma_context)

            processing_time = (time.time() - start_time) * 1000

            # Update metrics
            self._update_metrics(len(documents), processing_time, base_scores, enhanced_scores)

            return RerankingResult(
                success=True,
                reranked_documents=reranked_documents,
                scores=final_scores,
                explanations=explanations,
                model_used=self.config.model,
                processing_time_ms=processing_time,
                original_count=original_count,
                returned_count=len(reranked_documents),
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Reranking failed: {e}")

            return RerankingResult(
                success=False,
                error=str(e),
                model_used=self.config.model,
                processing_time_ms=processing_time,
                original_count=original_count,
                returned_count=0,
            )

    async def _rerank_with_nemo(self, query: str, documents: List[Document]) -> RerankingResult:
        """Get base reranking scores from NeMo Retriever."""
        try:
            client = await self._ensure_nemo_client()

            # Extract text content from documents
            passages = [doc.page_content for doc in documents]

            # Call NeMo reranking service
            response = await client.rerank_passages(
                query=query, passages=passages, model=self.config.model, use_langchain=True
            )

            if response.success:
                # Extract scores from response
                reranked_data = response.data["reranked_passages"]

                # Handle different response formats
                if isinstance(reranked_data, list) and reranked_data:
                    if isinstance(reranked_data[0], dict):
                        # Format: [{"text": "...", "score": 0.9, "index": 0}, ...]
                        scores = [0.0] * len(documents)
                        for item in reranked_data:
                            idx = item.get("index", 0)
                            if 0 <= idx < len(scores):
                                scores[idx] = item.get("score", 0.0)
                    else:
                        # Format: [0.9, 0.7, 0.5, ...]
                        scores = reranked_data[: len(documents)]
                else:
                    # Fallback: uniform scores
                    scores = [0.5] * len(documents)

                return RerankingResult(success=True, scores=scores, model_used=self.config.model)
            else:
                return RerankingResult(success=False, error=response.error, model_used=self.config.model)

        except Exception as e:
            return RerankingResult(success=False, error=str(e), model_used=self.config.model)

    def _apply_pharmaceutical_optimizations(
        self,
        documents: List[Document],
        base_scores: List[float],
        pharma_context: Optional[PharmaceuticalRerankingContext],
    ) -> List[float]:
        """Apply pharmaceutical domain optimizations to reranking scores."""
        if not pharma_context:
            return base_scores

        enhanced_scores = []

        for doc, base_score in zip(documents, base_scores):
            enhanced_score = base_score

            # Apply source authority weighting
            authority_score = self._calculate_source_authority_score(doc)
            enhanced_score *= authority_score

            # Apply content type weighting
            if self.config.cross_modal_weighting:
                content_score = self._calculate_content_type_score(doc, pharma_context)
                enhanced_score *= content_score
                self.metrics["cross_modal_operations"] += 1

            # Apply recency weighting
            recency_score = self._calculate_recency_score(doc, pharma_context.requires_recent_data)
            enhanced_score *= recency_score

            # Safety-critical boost
            if pharma_context.safety_critical:
                content_lower = doc.page_content.lower()
                if any(term in content_lower for term in ["safety", "adverse", "warning"]):
                    enhanced_score *= 1.3

            # Regulatory context boost
            if pharma_context.regulatory_context:
                metadata = doc.metadata
                if metadata.get("source_type") == "regulatory" or "fda" in str(metadata.get("source", "")).lower():
                    enhanced_score *= 1.2

            enhanced_scores.append(enhanced_score)

        return enhanced_scores

    def _generate_explanations(
        self,
        query: str,
        documents: List[Document],
        scores: List[float],
        pharma_context: Optional[PharmaceuticalRerankingContext],
    ) -> List[str]:
        """Generate explanations for reranking decisions."""
        explanations = []

        for doc, score in zip(documents, scores):
            explanation_parts = []

            # Base relevance
            explanation_parts.append(f"Base relevance score: {score:.3f}")

            # Source authority
            authority = self._calculate_source_authority_score(doc)
            if authority != 1.0:
                source_type = doc.metadata.get("source_type", "unknown")
                explanation_parts.append(f"Source authority ({source_type}): {authority:.2f}x")

            # Content type
            if pharma_context and self.config.cross_modal_weighting:
                content_score = self._calculate_content_type_score(doc, pharma_context)
                if content_score != 1.0:
                    content_type = doc.metadata.get("element_type", "text")
                    explanation_parts.append(f"Content type ({content_type}): {content_score:.2f}x")

            # Pharmaceutical context
            if pharma_context:
                if pharma_context.safety_critical and "safety" in doc.page_content.lower():
                    explanation_parts.append("Safety-critical content boost")

                if pharma_context.regulatory_context and "regulatory" in str(doc.metadata.get("source", "")):
                    explanation_parts.append("Regulatory context boost")

            explanation = "; ".join(explanation_parts)
            explanations.append(explanation)

        return explanations

    def _update_metrics(
        self, num_documents: int, processing_time_ms: float, base_scores: List[float], enhanced_scores: List[float]
    ):
        """Update performance metrics."""
        self.metrics["total_reranking_operations"] += 1
        self.metrics["total_documents_reranked"] += num_documents
        self.metrics["total_processing_time_ms"] += processing_time_ms

        # Calculate score improvement
        if base_scores and enhanced_scores:
            base_avg = sum(base_scores) / len(base_scores)
            enhanced_avg = sum(enhanced_scores) / len(enhanced_scores)
            improvement = enhanced_avg - base_avg
            self.metrics["average_score_improvement"] = (self.metrics["average_score_improvement"] + improvement) / 2

    async def rerank_hybrid_results(
        self,
        query: str,
        local_documents: List[Document],
        pubmed_documents: List[Document],
        local_weight: float = 1.0,
        pubmed_weight: float = 1.3,
    ) -> RerankingResult:
        """
        Rerank hybrid results from local documents and PubMed with source weighting.

        Args:
            query: Search query
            local_documents: Documents from local knowledge base
            pubmed_documents: Documents from PubMed
            local_weight: Weight for local documents
            pubmed_weight: Weight for PubMed documents

        Returns:
            RerankingResult with hybrid reranked documents
        """
        # Mark document sources
        for doc in local_documents:
            if "source_type" not in doc.metadata:
                doc.metadata["source_type"] = "local_doc"

        for doc in pubmed_documents:
            if "source_type" not in doc.metadata:
                doc.metadata["source_type"] = "pubmed"

        # Combine all documents
        all_documents = local_documents + pubmed_documents

        # Rerank all documents together
        result = await self.rerank_documents(query, all_documents)

        if result.success and result.scores:
            # Apply source-specific weighting
            enhanced_scores = []
            for doc, score in zip(result.reranked_documents, result.scores):
                if doc.metadata.get("source_type") == "pubmed":
                    enhanced_scores.append(score * pubmed_weight)
                else:
                    enhanced_scores.append(score * local_weight)

            # Re-sort with enhanced scores
            scored_docs = list(zip(result.reranked_documents, enhanced_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            result.reranked_documents = [doc for doc, _ in scored_docs]
            result.scores = [score for _, score in scored_docs]

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self.metrics.copy()

        if metrics["total_reranking_operations"] > 0:
            metrics["avg_processing_time_ms"] = (
                metrics["total_processing_time_ms"] / metrics["total_reranking_operations"]
            )
            metrics["avg_documents_per_operation"] = (
                metrics["total_documents_reranked"] / metrics["total_reranking_operations"]
            )
        else:
            metrics["avg_processing_time_ms"] = 0.0
            metrics["avg_documents_per_operation"] = 0.0

        return metrics

    def get_pharmaceutical_query_recommendations(self, query: str) -> Dict[str, Any]:
        """Get recommendations for optimizing pharmaceutical queries."""
        context = self._analyze_pharmaceutical_query(query)

        return {
            "detected_query_type": context.query_type,
            "recommended_content_priorities": context.content_priorities,
            "safety_critical": context.safety_critical,
            "requires_recent_data": context.requires_recent_data,
            "regulatory_context": context.regulatory_context,
            "optimization_suggestions": self._get_optimization_suggestions(context),
        }

    def _get_optimization_suggestions(self, context: PharmaceuticalRerankingContext) -> List[str]:
        """Generate optimization suggestions based on context."""
        suggestions = []

        if context.safety_critical:
            suggestions.append("Prioritize FDA-approved and regulatory documents")
            suggestions.append("Give higher weight to safety and adverse event data")

        if context.requires_recent_data:
            suggestions.append("Boost documents with recent publication dates")
            suggestions.append("Consider using latest clinical trial data")

        if context.regulatory_context:
            suggestions.append("Prioritize official regulatory guidance documents")
            suggestions.append("Weight FDA/EMA sources higher than other sources")

        if context.query_type == "dosing":
            suggestions.append("Give higher priority to dosing tables and charts")
            suggestions.append("Prioritize prescribing information documents")

        if context.query_type == "mechanism":
            suggestions.append("Prioritize pharmacology and mechanism sections")
            suggestions.append("Weight chemical formula and pathway information higher")

        return suggestions


# Convenience functions
async def create_nemo_reranking_service(
    config: Optional[RerankingConfig] = None, enable_pharmaceutical_optimization: bool = True, **kwargs
) -> NeMoRerankingService:
    """
    Factory function to create NeMo Reranking Service.

    Args:
        config: Reranking configuration
        enable_pharmaceutical_optimization: Enable pharmaceutical optimizations
        **kwargs: Additional configuration parameters

    Returns:
        Configured NeMoRerankingService
    """
    if not config:
        config = RerankingConfig(pharmaceutical_optimization=enable_pharmaceutical_optimization, **kwargs)

    service = NeMoRerankingService(config=config)

    # Ensure NeMo client is ready
    await service._ensure_nemo_client()

    return service


async def rerank_pharmaceutical_documents(
    query: str, documents: List[Document], top_k: Optional[int] = None, **kwargs
) -> RerankingResult:
    """
    Quick function to rerank pharmaceutical documents with optimized settings.

    Args:
        query: Search query
        documents: Documents to rerank
        top_k: Number of top documents to return
        **kwargs: Additional reranking parameters

    Returns:
        RerankingResult with pharmaceutical optimization
    """
    service = await create_nemo_reranking_service(**kwargs)
    return await service.rerank_documents(query, documents, top_k)


# Example usage
if __name__ == "__main__":

    async def test_reranking_service():
        """Test NeMo reranking service functionality."""
        service = await create_nemo_reranking_service()

        # Create test documents
        test_docs = [
            Document(
                page_content="Aspirin is an NSAID with anti-inflammatory and analgesic properties.",
                metadata={"source": "medical_textbook.pdf", "page": 1},
            ),
            Document(
                page_content="Clinical trial NCT123456 showed 85% efficacy in cardiovascular protection.",
                metadata={"source": "pubmed_article.pdf", "clinical_trial_id": "NCT123456"},
            ),
            Document(
                page_content="FDA warns about increased bleeding risk with aspirin use.",
                metadata={"source": "fda_safety_alert.pdf", "source_type": "regulatory"},
            ),
        ]

        query = "What are the safety concerns with aspirin?"

        try:
            result = await service.rerank_documents(query, test_docs)

            if result.success:
                print(f"Reranking successful: {result.returned_count} documents")
                print(f"Processing time: {result.processing_time_ms:.2f}ms")

                for i, (doc, score) in enumerate(zip(result.reranked_documents, result.scores)):
                    print(f"{i+1}. Score: {score:.3f} - {doc.page_content[:50]}...")

                # Get pharmaceutical recommendations
                recommendations = service.get_pharmaceutical_query_recommendations(query)
                print(f"Query type detected: {recommendations['detected_query_type']}")
                print(f"Safety critical: {recommendations['safety_critical']}")

            else:
                print(f"Reranking failed: {result.error}")

        except Exception as e:
            print(f"Test failed: {e}")

    asyncio.run(test_reranking_service())
