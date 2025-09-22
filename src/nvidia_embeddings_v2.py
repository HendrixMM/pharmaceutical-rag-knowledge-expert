"""
NVIDIA Embeddings V2 - Enhanced NeMo Integration

Backward-compatible embedding service that integrates NVIDIA NeMo Retriever services
while maintaining the existing API interface for seamless migration.

This service provides:
1. Drop-in replacement for nvidia_embeddings.py with identical interface
2. Advanced NeMo Retriever model selection and optimization
3. Pharmaceutical domain-specific embedding enhancements
4. Automatic fallback to legacy embeddings for compatibility
5. Performance monitoring and intelligent caching

Migration Strategy:
- Phase 1: Direct API compatibility (this file)
- Phase 2: Enhanced features with new optional parameters
- Phase 3: Full NeMo integration with multi-modal capabilities

<<use_mcp microsoft-learn>>
"""

import os
import logging
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import json

from langchain_core.embeddings import Embeddings

# Import both the legacy and new embedding services
from .nvidia_embeddings import NVIDIAEmbeddings as LegacyNVIDIAEmbeddings
from .nemo_embedding_service import NeMoEmbeddingService, EmbeddingConfig, PharmaceuticalContext
from .mcp_documentation_context import get_nemo_context

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingServiceMetrics:
    """Metrics for embedding service performance tracking."""
    total_requests: int = 0
    nemo_requests: int = 0
    legacy_requests: int = 0
    cache_hits: int = 0
    total_tokens_processed: int = 0
    avg_response_time_ms: float = 0.0
    last_model_used: Optional[str] = None
    pharmaceutical_optimizations_applied: int = 0

class NVIDIAEmbeddings(Embeddings):
    """
    Enhanced NVIDIA Embeddings V2 with NeMo Retriever integration.

    Provides backward compatibility with the original NVIDIAEmbeddings interface
    while leveraging advanced NeMo Retriever capabilities for improved performance
    and pharmaceutical domain optimization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 10,
        probe_on_init: Optional[bool] = None,
        fallback_model_name: Optional[str] = None,
        # New V2 parameters (optional for backward compatibility)
        enable_nemo_service: Optional[bool] = None,
        pharmaceutical_optimization: bool = True,
        content_type: str = "general",
        enable_performance_monitoring: bool = True,
        nemo_model_preference: Optional[str] = None
    ):
        """
        Initialize Enhanced NVIDIA Embeddings V2.

        Args:
            # Legacy parameters (maintained for backward compatibility)
            api_key: NVIDIA API key
            embedding_model_name: Name of the embedding model
            base_url: Base URL for NVIDIA API
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            batch_size: Number of texts to process in each batch
            probe_on_init: Whether to probe the model during initialization
            fallback_model_name: Fallback model name

            # Enhanced V2 parameters
            enable_nemo_service: Whether to use NeMo service (auto-detected if None)
            pharmaceutical_optimization: Enable pharmaceutical domain optimizations
            content_type: Type of content being processed for optimization
            enable_performance_monitoring: Track performance metrics
            nemo_model_preference: Preferred NeMo model for new service
        """

        # Store initialization parameters
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.legacy_params = {
            'api_key': api_key,
            'embedding_model_name': embedding_model_name,
            'base_url': base_url,
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'batch_size': batch_size,
            'probe_on_init': probe_on_init,
            'fallback_model_name': fallback_model_name
        }

        # Enhanced V2 configuration
        self.pharmaceutical_optimization = pharmaceutical_optimization
        self.content_type = content_type
        self.enable_performance_monitoring = enable_performance_monitoring
        self.nemo_model_preference = nemo_model_preference or os.getenv("NEMO_EMBEDDING_MODEL")

        # Determine which service to use
        self.enable_nemo_service = self._determine_service_preference(enable_nemo_service)

        # Initialize metrics tracking
        self.metrics = EmbeddingServiceMetrics()

        # Initialize services
        self.nemo_service: Optional[NeMoEmbeddingService] = None
        self.legacy_service: Optional[LegacyNVIDIAEmbeddings] = None
        self.active_service = None

        # Cache for embedding dimensions
        self._dimension: Optional[int] = None

        # Initialize the appropriate service(s)
        self._initialize_services()

        # Get MCP-enhanced documentation context for NeMo operations
        if self.enable_nemo_service:
            self._load_nemo_documentation_context()

        logger.info(f"Initialized NVIDIA Embeddings V2")
        logger.info(f"Active Service: {'NeMo' if self.enable_nemo_service else 'Legacy'}")
        logger.info(f"Pharmaceutical Optimization: {self.pharmaceutical_optimization}")
        logger.info(f"Content Type: {self.content_type}")

    def _determine_service_preference(self, enable_nemo_service: Optional[bool]) -> bool:
        """
        Determine which embedding service to use based on configuration and environment.

        Returns:
            True if NeMo service should be used, False for legacy service
        """
        # Explicit parameter takes precedence
        if enable_nemo_service is not None:
            return enable_nemo_service

        # Check environment variable
        env_preference = os.getenv("ENABLE_NEMO_EMBEDDINGS")
        if env_preference is not None:
            return env_preference.lower() in ("true", "1", "yes", "on")

        # Check if NVIDIA API key is available for NeMo service
        if not self.api_key:
            logger.warning("No NVIDIA API key available, using legacy service")
            return False

        # Check if pharmaceutical optimization is requested
        if self.pharmaceutical_optimization:
            logger.info("Pharmaceutical optimization requested, preferring NeMo service")
            return True

        # Default to NeMo service if available
        return True

    def _initialize_services(self):
        """Initialize the appropriate embedding service(s)."""

        if self.enable_nemo_service:
            try:
                # Configure NeMo embedding service
                embedding_config = EmbeddingConfig(
                    model=self.nemo_model_preference or "nv-embedqa-e5-v5",
                    batch_size=self.legacy_params.get('batch_size', 100),
                    enable_caching=True,
                    pharmaceutical_optimization=self.pharmaceutical_optimization
                )

                # Configure pharmaceutical context
                pharma_context = PharmaceuticalContext(
                    content_type=self.content_type,
                    requires_precision=self.content_type in ["drug_label", "regulatory_document", "clinical_trial"]
                )

                self.nemo_service = NeMoEmbeddingService(
                    config=embedding_config,
                    pharmaceutical_context=pharma_context
                )

                self.active_service = "nemo"
                logger.info("Successfully initialized NeMo embedding service")

            except Exception as e:
                logger.warning(f"Failed to initialize NeMo service: {e}")
                logger.info("Falling back to legacy embedding service")
                self.enable_nemo_service = False

        # Initialize legacy service (either as primary or fallback)
        if not self.enable_nemo_service or self.nemo_service is None:
            try:
                self.legacy_service = LegacyNVIDIAEmbeddings(**self.legacy_params)
                self.active_service = "legacy"
                logger.info("Successfully initialized legacy embedding service")

            except Exception as e:
                logger.error(f"Failed to initialize legacy embedding service: {e}")
                raise RuntimeError("Unable to initialize any embedding service") from e

    def _load_nemo_documentation_context(self):
        """Load latest NeMo documentation context for optimized operations."""
        try:
            context = get_nemo_context(
                operation_type="embedding",
                topic="pharmaceutical-optimization",
                pharmaceutical_focus=True
            )

            if context and context.pharmaceutical_optimized:
                logger.info("Loaded pharmaceutical-optimized NeMo documentation context")
                self.metrics.pharmaceutical_optimizations_applied += 1
            else:
                logger.debug("Standard NeMo documentation context loaded")

        except Exception as e:
            logger.warning(f"Failed to load NeMo documentation context: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            if self.enable_nemo_service and self.nemo_service:
                # Use NeMo service for enhanced embeddings
                result = self.nemo_service.embed_documents(texts)
                self.metrics.nemo_requests += 1
                self.metrics.last_model_used = getattr(self.nemo_service, 'current_model', 'nemo-service')

                if hasattr(self.nemo_service, 'pharmaceutical_optimization') and self.nemo_service.pharmaceutical_optimization:
                    self.metrics.pharmaceutical_optimizations_applied += 1

            else:
                # Use legacy service
                result = self.legacy_service.embed_documents(texts)
                self.metrics.legacy_requests += 1
                self.metrics.last_model_used = getattr(self.legacy_service, 'model_name', 'legacy-service')

            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time, len(texts))

            logger.info(f"Successfully embedded {len(texts)} documents using {self.active_service} service")
            return result

        except Exception as e:
            logger.error(f"Document embedding failed with {self.active_service} service: {e}")

            # Attempt fallback if using NeMo service
            if self.enable_nemo_service and self.legacy_service:
                logger.info("Attempting fallback to legacy service...")
                try:
                    result = self.legacy_service.embed_documents(texts)
                    self.metrics.legacy_requests += 1
                    self.metrics.last_model_used = getattr(self.legacy_service, 'model_name', 'legacy-fallback')

                    processing_time = (time.time() - start_time) * 1000
                    self._update_performance_metrics(processing_time, len(texts))

                    logger.info(f"Successfully embedded {len(texts)} documents using legacy fallback")
                    return result

                except Exception as fallback_error:
                    logger.error(f"Legacy fallback also failed: {fallback_error}")

            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            if self.enable_nemo_service and self.nemo_service:
                # Use NeMo service for enhanced embeddings
                result = self.nemo_service.embed_query(text)
                self.metrics.nemo_requests += 1
                self.metrics.last_model_used = getattr(self.nemo_service, 'current_model', 'nemo-service')

                if hasattr(self.nemo_service, 'pharmaceutical_optimization') and self.nemo_service.pharmaceutical_optimization:
                    self.metrics.pharmaceutical_optimizations_applied += 1

            else:
                # Use legacy service
                result = self.legacy_service.embed_query(text)
                self.metrics.legacy_requests += 1
                self.metrics.last_model_used = getattr(self.legacy_service, 'model_name', 'legacy-service')

            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time, 1)

            logger.debug(f"Successfully embedded query using {self.active_service} service")
            return result

        except Exception as e:
            logger.error(f"Query embedding failed with {self.active_service} service: {e}")

            # Attempt fallback if using NeMo service
            if self.enable_nemo_service and self.legacy_service:
                logger.debug("Attempting fallback to legacy service...")
                try:
                    result = self.legacy_service.embed_query(text)
                    self.metrics.legacy_requests += 1
                    self.metrics.last_model_used = getattr(self.legacy_service, 'model_name', 'legacy-fallback')

                    processing_time = (time.time() - start_time) * 1000
                    self._update_performance_metrics(processing_time, 1)

                    logger.debug("Successfully embedded query using legacy fallback")
                    return result

                except Exception as fallback_error:
                    logger.error(f"Legacy fallback also failed: {fallback_error}")

            raise

    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings from the active model.

        Returns:
            Embedding dimension, or None if it cannot be determined
        """
        if self._dimension is not None:
            return self._dimension

        try:
            if self.enable_nemo_service and self.nemo_service:
                # Try to get dimension from NeMo service
                if hasattr(self.nemo_service, 'get_embedding_dimension'):
                    dimension = self.nemo_service.get_embedding_dimension()
                else:
                    # Fallback: test embed to determine dimension
                    test_embedding = self.nemo_service.embed_query("test")
                    dimension = len(test_embedding) if test_embedding else None
            else:
                # Use legacy service
                dimension = self.legacy_service.get_embedding_dimension()

            if dimension:
                self._dimension = dimension
                logger.info(f"Determined embedding dimension: {dimension}")

            return dimension

        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test the connection to the active embedding service.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            if self.enable_nemo_service and self.nemo_service:
                # Test NeMo service
                if hasattr(self.nemo_service, 'test_connection'):
                    success = self.nemo_service.test_connection()
                else:
                    # Fallback test
                    test_embedding = self.nemo_service.embed_query("test")
                    success = test_embedding is not None and len(test_embedding) > 0

                if success:
                    logger.info("NeMo embedding service connection successful")
                    return True
                else:
                    logger.warning("NeMo service connection failed, testing legacy service...")

            # Test legacy service (either as primary or fallback)
            if self.legacy_service:
                success = self.legacy_service.test_connection()
                if success:
                    logger.info("Legacy embedding service connection successful")
                return success

            return False

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _update_performance_metrics(self, processing_time_ms: float, token_count: int):
        """Update performance tracking metrics."""
        if not self.enable_performance_monitoring:
            return

        try:
            # Update response time (running average)
            if self.metrics.total_requests > 1:
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (self.metrics.total_requests - 1) + processing_time_ms)
                    / self.metrics.total_requests
                )
            else:
                self.metrics.avg_response_time_ms = processing_time_ms

            # Update token count
            self.metrics.total_tokens_processed += token_count

        except Exception as e:
            logger.warning(f"Failed to update performance metrics: {e}")

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the active embedding service and its configuration.

        Returns:
            Dictionary containing service information
        """
        info = {
            "service_version": "v2_enhanced",
            "active_service": self.active_service,
            "nemo_service_enabled": self.enable_nemo_service,
            "pharmaceutical_optimization": self.pharmaceutical_optimization,
            "content_type": self.content_type,
            "performance_monitoring": self.enable_performance_monitoring
        }

        # Add service-specific information
        if self.enable_nemo_service and self.nemo_service:
            info["nemo_model"] = getattr(self.nemo_service, 'current_model', 'unknown')
            info["nemo_config"] = getattr(self.nemo_service, 'config', {})

        if self.legacy_service:
            info["legacy_model"] = getattr(self.legacy_service, 'model_name', 'unknown')
            info["legacy_base_url"] = getattr(self.legacy_service, 'base_url', 'unknown')

        return info

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embedding service.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.enable_performance_monitoring:
            return {"monitoring_enabled": False}

        total_requests = self.metrics.total_requests

        return {
            "monitoring_enabled": True,
            "total_requests": total_requests,
            "nemo_requests": self.metrics.nemo_requests,
            "legacy_requests": self.metrics.legacy_requests,
            "nemo_usage_percentage": (self.metrics.nemo_requests / total_requests * 100) if total_requests > 0 else 0,
            "cache_hits": self.metrics.cache_hits,
            "cache_hit_rate": (self.metrics.cache_hits / total_requests * 100) if total_requests > 0 else 0,
            "total_tokens_processed": self.metrics.total_tokens_processed,
            "avg_response_time_ms": self.metrics.avg_response_time_ms,
            "last_model_used": self.metrics.last_model_used,
            "pharmaceutical_optimizations_applied": self.metrics.pharmaceutical_optimizations_applied
        }

    def switch_to_nemo_service(self, force: bool = False) -> bool:
        """
        Switch to NeMo embedding service if available.

        Args:
            force: Force switch even if NeMo service was previously unavailable

        Returns:
            True if switch was successful, False otherwise
        """
        if self.enable_nemo_service and self.nemo_service and not force:
            logger.info("Already using NeMo service")
            return True

        try:
            if force or not self.nemo_service:
                # Re-initialize NeMo service
                self._initialize_nemo_service()

            if self.nemo_service:
                self.enable_nemo_service = True
                self.active_service = "nemo"
                logger.info("Successfully switched to NeMo embedding service")
                return True

        except Exception as e:
            logger.error(f"Failed to switch to NeMo service: {e}")

        return False

    def switch_to_legacy_service(self) -> bool:
        """
        Switch to legacy embedding service.

        Returns:
            True if switch was successful, False otherwise
        """
        if not self.legacy_service:
            logger.error("Legacy service not available")
            return False

        self.enable_nemo_service = False
        self.active_service = "legacy"
        logger.info("Switched to legacy embedding service")
        return True

    def _initialize_nemo_service(self):
        """Initialize or re-initialize the NeMo service."""
        embedding_config = EmbeddingConfig(
            model=self.nemo_model_preference or "nv-embedqa-e5-v5",
            batch_size=self.legacy_params.get('batch_size', 100),
            enable_caching=True,
            pharmaceutical_optimization=self.pharmaceutical_optimization
        )

        pharma_context = PharmaceuticalContext(
            content_type=self.content_type,
            requires_precision=self.content_type in ["drug_label", "regulatory_document", "clinical_trial"]
        )

        self.nemo_service = NeMoEmbeddingService(
            config=embedding_config,
            pharmaceutical_context=pharma_context
        )


def create_enhanced_embeddings(
    content_type: str = "general",
    pharmaceutical_optimization: bool = True,
    **kwargs
) -> NVIDIAEmbeddings:
    """
    Convenience function to create enhanced NVIDIA embeddings with pharmaceutical optimization.

    Args:
        content_type: Type of pharmaceutical content being processed
        pharmaceutical_optimization: Enable pharmaceutical domain optimizations
        **kwargs: Additional parameters passed to NVIDIAEmbeddings

    Returns:
        Configured NVIDIAEmbeddings instance
    """

    # Set up pharmaceutical-optimized defaults
    enhanced_kwargs = {
        'enable_nemo_service': True,
        'pharmaceutical_optimization': pharmaceutical_optimization,
        'content_type': content_type,
        'enable_performance_monitoring': True,
        **kwargs
    }

    # Select optimal NeMo model based on content type
    if pharmaceutical_optimization and 'nemo_model_preference' not in kwargs:
        model_recommendations = {
            "clinical_trial": "nv-embedqa-e5-v5",
            "drug_label": "nv-embedqa-e5-v5",
            "patent": "nv-embedqa-mistral7b-v2",
            "research_paper": "nv-embedqa-mistral7b-v2",
            "regulatory_document": "nv-embedqa-e5-v5",
            "multilingual_content": "nv-embedqa-mistral7b-v2"
        }

        enhanced_kwargs['nemo_model_preference'] = model_recommendations.get(
            content_type,
            "nv-embedqa-e5-v5"
        )

    return NVIDIAEmbeddings(**enhanced_kwargs)


# Maintain backward compatibility by providing the original class name
# This allows existing code to continue working without changes
LegacyEmbeddings = LegacyNVIDIAEmbeddings


def main():
    """Test the enhanced NVIDIA embeddings V2."""
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing Enhanced NVIDIA Embeddings V2...")

    # Test backward compatibility (should work exactly like original)
    print("\n1. Testing backward compatibility...")
    embeddings_compat = NVIDIAEmbeddings()

    if embeddings_compat.test_connection():
        print("✅ Backward compatibility test successful!")

        # Test basic embedding
        test_query = "What are the side effects of aspirin?"
        query_embedding = embeddings_compat.embed_query(test_query)
        print(f"✅ Query embedding successful! Dimension: {len(query_embedding)}")

    else:
        print("❌ Backward compatibility test failed!")
        return

    # Test enhanced pharmaceutical features
    print("\n2. Testing pharmaceutical optimization...")
    pharma_embeddings = create_enhanced_embeddings(
        content_type="drug_label",
        pharmaceutical_optimization=True
    )

    if pharma_embeddings.test_connection():
        print("✅ Pharmaceutical optimization test successful!")

        # Test pharmaceutical content
        pharma_texts = [
            "CONTRAINDICATIONS: This medication is contraindicated in patients with severe hepatic impairment.",
            "DOSAGE AND ADMINISTRATION: The recommended dose is 10mg once daily with or without food.",
            "ADVERSE REACTIONS: The most common adverse reactions (≥5%) are headache, nausea, and dizziness."
        ]

        doc_embeddings = pharma_embeddings.embed_documents(pharma_texts)
        print(f"✅ Pharmaceutical document embeddings successful! Shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")

        # Display service information
        service_info = pharma_embeddings.get_service_info()
        print(f"✅ Service Info: {service_info['active_service']} service with pharmaceutical optimization")

        # Display performance metrics
        metrics = pharma_embeddings.get_performance_metrics()
        if metrics.get('monitoring_enabled'):
            print(f"✅ Performance Metrics: {metrics['total_requests']} requests, {metrics['avg_response_time_ms']:.2f}ms avg response time")

    else:
        print("❌ Pharmaceutical optimization test failed!")


if __name__ == "__main__":
    main()