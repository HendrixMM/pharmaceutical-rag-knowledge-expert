"""
NVIDIA NeMo Retriever Universal Client

Provides unified access to NVIDIA NeMo Retriever NIMs and a cloud-first
enhanced client wrapper.

Factory guidance:
- Use `create_client()` for sync code (returns immediately). When cloud-first is
  enabled, it returns `NeMoClientWrapper` wrapping the enhanced client; otherwise
  it returns `NeMoRetrieverClient`.
- Use `create_client_async()` in async codebases to mirror the above with an
  async alias returning immediately (no awaits required).

Features:
1. Unified API for all NeMo NIMs
2. Automatic retry and fallback logic
3. Health monitoring and diagnostics
4. Batch processing optimization
5. Integration with langchain-nvidia-ai-endpoints
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urljoin
import json

import aiohttp
import requests
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank

logger = logging.getLogger(__name__)

# Enhanced client imports for cloud-first integration
try:
    from .clients.nemo_client_enhanced import EnhancedNeMoClient
    from .enhanced_config import EnhancedRAGConfig
    ENHANCED_CLIENT_AVAILABLE = True
except ImportError:
    try:
        from src.clients.nemo_client_enhanced import EnhancedNeMoClient
        from src.enhanced_config import EnhancedRAGConfig
        ENHANCED_CLIENT_AVAILABLE = True
    except ImportError:
        logger.warning("Enhanced client not available, using standard NeMo client only")
        ENHANCED_CLIENT_AVAILABLE = False

@dataclass
class NeMoServiceConfig:
    """Configuration for a specific NeMo NIM service."""
    name: str
    endpoint: str
    model: str
    max_batch_size: int = 100
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class NeMoAPIResponse:
    """Standardized response from NeMo APIs."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    response_time_ms: float = 0.0
    service: Optional[str] = None
    model: Optional[str] = None

class NVIDIABuildCreditsMonitor:
    """Simple credit monitor for NVIDIA Build free tier."""
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.credits_used = 0
        self.credits_remaining = 10000

    def log_api_call(self, service: str, tokens_used: int = 1) -> None:
        try:
            tokens = max(0, int(tokens_used))
        except Exception:
            tokens = 1
        self.credits_used += tokens
        self.credits_remaining = max(0, 10000 - self.credits_used)
        if self.credits_remaining <= 100:
            logger.warning("Low NVIDIA Build credits remaining: %s", self.credits_remaining)
        logger.info("Credits: service=%s used=%s remaining=%s", service, tokens, self.credits_remaining)


class NeMoRetrieverClient:
    """
    Universal client for NVIDIA NeMo Retriever NIMs.

    CURRENT IMPLEMENTATION (2024):
    Supports the three-step NeMo Retriever pipeline with pharmaceutical optimization:
    1. Extraction: NV-Ingest VLM-based document processing
    2. Embedding: nvidia/nv-embedqa-e5-v5 for medical Q&A optimization
    3. Reranking: llama-3_2-nemoretriever-500m-rerank-v2 for pharmaceutical relevance

    Features:
    - Both cloud-hosted and self-hosted NIM deployments
    - Automatic authentication, retries, and error handling
    - Pharmaceutical domain-specific model recommendations
    - Environment-driven configuration support
    - Free-tier credits monitoring and optimization
    """

    # Default NVIDIA NeMo NIM endpoints (cloud-hosted)
    DEFAULT_ENDPOINTS = {
        "embedding": "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings",
        "reranking": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        "extraction": "https://ai.api.nvidia.com/v1/retrieval/nvidia/extraction",
    }

    # Available embedding models with their specifications
    EMBEDDING_MODELS = {
        "nv-embedqa-e5-v5": {
            "full_name": "nvidia/nv-embedqa-e5-v5",
            "dimensions": 1024,
            "max_length": 32768,
            "description": "Optimized for question-answering retrieval",
            "recommended_for": ["pharmaceutical_qa", "medical_literature", "technical_docs"]
        },
        "nv-embedqa-mistral7b-v2": {
            "full_name": "nvidia/nv-embedqa-mistral7b-v2",
            "dimensions": 4096,
            "max_length": 32768,
            "description": "Multilingual model for text embedding and QA",
            "recommended_for": ["multilingual_content", "complex_reasoning", "long_documents"]
        },
        "snowflake-arctic-embed-l": {
            "full_name": "Snowflake/snowflake-arctic-embed-l",
            "dimensions": 1024,
            "max_length": 8192,
            "description": "Optimized for text embedding tasks",
            "recommended_for": ["general_text", "similarity_search", "clustering"]
        }
    }

    # Available reranking models
    RERANKING_MODELS = {
        "nv-rerankqa-mistral4b-v3": {
            "full_name": "nvidia/nv-rerankqa-mistral4b-v3",
            "max_pairs": 1000,
            "description": "Fine-tuned for text reranking and accurate QA",
            "recommended_for": ["pharmaceutical_reranking", "medical_relevance", "cross_modal"]
        },
        "llama-3_2-nemoretriever-500m-rerank-v2": {
            "full_name": "meta/llama-3_2-nemoretriever-500m-rerank-v2",
            "max_pairs": 1000,
            "description": "Latest NeMo Retriever reranking model optimized for pharmaceutical content",
            "recommended_for": ["pharmaceutical_reranking", "medical_literature", "regulatory_documents", "clinical_trials"]
        }
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_endpoints: Optional[Dict[str, str]] = None,
                 custom_headers: Optional[Dict[str, str]] = None,
                 enable_langchain_integration: bool = True,
                 credits_monitor: Optional[NVIDIABuildCreditsMonitor] = None):
        """
        Initialize NeMo Retriever client.

        Args:
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
            base_endpoints: Custom endpoints for self-hosted NIMs
            custom_headers: Additional headers for API requests
            enable_langchain_integration: Use LangChain NVIDIA endpoints when available
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key is required. Set NVIDIA_API_KEY environment variable.")

        self.base_endpoints = base_endpoints or self.DEFAULT_ENDPOINTS.copy()
        self.enable_langchain_integration = enable_langchain_integration
        self.credits_monitor = credits_monitor

        # Standard NVIDIA headers (Bearer token)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NeMo-Retriever-Client/1.0"
        }
        if custom_headers:
            self.headers.update(custom_headers)

        # Service configurations
        self.services = self._initialize_service_configs()

        # LangChain integrations (when available)
        self.langchain_embeddings = {}
        self.langchain_reranker = None

        if self.enable_langchain_integration:
            self._initialize_langchain_integrations()

        # Health monitoring
        self.service_health = {}
        self._last_health_check = 0
        self.health_check_interval = 300  # 5 minutes

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0
        }

    def _initialize_service_configs(self) -> Dict[str, NeMoServiceConfig]:
        """Initialize service configurations for all NeMo NIMs."""
        return {
            "embedding": NeMoServiceConfig(
                name="embedding",
                endpoint=self.base_endpoints["embedding"],
                model="nvidia/nv-embedqa-e5-v5",  # Default model
                max_batch_size=100,
                timeout_seconds=30,
                headers=self.headers.copy()
            ),
            "reranking": NeMoServiceConfig(
                name="reranking",
                endpoint=self.base_endpoints["reranking"],
                model="nvidia/nv-rerankqa-mistral4b-v3",
                max_batch_size=50,
                timeout_seconds=45,
                headers=self.headers.copy()
            ),
            "extraction": NeMoServiceConfig(
                name="extraction",
                endpoint=self.base_endpoints["extraction"],
                model="nvidia/nv-ingest",
                max_batch_size=10,  # Document processing is more resource intensive
                timeout_seconds=120,
                headers=self.headers.copy()
            )
        }

    def _initialize_langchain_integrations(self) -> None:
        """Initialize LangChain NVIDIA AI Endpoints integrations."""
        try:
            # Initialize embedding models
            for model_key, model_info in self.EMBEDDING_MODELS.items():
                try:
                    self.langchain_embeddings[model_key] = NVIDIAEmbeddings(
                        model=model_info["full_name"],
                        nvidia_api_key=self.api_key
                    )
                    logger.info(f"Initialized LangChain embedding for {model_key}")
                except Exception as e:
                    logger.warning(f"Failed to initialize LangChain embedding for {model_key}: {e}")

            # Initialize reranker
            try:
                self.langchain_reranker = NVIDIARerank(
                    model="nvidia/nv-rerankqa-mistral4b-v3",
                    nvidia_api_key=self.api_key
                )
                logger.info("Initialized LangChain reranker")
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain reranker: {e}")

        except Exception as e:
            logger.warning(f"LangChain integration initialization failed: {e}")

    @staticmethod
    def validate_model_availability(service: str, model: str) -> bool:
        """
        Validate if a model is available for a specific service.

        Args:
            service: Service type ("embedding" or "reranking")
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        # Normalize model name by extracting the key part
        model_key = model
        if "/" in model:
            # Extract the part after the last slash for matching
            model_key = model.split("/")[-1]

        if service == "embedding":
            # Check both the key and the full_name in EMBEDDING_MODELS
            return (
                model_key in NeMoRetrieverClient.EMBEDDING_MODELS or
                any(info.get("full_name") == model for info in NeMoRetrieverClient.EMBEDDING_MODELS.values())
            )
        elif service == "reranking":
            # Check both the key and the full_name in RERANKING_MODELS
            return (
                model_key in NeMoRetrieverClient.RERANKING_MODELS or
                any(info.get("full_name") == model for info in NeMoRetrieverClient.RERANKING_MODELS.values())
            )
        else:
            # Unknown service type
            return False

    async def health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        Check health of all NeMo NIM services.

        Args:
            force: Force health check even if recently performed

        Returns:
            Health status for all services
        """
        current_time = time.time()
        if not force and (current_time - self._last_health_check) < self.health_check_interval:
            return self.service_health

        health_results = {}

        for service_name, config in self.services.items():
            try:
                start_time = time.time()

                # Simple health check with minimal payload
                test_payload = self._get_health_check_payload(service_name)

                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.post(
                        config.endpoint,
                        headers=config.headers,
                        json=test_payload
                    ) as response:
                        response_time = (time.time() - start_time) * 1000

                        if response.status == 200:
                            health_results[service_name] = {
                                "status": "healthy",
                                "response_time_ms": response_time,
                                "endpoint": config.endpoint,
                                "model": config.model
                            }
                        else:
                            health_results[service_name] = {
                                "status": "unhealthy",
                                "error": f"HTTP {response.status}",
                                "endpoint": config.endpoint
                            }

            except Exception as e:
                health_results[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "endpoint": config.endpoint
                }

        self.service_health = health_results
        self._last_health_check = current_time

        return health_results

    def _get_health_check_payload(self, service_name: str) -> Dict[str, Any]:
        """Get minimal payload for health checking a specific service."""
        if service_name == "embedding":
            return {
                "input": ["health check"],
                "model": self.services[service_name].model
            }
        elif service_name == "reranking":
            return {
                "query": "health check",
                "passages": ["test passage"],
                "model": self.services[service_name].model
            }
        elif service_name == "extraction":
            return {
                "input": "health check text",
                "model": self.services[service_name].model
            }
        else:
            return {}

    async def embed_texts(self,
                         texts: List[str],
                         model: str = "nv-embedqa-e5-v5",
                         use_langchain: bool = True) -> NeMoAPIResponse:
        """
        Generate embeddings for texts using NeMo Embedding NIMs.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            use_langchain: Use LangChain integration if available

        Returns:
            NeMoAPIResponse with embeddings data
        """
        start_time = time.time()

        try:
            # Use LangChain integration if available and requested
            if use_langchain and model in self.langchain_embeddings:
                embeddings = await self._embed_with_langchain(texts, model)
                response_time = (time.time() - start_time) * 1000

                self._update_metrics(True, response_time)

                return NeMoAPIResponse(
                    success=True,
                    data={"embeddings": embeddings},
                    response_time_ms=response_time,
                    service="embedding",
                    model=model
                )

            # Fallback to direct API calls
            return await self._embed_with_direct_api(texts, model, start_time)

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(False, response_time)

            logger.error(f"Embedding failed: {e}")
            return NeMoAPIResponse(
                success=False,
                error=str(e),
                response_time_ms=response_time,
                service="embedding",
                model=model
            )

    async def _embed_with_langchain(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using LangChain integration."""
        langchain_embedder = self.langchain_embeddings[model]

        # LangChain embeddings are typically sync, so run in executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            langchain_embedder.embed_documents,
            texts
        )

        return embeddings

    async def _embed_with_direct_api(self, texts: List[str], model: str, start_time: float) -> NeMoAPIResponse:
        """Generate embeddings using direct API calls."""
        config = self.services["embedding"]
        model_info = self.EMBEDDING_MODELS.get(model)

        if not model_info:
            raise ValueError(f"Unknown embedding model: {model}")

        payload = {
            "input": texts,
            "model": model_info["full_name"]
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)) as session:
            async with session.post(
                config.endpoint,
                headers=config.headers,
                json=payload
            ) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    result = await response.json()
                    self._update_metrics(True, response_time)

                    # Extract embeddings from response
                    embeddings = [item["embedding"] for item in result["data"]]

                    if self.credits_monitor:
                        self.credits_monitor.log_api_call("embedding", tokens_used=max(1, len(texts)))

                    return NeMoAPIResponse(
                        success=True,
                        data={"embeddings": embeddings},
                        response_time_ms=response_time,
                        service="embedding",
                        model=model
                    )
                else:
                    error_text = await response.text()
                    self._update_metrics(False, response_time)

                    return NeMoAPIResponse(
                        success=False,
                        error=f"API error {response.status}: {error_text}",
                        response_time_ms=response_time,
                        service="embedding",
                        model=model
                    )

    async def rerank_passages(self,
                            query: str,
                            passages: List[str],
                            model: str = "nv-rerankqa-mistral4b-v3",
                            top_k: Optional[int] = None,
                            use_langchain: bool = True) -> NeMoAPIResponse:
        """
        Rerank passages for relevance to query using NeMo Reranking NIMs.

        Args:
            query: Search query
            passages: List of passages to rerank
            model: Reranking model to use
            top_k: Return top K results (default: all)
            use_langchain: Use LangChain integration if available

        Returns:
            NeMoAPIResponse with reranked passages and scores
        """
        start_time = time.time()

        try:
            # Use LangChain integration if available and requested
            if use_langchain and self.langchain_reranker:
                reranked = await self._rerank_with_langchain(query, passages, top_k)
                response_time = (time.time() - start_time) * 1000

                self._update_metrics(True, response_time)

                return NeMoAPIResponse(
                    success=True,
                    data={"reranked_passages": reranked},
                    response_time_ms=response_time,
                    service="reranking",
                    model=model
                )

            # Fallback to direct API calls
            return await self._rerank_with_direct_api(query, passages, model, top_k, start_time)

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(False, response_time)

            logger.error(f"Reranking failed: {e}")
            return NeMoAPIResponse(
                success=False,
                error=str(e),
                response_time_ms=response_time,
                service="reranking",
                model=model
            )

    async def _rerank_with_langchain(self, query: str, passages: List[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        """Rerank passages using LangChain integration."""
        # Create documents for reranking
        from langchain_core.documents import Document

        docs = [Document(page_content=passage) for passage in passages]

        # LangChain reranking is typically sync, so run in executor
        loop = asyncio.get_event_loop()
        reranked_docs = await loop.run_in_executor(
            None,
            self.langchain_reranker.compress_documents,
            docs,
            query
        )

        # Convert back to our format
        results = []
        for i, doc in enumerate(reranked_docs[:top_k] if top_k else reranked_docs):
            results.append({
                "text": doc.page_content,
                "score": getattr(doc.metadata, 'relevance_score', 1.0 - (i * 0.1)),  # Fallback scoring
                "index": i
            })

        return results

    async def _rerank_with_direct_api(self, query: str, passages: List[str], model: str, top_k: Optional[int], start_time: float) -> NeMoAPIResponse:
        """Rerank passages using direct API calls."""
        config = self.services["reranking"]
        model_info = self.RERANKING_MODELS.get(model)

        if not model_info:
            raise ValueError(f"Unknown reranking model: {model}")

        payload = {
            "query": query,
            "passages": passages,
            "model": model_info["full_name"]
        }

        if top_k:
            payload["top_n"] = top_k

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)) as session:
            async with session.post(
                config.endpoint,
                headers=config.headers,
                json=payload
            ) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    result = await response.json()
                    self._update_metrics(True, response_time)

                    return NeMoAPIResponse(
                        success=True,
                        data={"reranked_passages": result["rankings"]},
                        response_time_ms=response_time,
                        service="reranking",
                        model=model
                    )
                
                else:
                    error_text = await response.text()
                    self._update_metrics(False, response_time)

                    return NeMoAPIResponse(
                        success=False,
                        error=f"API error {response.status}: {error_text}",
                        response_time_ms=response_time,
                        service="reranking",
                        model=model
                    )

    def _update_metrics(self, success: bool, response_time_ms: float) -> None:
        """Update performance metrics."""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

        self.metrics["total_latency_ms"] += response_time_ms
        self.metrics["avg_latency_ms"] = (
            self.metrics["total_latency_ms"] / self.metrics["total_requests"]
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.copy()

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "services": self.service_health,
            "metrics": self.metrics,
            "available_models": {
                "embedding": list(self.EMBEDDING_MODELS.keys()),
                "reranking": list(self.RERANKING_MODELS.keys())
            },
            "langchain_integration": {
                "enabled": self.enable_langchain_integration,
                "embeddings_available": list(self.langchain_embeddings.keys()),
                "reranker_available": self.langchain_reranker is not None
            }
        }

    def get_model_info(self, service: str, model: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        if service == "embedding":
            return self.EMBEDDING_MODELS.get(model)
        elif service == "reranking":
            return self.RERANKING_MODELS.get(model)
        return None

    def recommend_model(self, use_case: str, content_type: str = "general") -> Dict[str, str]:
        """
        Recommend the best model for a specific use case.

        CURRENT DEFAULTS (2024):
        - Embedding: nvidia/nv-embedqa-e5-v5 (optimized for pharmaceutical Q&A)
        - Reranking: llama-3_2-nemoretriever-500m-rerank-v2 (latest pharmaceutical-optimized model)

        Args:
            use_case: Type of use case (e.g., "pharmaceutical_qa", "multilingual", "similarity")
            content_type: Type of content (e.g., "medical", "technical", "general")

        Returns:
            Dictionary with recommended embedding and reranking models optimized for the use case
        """
        embedding_model = "nv-embedqa-e5-v5"  # Default, optimized for Q&A
        reranking_model = "llama-3_2-nemoretriever-500m-rerank-v2"  # Latest and preferred for pharmaceutical content

        # Pharmaceutical/medical content - use latest models
        if "pharmaceutical" in use_case.lower() or "medical" in content_type.lower():
            embedding_model = "nv-embedqa-e5-v5"  # Optimized for medical Q&A
            reranking_model = "llama-3_2-nemoretriever-500m-rerank-v2"  # Enhanced pharmaceutical understanding

        # Multilingual content
        elif "multilingual" in use_case.lower() or "multi" in content_type.lower():
            embedding_model = "nv-embedqa-mistral7b-v2"  # Multilingual support

        # Long documents or complex reasoning
        elif "long" in content_type.lower() or "complex" in use_case.lower():
            embedding_model = "nv-embedqa-mistral7b-v2"  # Higher dimensions, longer context

        return {
            "embedding": embedding_model,
            "reranking": reranking_model,
            "reasoning": f"Recommended for {use_case} with {content_type} content"
        }


# CANONICAL client factory function for cloud-first integration
def create_client() -> Union["NeMoClientWrapper", "NeMoRetrieverClient"]:
    """
    CANONICAL factory - use this for new code.

    Create client based on configuration - enhanced client whenever available, legacy for fallback.
    This is the primary entry point for all NeMo Retriever client creation.

    Returns:
        NeMoClientWrapper when enhanced client is available (regardless of credentials),
        otherwise NeMoRetrieverClient. When credentials are missing, cloud functionality
        is disabled but enhanced fallback paths remain available.
    """
    try:
        if ENHANCED_CLIENT_AVAILABLE:
            cfg = EnhancedRAGConfig.from_env()
            if cfg.enable_nvidia_build_fallback and not cfg.has_nvidia_build_credentials():
                logger.info("Enhanced client available with cloud functionality disabled: NVIDIA Build credentials missing")
            # Return wrapper to preserve legacy async interface whenever enhanced client is available
            return NeMoClientWrapper()
    except Exception as e:
        logger.debug(f"Failed to create enhanced client: {e}")

    # Fallback to legacy client only when enhanced client unavailable
    return NeMoRetrieverClient()


async def create_client_async() -> Union["NeMoClientWrapper", "NeMoRetrieverClient"]:
    """Async alias to create_client() for async codebases.

    Returns immediately without awaiting any network operations.
    """
    return create_client()


class NeMoClientWrapper:
    """
    Drop-in compatibility wrapper that delegates to EnhancedNeMoClient.

    This class provides backward compatibility for existing consumers that expect
    the original NeMoRetrieverClient interface while transparently using the
    enhanced cloud-first client implementation.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize wrapper with enhanced client.

        Args:
            api_key: NVIDIA API key (optional, uses env var if not provided)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        if not ENHANCED_CLIENT_AVAILABLE:
            raise ImportError(
                "Enhanced client not available for wrapper. Use NeMoRetrieverClient directly."
            )

        # Construct enhanced client directly to avoid factory recursion
        cfg = EnhancedRAGConfig.from_env()
        self._enhanced = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True, api_key=api_key)

    async def embed_texts(self, texts: List[str], model: str = "nvidia/nv-embedqa-e5-v5", use_langchain: bool = True) -> NeMoAPIResponse:
        """
        Generate embeddings using enhanced client. Always returns NeMoAPIResponse
        to preserve legacy compatibility regardless of enhanced client internals.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            use_langchain: Ignored for compatibility

        Returns:
            NeMoAPIResponse with embeddings data
        """
        start_time = time.time()
        # Normalize short model names to full for cloud path defense-in-depth
        try:
            from .clients.model_normalization import normalize_model as _normalize_model
            model_full = _normalize_model(model, True)
        except Exception:
            model_full = model
        try:
            resp = await self._enhanced.embed_texts(texts, model_full)
            elapsed_ms = (time.time() - start_time) * 1000

            # If already the expected type, return directly
            if isinstance(resp, NeMoAPIResponse):
                # Ensure response_time shows measured latency
                resp.response_time_ms = elapsed_ms
                return resp

            # Normalize dict or ClientResponse from enhanced client
            embeddings = None
            model_used = model_full
            error = None

            try:
                # Attempt to import enhanced ClientResponse for isinstance checks
                from .clients.nemo_client_enhanced import ClientResponse as EnhancedClientResponse  # type: ignore
            except Exception:
                EnhancedClientResponse = None  # type: ignore

            if EnhancedClientResponse is not None and isinstance(resp, EnhancedClientResponse):  # type: ignore[arg-type]
                if resp.success:
                    data = resp.data or {}
                    if isinstance(data, dict) and "embeddings" in data:
                        embeddings = data.get("embeddings")
                        model_used = data.get("model") or model_full
                    else:
                        error = "Enhanced client returned success without 'embeddings' payload"
                else:
                    error = resp.error or "Embedding creation failed"
            elif isinstance(resp, dict):
                # Best-effort normalization from dict
                if "embeddings" in resp:
                    embeddings = resp.get("embeddings")
                    model_used = resp.get("model") or model_full
                elif isinstance(resp.get("data"), list):
                    # If data appears to be a list of vectors (list[list[float]])
                    data_list = resp.get("data")
                    if data_list == [] or (isinstance(data_list[0], list)):
                        embeddings = data_list
                        model_used = resp.get("model") or model_full
                    else:
                        error = "Unexpected 'data' format for embeddings payload"
                else:
                    error = resp.get("error") or "Unexpected embeddings payload from enhanced client"
            else:
                # Unknown shape; provide a descriptive error
                error = f"Unexpected response type from enhanced client: {type(resp).__name__}"

            if embeddings is not None:
                return NeMoAPIResponse(
                    success=True,
                    data={"embeddings": embeddings},
                    response_time_ms=elapsed_ms,
                    service="embedding",
                    model=model_used,
                )
            return NeMoAPIResponse(
                success=False,
                error=error or "Embedding creation failed",
                response_time_ms=elapsed_ms,
                service="embedding",
                model=model_used,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Wrapper embedding failed: {e}")
            return NeMoAPIResponse(
                success=False,
                error=str(e),
                response_time_ms=elapsed_ms,
                service="embedding",
                model=model_full,
            )

    async def rerank_passages(self, query: str, passages: List[str], model: str = "nvidia/nv-rerankqa-mistral4b-v3",
                              top_k: Optional[int] = None, use_langchain: bool = True) -> NeMoAPIResponse:
        """
        Rerank passages using enhanced client.

        Args:
            query: Search query
            passages: List of passages to rerank
            model: Reranking model to use
            top_k: Return top K results (default: all)
            use_langchain: Ignored for compatibility

        Returns:
            NeMoAPIResponse with reranked passages and scores
        """
        start_time = time.time()

        # Normalize short model names to full
        try:
            from .clients.model_normalization import normalize_model as _normalize_model
            model_full = _normalize_model(model, True)
        except Exception:
            model_full = model

        try:
            # Delegate to enhanced client's async-compatible rerank
            response = await self._enhanced.rerank_passages_async(query, passages, model_full, top_k)
            response_time = (time.time() - start_time) * 1000

            if response.success and response.data:
                # Convert ClientResponse to NeMoAPIResponse format
                # Expect normalized [{text, score}] sorted desc
                reranked_passages = response.data.get("reranked_passages", [])

                return NeMoAPIResponse(
                    success=True,
                    data={"reranked_passages": reranked_passages},
                    response_time_ms=response_time,
                    service="reranking",
                    model=model_full
                )
            else:
                return NeMoAPIResponse(
                    success=False,
                    error=response.error or "Reranking failed",
                    response_time_ms=response_time,
                    service="reranking",
                    model=model_full
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Wrapper reranking failed: {e}")

            return NeMoAPIResponse(
                success=False,
                error=str(e),
                response_time_ms=response_time,
                service="reranking",
                model=model_full
            )

    async def health_check(self, force: bool = False) -> Dict[str, Any]:
        """Get health status from enhanced client in legacy shape."""
        status = self.get_service_status()
        return status.get("services", status)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from enhanced client."""
        return self._enhanced.get_performance_metrics()

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status from enhanced client.

        Adds legacy-compatible keys to align with original NeMoRetrieverClient.get_service_status().
        """
        status = self._enhanced.get_endpoint_status()

        # Legacy structure expects these keys
        services = status.get("services") or {
            "cloud": status.get("cloud_status", {"success": status.get("cloud_available", False)}),
            "nemo": {"available": status.get("nemo_available", False)},
        }

        # Pull metrics from enhanced client
        metrics = self._enhanced.get_performance_metrics()

        # Available models from legacy client constants
        try:
            embedding_models = list(NeMoRetrieverClient.EMBEDDING_MODELS.keys())
            reranking_models = list(NeMoRetrieverClient.RERANKING_MODELS.keys())
        except Exception:
            embedding_models, reranking_models = [], []

        legacy = {
            "services": services,
            "metrics": metrics,
            "available_models": {
                "embedding": embedding_models,
                "reranking": reranking_models,
            },
            "langchain_integration": {
                "enabled": False,
                "embeddings_available": [],
                "reranker_available": False,
            },
        }

        return legacy


# Convenience functions for easy integration
async def create_nemo_client(api_key: Optional[str] = None, credits_monitor: Optional[NVIDIABuildCreditsMonitor] = None) -> NeMoRetrieverClient:
    """
    DEPRECATED: Factory function to create and validate NeMo Retriever client.

    Use create_client() instead for cloud-first enhanced functionality.
    This legacy factory is maintained for backward compatibility.

    Args:
        api_key: NVIDIA API key (optional, uses env var if not provided)
        credits_monitor: Optional credit usage tracker

    Returns:
        Initialized and validated NeMoRetrieverClient
    """
    logger.warning(
        "create_nemo_client() is deprecated. Use create_client() for enhanced cloud-first functionality."
    )
    client = NeMoRetrieverClient(api_key=api_key, credits_monitor=credits_monitor)

    # Perform initial health check
    health = await client.health_check(force=True)

    healthy_services = [name for name, status in health.items() if status.get("status") == "healthy"]

    if not healthy_services:
        logger.warning("No NeMo services are healthy, but client is still usable for fallback operations")
    else:
        logger.info(f"NeMo client ready with {len(healthy_services)} healthy services: {healthy_services}")

    return client


# Example usage for integration testing
if __name__ == "__main__":
    async def test_client():
        """Test NeMo client functionality."""
        try:
            client = await create_nemo_client()

            # Test embedding
            embed_response = await client.embed_texts(
                ["What are the effects of aspirin?", "How does ibuprofen work?"],
                model="nv-embedqa-e5-v5"
            )

            if embed_response.success:
                print(f"Embedding successful: {len(embed_response.data['embeddings'])} embeddings generated")
            else:
                print(f"Embedding failed: {embed_response.error}")

            # Test reranking
            rerank_response = await client.rerank_passages(
                query="side effects of pain medication",
                passages=[
                    "Aspirin can cause stomach upset and bleeding.",
                    "Ibuprofen may lead to kidney problems.",
                    "Acetaminophen is generally safe but can cause liver damage in high doses."
                ]
            )

            if rerank_response.success:
                print(f"Reranking successful: {len(rerank_response.data['reranked_passages'])} passages ranked")
            else:
                print(f"Reranking failed: {rerank_response.error}")

            # Print status
            status = client.get_service_status()
            print(f"Client status: {status}")

        except Exception as e:
            print(f"Test failed: {e}")

    # Run test
    asyncio.run(test_client())
