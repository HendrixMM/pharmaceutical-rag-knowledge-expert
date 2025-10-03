"""
Enhanced NVIDIA NeMo Embedding Service

Advanced embedding service that leverages the latest NVIDIA NeMo Retriever embedding models
with pharmaceutical domain optimization and backward compatibility.

Key Features:
- Multi-model support (NV-EmbedQA-E5-v5, NV-EmbedQA-Mistral7B-v2, Snowflake-Arctic-Embed-L)
- Pharmaceutical domain-specific optimizations
- Intelligent model selection based on content type
- Batch processing with automatic optimization
- Caching and performance monitoring
- Fallback to existing nvidia_embeddings.py for compatibility

Based on latest NVIDIA NeMo Retriever documentation patterns and best practices.
"""
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .nemo_retriever_client import create_nemo_client
from .nemo_retriever_client import NeMoRetrieverClient
from .nvidia_embeddings import NVIDIAEmbeddings as LegacyNVIDIAEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""

    success: bool
    embeddings: Optional[List[List[float]]] = None
    model_used: Optional[str] = None
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""

    model: str = "nv-embedqa-e5-v5"
    batch_size: int = 100
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_text_length: int = 32768
    truncate_strategy: str = "end"  # "start", "end", "middle"
    normalize_embeddings: bool = True
    pharmaceutical_optimization: bool = True


@dataclass
class PharmaceuticalContext:
    """Context information for pharmaceutical content optimization."""

    content_type: str = "general"  # "clinical_trial", "drug_label", "patent", "research_paper"
    domain_terms: List[str] = field(default_factory=list)
    requires_precision: bool = False  # For regulatory/safety-critical content
    language: str = "en"


class NeMoEmbeddingService(Embeddings):
    """
    Enhanced embedding service using NVIDIA NeMo Retriever models.

    Provides intelligent model selection, pharmaceutical optimizations,
    and maintains backward compatibility with existing systems.
    """

    # Model recommendations for different pharmaceutical use cases
    PHARMACEUTICAL_MODEL_RECOMMENDATIONS = {
        "clinical_trial": {
            "primary": "nv-embedqa-e5-v5",
            "reason": "Optimized for question-answering retrieval in medical context",
        },
        "drug_label": {
            "primary": "nv-embedqa-e5-v5",
            "reason": "High precision for safety-critical pharmaceutical information",
        },
        "patent": {
            "primary": "nv-embedqa-mistral7b-v2",
            "reason": "Better handling of complex technical and legal language",
        },
        "research_paper": {
            "primary": "nv-embedqa-mistral7b-v2",
            "reason": "Superior performance on long-form scientific content",
        },
        "regulatory_document": {
            "primary": "nv-embedqa-e5-v5",
            "reason": "Consistent and reliable embeddings for compliance",
        },
        "multilingual_content": {"primary": "nv-embedqa-mistral7b-v2", "reason": "Native multilingual support"},
        "general": {"primary": "nv-embedqa-e5-v5", "reason": "Balanced performance for mixed pharmaceutical content"},
    }

    # Pharmaceutical terminology that benefits from specialized handling
    PHARMA_TERMINOLOGY = {
        "drug_classes": [
            "antibiotics",
            "antivirals",
            "antifungals",
            "analgesics",
            "anti-inflammatory",
            "anticoagulants",
            "antihistamines",
            "antidepressants",
            "antipsychotics",
            "beta-blockers",
            "ace inhibitors",
            "diuretics",
            "statins",
            "biologics",
        ],
        "medical_terms": [
            "pharmacokinetics",
            "pharmacodynamics",
            "bioavailability",
            "metabolism",
            "clearance",
            "half-life",
            "bioequivalence",
            "contraindication",
            "indication",
            "adverse event",
            "side effect",
            "drug interaction",
            "dosage",
            "titration",
        ],
        "regulatory_terms": [
            "clinical trial",
            "phase I",
            "phase II",
            "phase III",
            "IRB",
            "IND",
            "NDA",
            "BLA",
            "ANDA",
            "FDA",
            "EMA",
            "ICH",
            "GCP",
            "GLP",
            "cGMP",
            "validation",
        ],
    }

    def __init__(
        self,
        nemo_client: Optional[NeMoRetrieverClient] = None,
        config: Optional[EmbeddingConfig] = None,
        pharmaceutical_context: Optional[PharmaceuticalContext] = None,
        enable_legacy_fallback: bool = True,
        legacy_embeddings_config: Optional[Dict[str, Any]] = None,
        enable_multi_model_strategy: bool = True,
        enable_performance_monitoring: bool = True,
    ):
        """
        Initialize NeMo Embedding Service.

        Args:
            nemo_client: Pre-configured NeMo client (creates new if None)
            config: Embedding configuration
            pharmaceutical_context: Pharmaceutical context for optimization
            enable_legacy_fallback: Enable fallback to existing nvidia_embeddings
            legacy_embeddings_config: Configuration for legacy embeddings
            enable_multi_model_strategy: Enable advanced multi-model selection
            enable_performance_monitoring: Enable performance monitoring and alerting
        """
        self.nemo_client = nemo_client
        self.config = config or EmbeddingConfig()
        self.pharmaceutical_context = pharmaceutical_context or PharmaceuticalContext()
        self.enable_legacy_fallback = enable_legacy_fallback
        self.enable_multi_model_strategy = enable_multi_model_strategy

        # Multi-model strategy for intelligent model selection
        self.multi_model_strategy = None
        if enable_multi_model_strategy:
            self.multi_model_strategy = MultiModelEmbeddingStrategy()
            logger.info("Multi-model embedding strategy enabled")

        # Pharmaceutical embedding optimizer for domain-specific enhancements
        self.pharmaceutical_optimizer = None
        if self.config.pharmaceutical_optimization:
            self.pharmaceutical_optimizer = PharmaceuticalEmbeddingOptimizer()
            logger.info("Pharmaceutical embedding optimizer enabled")

        # Performance monitoring and fallback management
        self.service_name = f"nemo_embedding_service_{id(self)}"
        self.enable_performance_monitoring = enable_performance_monitoring
        if enable_performance_monitoring:
            # Register this service with the global performance monitor
            performance_monitor.register_service(
                service_name=self.service_name, health_check_function=self._health_check
            )
            logger.info(f"Performance monitoring enabled for {self.service_name}")

        # Legacy embeddings for fallback
        self.legacy_embeddings = None
        if enable_legacy_fallback:
            try:
                legacy_config = legacy_embeddings_config or {}
                self.legacy_embeddings = LegacyNVIDIAEmbeddings(**legacy_config)
                logger.info("Legacy NVIDIA embeddings initialized for fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize legacy embeddings: {e}")

        # Current model tracking
        self.current_model = self.config.model
        self.last_model_selection: Optional[ModelSelectionResult] = None

        # Caching
        self.embedding_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}

        # Performance metrics
        self.metrics = {
            "total_embeddings": 0,
            "total_processing_time_ms": 0.0,
            "models_used": {},
            "pharmaceutical_optimizations": 0,
            "multi_model_selections": 0,
            "cache_performance": self.cache_stats,
        }

    async def _ensure_nemo_client(self) -> NeMoRetrieverClient:
        """Ensure NeMo client is available."""
        if not self.nemo_client:
            self.nemo_client = await create_nemo_client()
        return self.nemo_client

    def _analyze_pharmaceutical_context(self, texts: List[str]) -> PharmaceuticalContext:
        """
        Analyze texts to determine pharmaceutical context and optimization needs.

        Args:
            texts: List of texts to analyze

        Returns:
            PharmaceuticalContext with optimization recommendations
        """
        combined_text = " ".join(texts).lower()

        # Determine content type
        content_type = "general"

        if any(term in combined_text for term in ["clinical trial", "phase i", "phase ii", "phase iii"]):
            content_type = "clinical_trial"
        elif any(term in combined_text for term in ["drug label", "prescribing information", "contraindication"]):
            content_type = "drug_label"
        elif any(term in combined_text for term in ["patent", "invention", "claims", "prior art"]):
            content_type = "patent"
        elif any(term in combined_text for term in ["research", "study", "journal", "publication"]):
            content_type = "research_paper"
        elif any(term in combined_text for term in ["fda", "regulatory", "submission", "approval"]):
            content_type = "regulatory_document"

        # Detect domain terms
        domain_terms = []
        for category, terms in self.PHARMA_TERMINOLOGY.items():
            found_terms = [term for term in terms if term in combined_text]
            domain_terms.extend(found_terms)

        # Determine if high precision is required
        requires_precision = any(
            term in combined_text
            for term in [
                "safety",
                "contraindication",
                "black box",
                "warning",
                "adverse",
                "dosage",
                "regulatory",
                "compliance",
                "validation",
            ]
        )

        # Basic language detection (simplified)
        language = "en"  # Default to English
        if any(term in combined_text for term in ["español", "français", "deutsch", "中文"]):
            language = "multilingual"

        return PharmaceuticalContext(
            content_type=content_type,
            domain_terms=domain_terms,
            requires_precision=requires_precision,
            language=language,
        )

    def _select_optimal_model(self, texts: List[str], pharma_context: Optional[PharmaceuticalContext] = None) -> str:
        """
        Select the optimal embedding model based on content analysis and pharmaceutical context.

        Args:
            texts: Texts to be embedded for analysis
            pharma_context: Optional pharmaceutical context (will be analyzed if not provided)

        Returns:
            Model name to use for embedding
        """
        # Use advanced multi-model strategy if enabled
        if self.enable_multi_model_strategy and self.multi_model_strategy:
            try:
                selection_result = select_optimal_embedding_model(
                    texts=texts,
                    content_type=pharma_context.content_type if pharma_context else None,
                    pharmaceutical_optimization=self.config.pharmaceutical_optimization,
                )

                self.last_model_selection = selection_result
                self.current_model = selection_result.selected_model
                self.metrics["multi_model_selections"] += 1

                logger.info(f"Multi-model strategy selected: {selection_result.selected_model}")
                logger.debug(f"Selection reason: {selection_result.selection_reason}")
                logger.debug(f"Confidence: {selection_result.confidence_score:.3f}")

                return selection_result.selected_model

            except Exception as e:
                logger.warning(f"Multi-model strategy failed, falling back to simple selection: {e}")

        # Fallback to simple pharmaceutical-specific selection
        if pharma_context:
            # Handle multilingual content
            if pharma_context.language != "en":
                return "nv-embedqa-mistral7b-v2"

            # Use pharmaceutical-specific recommendations
            content_type = pharma_context.content_type
            if content_type in self.PHARMACEUTICAL_MODEL_RECOMMENDATIONS:
                recommendation = self.PHARMACEUTICAL_MODEL_RECOMMENDATIONS[content_type]
                logger.debug(f"Selected {recommendation['primary']} for {content_type}: {recommendation['reason']}")
                return recommendation["primary"]

        # Final fallback to configured default
        return self.config.model

    def _create_cache_key(self, texts: List[str], model: str) -> str:
        """Create cache key for embedding results."""
        combined_text = "".join(sorted(texts))
        content_hash = hashlib.md5(f"{combined_text}:{model}".encode()).hexdigest()
        return f"embed:{model}:{content_hash}"

    def _get_cached_embeddings(self, cache_key: str) -> Optional[List[List[float]]]:
        """Retrieve embeddings from cache if available and not expired."""
        if not self.config.enable_caching:
            return None

        if cache_key in self.embedding_cache:
            cached_data = self.embedding_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.config.cache_ttl_seconds:
                self.cache_stats["hits"] += 1
                return cached_data["embeddings"]
            else:
                # Remove expired entry
                del self.embedding_cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    def _cache_embeddings(self, cache_key: str, embeddings: List[List[float]]) -> None:
        """Cache embeddings for future use."""
        if self.config.enable_caching:
            self.embedding_cache[cache_key] = {"embeddings": embeddings, "timestamp": time.time()}

    def _preprocess_texts(self, texts: List[str], pharma_context: PharmaceuticalContext) -> List[str]:
        """
        Preprocess texts for optimal embedding generation with pharmaceutical optimization.

        Args:
            texts: Input texts
            pharma_context: Pharmaceutical context

        Returns:
            Preprocessed texts optimized for pharmaceutical domain
        """
        processed_texts = []

        # Apply pharmaceutical optimization if enabled
        if self.config.pharmaceutical_optimization and self.pharmaceutical_optimizer:
            try:
                # Convert PharmaceuticalContext content_type to PharmaceuticalContentType enum
                content_type = None
                if pharma_context.content_type:
                    try:
                        content_type = PharmaceuticalContentType(pharma_context.content_type)
                    except ValueError:
                        logger.debug(f"Unknown content type for optimization: {pharma_context.content_type}")

                # Apply pharmaceutical optimization
                optimization_results = self.pharmaceutical_optimizer.optimize_pharmaceutical_content(
                    texts=texts,
                    content_type=content_type,
                    enable_drug_normalization=True,
                    enable_terminology_standardization=True,
                    enable_safety_detection=True,
                    enable_regulatory_enhancement=True,
                )

                # Extract optimized texts and update metrics
                optimized_texts = [result.optimized_text for result in optimization_results]

                # Log optimization summary
                total_optimizations = sum(len(result.optimizations_applied) for result in optimization_results)
                if total_optimizations > 0:
                    logger.info(f"Applied {total_optimizations} pharmaceutical optimizations across {len(texts)} texts")
                    self.metrics["pharmaceutical_optimizations"] += total_optimizations

                texts = optimized_texts

            except Exception as e:
                logger.warning(f"Pharmaceutical optimization failed, using original texts: {e}")

        # Standard preprocessing
        for text in texts:
            processed_text = text

            # Basic pharmaceutical-specific preprocessing (fallback)
            if self.config.pharmaceutical_optimization and pharma_context.domain_terms:
                # Ensure pharmaceutical terms are properly formatted
                for term in pharma_context.domain_terms:
                    # Standardize term formatting
                    processed_text = processed_text.replace(term.upper(), term.title())

            # Handle text length limits
            if len(processed_text) > self.config.max_text_length:
                if self.config.truncate_strategy == "end":
                    processed_text = processed_text[: self.config.max_text_length]
                elif self.config.truncate_strategy == "start":
                    processed_text = processed_text[-self.config.max_text_length :]
                elif self.config.truncate_strategy == "middle":
                    mid_point = len(processed_text) // 2
                    half_max = self.config.max_text_length // 2
                    start_part = processed_text[:half_max]
                    end_part = processed_text[mid_point - half_max :]
                    processed_text = start_part + "..." + end_part

            processed_texts.append(processed_text)

        return processed_texts

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using optimal NeMo model selection.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        start_time = time.time()

        try:
            # Analyze pharmaceutical context
            pharma_context = None
            if self.config.pharmaceutical_optimization:
                pharma_context = self._analyze_pharmaceutical_context(texts)
                self.metrics["pharmaceutical_optimizations"] += 1

            # Select optimal model using enhanced strategy
            model = self._select_optimal_model(texts, pharma_context)

            # Preprocess texts
            processed_texts = self._preprocess_texts(texts, pharma_context) if pharma_context else texts

            # Check cache
            cache_key = self._create_cache_key(processed_texts, model)
            cached_embeddings = self._get_cached_embeddings(cache_key)

            if cached_embeddings:
                processing_time = (time.time() - start_time) * 1000
                self._update_metrics(len(texts), processing_time, model, cache_hit=True)
                return cached_embeddings

            # Generate embeddings using NeMo
            result = await self._generate_embeddings_with_nemo(processed_texts, model)

            if result.success:
                # Cache results
                self._cache_embeddings(cache_key, result.embeddings)

                processing_time = (time.time() - start_time) * 1000
                self._update_metrics(len(texts), processing_time, model, cache_hit=False)

                return result.embeddings
            else:
                # Fallback to legacy implementation
                if self.enable_legacy_fallback and self.legacy_embeddings:
                    logger.warning(f"NeMo embedding failed, falling back to legacy: {result.error}")
                    legacy_embeddings = self.legacy_embeddings.embed_documents(texts)

                    processing_time = (time.time() - start_time) * 1000
                    self._update_metrics(len(texts), processing_time, "legacy", cache_hit=False)

                    return legacy_embeddings
                else:
                    raise RuntimeError(f"Embedding failed and no fallback available: {result.error}")

        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            # Last resort fallback
            if self.enable_legacy_fallback and self.legacy_embeddings:
                return self.legacy_embeddings.embed_documents(texts)
            else:
                raise

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    async def _generate_embeddings_with_nemo(self, texts: List[str], model: str) -> EmbeddingResult:
        """Generate embeddings using NeMo Retriever client."""
        try:
            client = await self._ensure_nemo_client()

            # Use NeMo client to generate embeddings
            response = await client.embed_texts(
                texts=texts, model=model, use_langchain=True  # Prefer LangChain integration when available
            )

            if response.success:
                embeddings = response.data["embeddings"]

                # Normalize embeddings if requested
                if self.config.normalize_embeddings:
                    embeddings = [self._normalize_embedding(emb) for emb in embeddings]

                return EmbeddingResult(
                    success=True, embeddings=embeddings, model_used=model, processing_time_ms=response.response_time_ms
                )
            else:
                return EmbeddingResult(success=False, error=response.error, model_used=model)

        except Exception as e:
            return EmbeddingResult(success=False, error=str(e), model_used=model)

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            return (embedding_array / norm).tolist()
        return embedding

    def _update_metrics(self, num_texts: int, processing_time_ms: float, model: str, cache_hit: bool):
        """Update performance metrics."""
        self.metrics["total_embeddings"] += num_texts
        self.metrics["total_processing_time_ms"] += processing_time_ms

        if model not in self.metrics["models_used"]:
            self.metrics["models_used"][model] = 0
        self.metrics["models_used"][model] += num_texts

        self.cache_stats["total_requests"] += 1

    # Synchronous wrapper methods for compatibility
    def embed_documents_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_documents."""
        return asyncio.run(self.embed_documents(texts))

    def embed_query_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embed_query."""
        return asyncio.run(self.embed_query(text))

    # Additional pharmaceutical-specific methods
    async def embed_pharmaceutical_documents(
        self, documents: List[Document], preserve_metadata: bool = True
    ) -> List[Document]:
        """
        Embed pharmaceutical documents with enhanced metadata preservation.

        Args:
            documents: List of documents to embed
            preserve_metadata: Whether to preserve document metadata

        Returns:
            Documents with embeddings added to metadata
        """
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embed_documents(texts)

        enhanced_docs = []
        for doc, embedding in zip(documents, embeddings):
            new_metadata = doc.metadata.copy() if preserve_metadata else {}
            new_metadata.update(
                {
                    "embedding": embedding,
                    "embedding_model": self.config.model,
                    "embedding_service": "nemo_enhanced",
                    "embedding_timestamp": time.time(),
                }
            )

            enhanced_doc = Document(page_content=doc.page_content, metadata=new_metadata)
            enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def _health_check(self) -> ServiceHealth:
        """
        Perform health check for this embedding service.

        Returns:
            Current health status
        """
        try:
            # Basic connectivity check
            if self.nemo_client is None:
                return ServiceHealth.DOWN

            # Check if we can get a simple embedding (this is a simplified check)
            # In production, you might want a more sophisticated health check
            if self.legacy_embeddings:
                # Test legacy service as fallback indicator
                try:
                    test_result = self.legacy_embeddings.test_connection()
                    if test_result:
                        return ServiceHealth.HEALTHY
                    else:
                        return ServiceHealth.DEGRADED
                except Exception:
                    return ServiceHealth.UNHEALTHY

            return ServiceHealth.HEALTHY

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return ServiceHealth.UNHEALTHY

    def _update_metrics(
        self, text_count: int, processing_time_ms: float, model_used: str, cache_hit: bool = False, success: bool = True
    ):
        """
        Update embedding performance metrics.

        Args:
            text_count: Number of texts processed
            processing_time_ms: Processing time in milliseconds
            model_used: Name of the model that was used
            cache_hit: Whether this was a cache hit
            success: Whether the embedding was successful
        """
        # Update local metrics
        self.metrics["total_embeddings"] += text_count
        self.metrics["total_processing_time_ms"] += processing_time_ms

        if model_used not in self.metrics["models_used"]:
            self.metrics["models_used"][model_used] = 0
        self.metrics["models_used"][model_used] += text_count

        # Update cache stats
        self.cache_stats["total_requests"] += 1
        if cache_hit:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1

        # Update multi-model strategy performance if enabled
        if not cache_hit:  # Only update performance for actual model calls
            self.update_model_performance(model_name=model_used, response_time_ms=processing_time_ms, success=success)

        # Update global performance monitor if enabled
        if self.enable_performance_monitoring:
            monitor_embedding_request(
                service_name=self.service_name,
                response_time_ms=processing_time_ms,
                success=success,
                is_pharmaceutical=self.config.pharmaceutical_optimization,
                cache_hit=cache_hit,
            )

    def get_pharmaceutical_model_recommendation(self, content_type: str) -> Dict[str, str]:
        """Get model recommendation for specific pharmaceutical content type."""
        return self.PHARMACEUTICAL_MODEL_RECOMMENDATIONS.get(
            content_type, self.PHARMACEUTICAL_MODEL_RECOMMENDATIONS["general"]
        )

    def update_model_performance(
        self, model_name: str, response_time_ms: float, success: bool, accuracy_score: Optional[float] = None
    ):
        """
        Update performance metrics for multi-model strategy.

        Args:
            model_name: Name of the model that was used
            response_time_ms: Response time in milliseconds
            success: Whether the embedding was successful
            accuracy_score: Optional accuracy score
        """
        if self.enable_multi_model_strategy and self.multi_model_strategy:
            try:
                self.multi_model_strategy.update_model_performance(
                    model_name=model_name,
                    response_time_ms=response_time_ms,
                    success=success,
                    accuracy_score=accuracy_score,
                    pharmaceutical_context=self.config.pharmaceutical_optimization,
                )
            except Exception as e:
                logger.warning(f"Failed to update multi-model performance: {e}")

    def get_model_selection_info(self) -> Dict[str, Any]:
        """
        Get information about the last model selection.

        Returns:
            Dictionary with model selection details
        """
        info = {
            "current_model": self.current_model,
            "multi_model_strategy_enabled": self.enable_multi_model_strategy,
            "last_selection": None,
        }

        if self.last_model_selection:
            info["last_selection"] = {
                "selected_model": self.last_model_selection.selected_model,
                "confidence_score": self.last_model_selection.confidence_score,
                "selection_reason": self.last_model_selection.selection_reason,
                "fallback_models": self.last_model_selection.fallback_models,
                "pharmaceutical_optimization_applied": self.last_model_selection.pharmaceutical_optimization_applied,
            }

        return info

    def get_multi_model_performance_report(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive performance report from multi-model strategy.

        Returns:
            Performance report or None if multi-model strategy is disabled
        """
        if self.enable_multi_model_strategy and self.multi_model_strategy:
            try:
                return self.multi_model_strategy.get_model_performance_report()
            except Exception as e:
                logger.warning(f"Failed to get multi-model performance report: {e}")

        return None

    def get_pharmaceutical_optimization_report(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive pharmaceutical optimization report.

        Returns:
            Optimization report or None if pharmaceutical optimization is disabled
        """
        if self.config.pharmaceutical_optimization and self.pharmaceutical_optimizer:
            try:
                return self.pharmaceutical_optimizer.get_optimization_statistics()
            except Exception as e:
                logger.warning(f"Failed to get pharmaceutical optimization report: {e}")

        return None

    def add_custom_pharmaceutical_term(self, drug_name: str, synonyms: List[str], category: str = "custom"):
        """
        Add custom pharmaceutical term mapping for specialized content.

        Args:
            drug_name: Canonical drug name
            synonyms: List of synonyms and alternative names
            category: Drug category
        """
        if self.pharmaceutical_optimizer:
            self.pharmaceutical_optimizer.add_custom_drug_mapping(
                canonical_name=drug_name, synonyms=synonyms, category=category
            )
            logger.info(f"Added custom pharmaceutical term: {drug_name}")

    def get_performance_monitoring_report(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive performance monitoring report.

        Returns:
            Performance monitoring report or None if monitoring is disabled
        """
        if self.enable_performance_monitoring:
            try:
                # Get service-specific report from global monitor
                full_report = performance_monitor.get_performance_report()

                # Extract this service's data
                service_data = full_report.get("services", {}).get(self.service_name, {})

                return {
                    "service_name": self.service_name,
                    "service_health": performance_monitor.get_service_health(self.service_name).value,
                    "performance_data": service_data,
                    "optimization_recommendations": performance_monitor.get_optimization_recommendations(),
                    "active_alerts": [
                        alert for alert in performance_monitor.active_alerts if alert.service_name == self.service_name
                    ],
                }
            except Exception as e:
                logger.warning(f"Failed to get performance monitoring report: {e}")

        return None

    def start_performance_monitoring(self):
        """Start background performance monitoring."""
        if self.enable_performance_monitoring:
            performance_monitor.start_monitoring()
            logger.info(f"Started performance monitoring for {self.service_name}")

    def stop_performance_monitoring(self):
        """Stop background performance monitoring."""
        if self.enable_performance_monitoring:
            performance_monitor.stop_monitoring()
            logger.info(f"Stopped performance monitoring for {self.service_name}")

    def get_service_health_status(self) -> str:
        """
        Get current health status of this service.

        Returns:
            Health status string
        """
        if self.enable_performance_monitoring:
            return performance_monitor.get_service_health(self.service_name).value
        else:
            # Fallback health check
            return self._health_check().value

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self.metrics.copy()

        if metrics["total_embeddings"] > 0:
            metrics["avg_processing_time_ms"] = metrics["total_processing_time_ms"] / metrics["total_embeddings"]
        else:
            metrics["avg_processing_time_ms"] = 0.0

        # Cache performance
        total_requests = self.cache_stats["total_requests"]
        if total_requests > 0:
            metrics["cache_hit_rate"] = self.cache_stats["hits"] / total_requests
        else:
            metrics["cache_hit_rate"] = 0.0

        return metrics

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}
        logger.info("Embedding cache cleared")


# Factory functions for easy integration
async def create_nemo_embedding_service(
    config: Optional[EmbeddingConfig] = None, enable_pharmaceutical_optimization: bool = True, **kwargs
) -> NeMoEmbeddingService:
    """
    Factory function to create NeMo Embedding Service.

    Args:
        config: Embedding configuration
        enable_pharmaceutical_optimization: Enable pharmaceutical domain optimizations
        **kwargs: Additional configuration parameters

    Returns:
        Configured NeMoEmbeddingService
    """
    if not config:
        config = EmbeddingConfig(pharmaceutical_optimization=enable_pharmaceutical_optimization, **kwargs)

    service = NeMoEmbeddingService(config=config)

    # Ensure NeMo client is ready
    await service._ensure_nemo_client()

    return service


def create_nemo_embedding_service_sync(
    config: Optional[EmbeddingConfig] = None, enable_pharmaceutical_optimization: bool = True, **kwargs
) -> NeMoEmbeddingService:
    """
    Synchronous factory function for NeMo Embedding Service.

    Args:
        config: Embedding configuration
        enable_pharmaceutical_optimization: Enable pharmaceutical domain optimizations
        **kwargs: Additional configuration parameters

    Returns:
        Configured NeMoEmbeddingService
    """
    return asyncio.run(
        create_nemo_embedding_service(
            config=config, enable_pharmaceutical_optimization=enable_pharmaceutical_optimization, **kwargs
        )
    )


# Convenience function for pharmaceutical RAG systems
async def create_pharmaceutical_embedding_service(
    content_types: List[str] = None, enable_caching: bool = True, **kwargs
) -> NeMoEmbeddingService:
    """
    Create embedding service optimized for pharmaceutical content.

    Args:
        content_types: List of content types to optimize for
        enable_caching: Enable embedding caching
        **kwargs: Additional configuration

    Returns:
        Pharmaceutical-optimized NeMoEmbeddingService
    """
    config = EmbeddingConfig(pharmaceutical_optimization=True, enable_caching=enable_caching, **kwargs)

    service = await create_nemo_embedding_service(config=config)

    logger.info(f"Created pharmaceutical embedding service optimized for: {content_types or 'all types'}")

    return service


# Example usage
if __name__ == "__main__":

    async def test_embedding_service():
        """Test NeMo embedding service functionality."""
        service = await create_pharmaceutical_embedding_service()

        # Test with pharmaceutical content
        test_texts = [
            "Aspirin is an NSAID used for pain relief and inflammation reduction.",
            "Phase III clinical trial showed 85% efficacy in reducing cardiovascular events.",
            "Contraindications include active bleeding and severe hepatic impairment.",
        ]

        try:
            embeddings = await service.embed_documents(test_texts)
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding dimensions: {len(embeddings[0])}")

            metrics = service.get_metrics()
            print(f"Performance metrics: {metrics}")

        except Exception as e:
            print(f"Test failed: {e}")

    asyncio.run(test_embedding_service())
