"""Enhanced RAG agent with medical guardrails and pharmaceutical analysis."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
import warnings
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union, TypedDict

from langchain_core.documents import Document

from .rag_agent import RAGAgent, RAGResponse
from .medical_guardrails import MedicalGuardrails
from .synthesis_engine import SynthesisEngine
from .ddi_pk_processor import DDIPKProcessor
from .enhanced_config import EnhancedRAGConfig
from .enhanced_pubmed_agent import PubMedIntegrationManager

try:
    from guardrails.actions import contains_medical_disclaimer
except ImportError:
    # Fallback if guardrails not available
    def contains_medical_disclaimer(text: Optional[str]) -> bool:
        """Fallback disclaimer detection."""
        if not text:
            return False
        normalized = text.lower()

        # Check standard disclaimer phrases
        if "medical disclaimer" in normalized or "not intended as medical advice" in normalized:
            return True

        return False

logger = logging.getLogger(__name__)


class SafetyMetadata(TypedDict):
    """Safety validation metadata."""
    input_validation: Dict[str, Any]
    output_validation: Dict[str, Any]
    retrieval_validation: Dict[str, Any]
    warnings: List[str]
    blocked: bool
    mode: str


class ErrorDict(TypedDict):
    """Standardized error format."""
    type: str
    message: str
    details: Optional[Dict[str, Any]]
    stage: Optional[str]


class SourceEntry(TypedDict):
    """Individual source document entry with standardized metadata."""
    doc_id: str
    title: Optional[str]
    content: str
    metadata: Dict[str, Any]
    score: Optional[float]
    source_type: str  # "local", "pubmed", etc.


class AnalysisResults(TypedDict):
    """Analysis results from synthesis and DDI/PK processors."""
    synthesis: Optional[Dict[str, Any]]
    ddi_pk: Optional[Dict[str, Any]]
    enabled: bool
    execution_time: float


class EnhancedRAGPayload(TypedDict):
    """Enhanced response payload from safe RAG methods.

    This TypedDict describes the structure returned by ask_question_safe_sync()
    and ask_question_safe() methods, including safety metadata and analysis results.
    """
    answer: str
    sources: List[Union[Document, SourceEntry]]
    error: Optional[Union[str, ErrorDict]]
    processing_time: float
    confidence_scores: Optional[List[float]]
    disclaimer: Optional[str]
    needs_rebuild: bool
    safety: SafetyMetadata
    analysis: AnalysisResults
    metrics: Dict[str, Any]
    component_health: Dict[str, Dict[str, Any]]


class EnhancedRAGAgent:
    """Wrap the base RAG agent with medical safety validation and advanced analysis.

    Note: When force_disclaimer_in_answer=True, the disclaimer is applied to the answer text
    regardless of rails configuration. When None, uses default behavior. Consumers should
    prefer the separate 'disclaimer' field in responses to avoid duplication.
    """

    def __init__(
        self,
        docs_folder: str,
        api_key: str,
        vector_db_path: str = "./vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: Optional[str] = None,
        enable_preflight_embedding: Optional[bool] = None,
        append_disclaimer_in_answer: Optional[bool] = None,
        force_disclaimer_in_answer: Optional[bool] = None,
        *,
        guardrails_config_path: Optional[str] = None,
        enable_synthesis: bool = False,
        enable_ddi_analysis: bool = False,
        safety_mode: str = "balanced",
        default_safe: bool = False,
        config: Optional[EnhancedRAGConfig] = None,
        pubmed_query_engine: Optional[EnhancedQueryEngine] = None,
        pubmed_scraper: Optional[EnhancedPubMedScraper] = None,
    ) -> None:
        """Initialize Enhanced RAG Agent with medical safety and analysis capabilities.

        Args:
            docs_folder: Path to PDF documents folder
            api_key: NVIDIA API key
            vector_db_path: Path for vector database storage
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            embedding_model_name: Optional embedding model name
            enable_preflight_embedding: Forces question-time preflight embedding when True
            append_disclaimer_in_answer: Controls whether MEDICAL_DISCLAIMER is appended to answer text
            force_disclaimer_in_answer: When True, forces disclaimer in answer regardless of rails;
                when False, skips disclaimer; when None (default), uses standard behavior
            guardrails_config_path: Path to guardrails configuration file
            enable_synthesis: Whether to enable synthesis analysis (default: False)
            enable_ddi_analysis: Whether to enable DDI/PK analysis (default: False)
            safety_mode: Safety validation mode ("strict", "balanced", "permissive")
            default_safe: Deprecated flag for backward compatibility

        Note:
            Advanced analysis (synthesis, DDI/PK) is disabled by default to avoid
            surprise performance costs. Set enable_analysis=True when calling query methods
            to enable these features.
        """
        self.base_agent = RAGAgent(
            docs_folder=docs_folder,
            api_key=api_key,
            vector_db_path=vector_db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_name,
            enable_preflight_embedding=enable_preflight_embedding,
            append_disclaimer_in_answer=append_disclaimer_in_answer,
        )

        self.guardrails_config_path = guardrails_config_path
        self.force_disclaimer_in_answer = force_disclaimer_in_answer
        self.enable_synthesis = enable_synthesis
        self.enable_ddi_analysis = enable_ddi_analysis
        self.safety_mode = (safety_mode or "balanced").lower()
        if self.safety_mode not in {"strict", "balanced", "permissive"}:
            logger.warning("Unknown safety_mode '%s'; defaulting to 'balanced'", safety_mode)
            self.safety_mode = "balanced"

        # When True, ask_question delegates to the safe pipeline and returns its dict payload.
        self._default_safe = bool(default_safe)
        if self._default_safe:
            logger.warning(
                "The 'default_safe' flag is deprecated and will be removed in a future version. "
                "Use ask_question_safe_sync() explicitly when you need the guarded response structure."
            )

        # Configuration flags - indicate what was requested at initialization
        self.guardrails_configured = bool(guardrails_config_path)
        self.synthesis_configured = enable_synthesis
        self.ddi_analysis_configured = enable_ddi_analysis

        # Usage flags - indicate what's actually available/enabled after initialization
        self.guardrails_available = False
        self.synthesis_available = False
        self.ddi_analysis_available = False

        self.guardrails: Optional[MedicalGuardrails] = None
        self.synthesis_engine: Optional[SynthesisEngine] = None
        self.ddi_processor: Optional[DDIPKProcessor] = None
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.safety_metrics: Dict[str, Any] = {
            "total_queries": 0,
            "blocked_queries": 0,
            "input_validation_failures": 0,
            "output_validation_failures": 0,
            "analysis_failures": 0,
            "retrieval_validation_failures": 0,
            "last_processing_time": 0.0,
            "safety_mode": self.safety_mode,
            "guardrails_enabled": False,
            "guardrails_active": False,
            "guardrails_configured": bool(guardrails_config_path),
            "guardrails_used": False,
            "pii_blocks": 0,
            "jailbreak_detections": 0,
            "disclaimer_enforcements": 0,
            "rails_usage": {"input": False, "output": False},
            "last_stage_timings": {},
        }

        # Enhanced configuration and PubMed integration
        self.config = config or EnhancedRAGConfig.from_env()
        self._pubmed_integration = PubMedIntegrationManager(
            config=self.config,
            pubmed_query_engine=pubmed_query_engine,
            pubmed_scraper=pubmed_scraper
        )

        # Maintain backward compatibility for metrics
        self.pubmed_metrics = self._pubmed_integration.pubmed_metrics

        # Thread-safe locks for shared state
        self._metrics_lock = threading.RLock()
        self._base_call_lock = threading.RLock()
        self._components_lock = threading.RLock()
        self._components_initialized = False

    @property
    def pubmed_scraper(self) -> Optional[EnhancedPubMedScraper]:
        """Get the PubMed scraper instance. Read-only access."""
        return self._pubmed_integration.pubmed_scraper

    @property
    def pubmed_query_engine(self) -> Optional[EnhancedQueryEngine]:
        """Get the PubMed query engine instance. Read-only access."""
        return self._pubmed_integration.pubmed_query_engine

    # Backward compatibility aliases for deprecated private attributes
    @property
    def _pubmed_scraper(self) -> Optional[EnhancedPubMedScraper]:
        """Deprecated: Use pubmed_scraper property instead."""
        return self._pubmed_integration.pubmed_scraper

    @property
    def _pubmed_query_engine(self) -> Optional[EnhancedQueryEngine]:
        """Deprecated: Use pubmed_query_engine property instead."""
        return self._pubmed_integration.pubmed_query_engine

    def _pubmed_available(self) -> bool:
        """Deprecated: Use pubmed_available property instead."""
        return self._pubmed_integration.is_pubmed_available()

    @property
    def pubmed_available(self) -> bool:
        """Check if PubMed integration is available."""
        return self._pubmed_integration.is_pubmed_available()

    def _initialize_components(self) -> None:
        """Initialise guardrails, synthesis engine, and DDI processor with health checks."""
        # Medical guardrails
        guardrails_status: Dict[str, Any] = {"status": "unconfigured"}
        if self.guardrails_config_path:
            try:
                self.guardrails = MedicalGuardrails(self.guardrails_config_path)
                guardrails_status = {"status": "ready", "config_path": self.guardrails_config_path}
                self.safety_metrics["guardrails_enabled"] = bool(self.guardrails and self.guardrails.enabled)
                self.safety_metrics["guardrails_active"] = bool(self.guardrails and self.guardrails.enabled)
                self.guardrails_available = bool(self.guardrails and self.guardrails.enabled)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to initialize medical guardrails: %s", exc)
                guardrails_status = {
                    "status": "error",
                    "message": str(exc),
                }
                self.safety_metrics["guardrails_enabled"] = False
                self.safety_metrics["guardrails_active"] = False
        else:
            logger.info("Medical guardrails configuration path not provided; running without guardrails")
            self.safety_metrics["guardrails_enabled"] = False
            self.safety_metrics["guardrails_active"] = False
            self.guardrails_available = False
        self.component_health["medical_guardrails"] = guardrails_status

        # Synthesis engine
        synthesis_status: Dict[str, Any] = {"status": "disabled"}
        if self.synthesis_configured:
            try:
                self.synthesis_engine = SynthesisEngine()
                synthesis_status = {"status": "ready"}
                self.synthesis_available = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to initialize synthesis engine: %s", exc)
                synthesis_status = {"status": "error", "message": str(exc)}
                self.enable_synthesis = False
                self.synthesis_available = False
        else:
            logger.info("Synthesis engine disabled")
            self.synthesis_available = False
        self.component_health["synthesis_engine"] = synthesis_status

        # DDI/PK processor
        ddi_status: Dict[str, Any] = {"status": "disabled"}
        if self.ddi_analysis_configured:
            try:
                self.ddi_processor = DDIPKProcessor()
                ddi_status = {"status": "ready"}
                self.ddi_analysis_available = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to initialize DDI/PK processor: %s", exc)
                ddi_status = {"status": "error", "message": str(exc)}
                self.enable_ddi_analysis = False
                self.ddi_analysis_available = False
        else:
            logger.info("DDI/PK analysis disabled")
            self.ddi_analysis_available = False
        self.component_health["ddi_pk_processor"] = ddi_status

        # Initialize PubMed component health
        self._update_pubmed_component_health()

        self._components_initialized = True

    # ------------------------------------------------------------------
    # PubMed integration hooks
    # ------------------------------------------------------------------

    def _ensure_pubmed_components(self) -> None:
        """Ensure PubMed components are initialized via the integration manager."""
        self._pubmed_integration.initialize_pubmed_components()

    def _update_pubmed_component_health(self) -> None:
        """Update PubMed component health in the main component health dictionary."""
        pubmed_health = self._pubmed_integration.get_pubmed_health()
        if pubmed_health:
            self.component_health["pubmed_integration"] = pubmed_health

    def _should_activate_pubmed_for_query(self, *, force_pubmed: bool = False) -> bool:
        """Check if PubMed should be activated for this query."""
        return self._pubmed_integration.should_activate_pubmed_for_query(force_pubmed=force_pubmed)

    def _filter_pubmed_results(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter PubMed results for relevance and quality."""
        return self._pubmed_integration.filter_pubmed_results(payload)

    def _record_pubmed_success(self, mode: str, info: Dict[str, Any]) -> None:
        with self._metrics_lock:
            self.pubmed_metrics["total_queries"] += 1
            if info.get("cache_hit"):
                self.pubmed_metrics["cache_hits"] += 1
            self.pubmed_metrics["last_latency_ms"] = info.get("latency_ms")
            self.pubmed_metrics["last_mode"] = mode
            self.pubmed_metrics["last_result_count"] = len(info.get("results") or [])
            self.pubmed_metrics["last_error"] = None
            self.pubmed_metrics["last_updated"] = time.time()

    def _record_pubmed_failure(self, mode: str, info: Dict[str, Any]) -> None:
        with self._metrics_lock:
            self.pubmed_metrics["total_queries"] += 1
            self.pubmed_metrics["error_count"] += 1
            self.pubmed_metrics["last_mode"] = mode
            self.pubmed_metrics["last_latency_ms"] = info.get("latency_ms")
            self.pubmed_metrics["last_error"] = info.get("error")
            self.pubmed_metrics["last_updated"] = time.time()

    def _record_pubmed_skip(self, mode: str, info: Dict[str, Any]) -> None:
        with self._metrics_lock:
            self.pubmed_metrics["last_mode"] = mode
            self.pubmed_metrics["last_error"] = info.get("reason") or info.get("error")
            self.pubmed_metrics["last_latency_ms"] = info.get("latency_ms")
            self.pubmed_metrics["last_updated"] = time.time()

    def _execute_pubmed_query(
        self,
        query: str,
        *,
        max_items: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        allow_runtime_extraction_for_filters: bool = False,
        mode: str,
    ) -> Dict[str, Any]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            info = {"enabled": False, "reason": "empty_query"}
            self._record_pubmed_skip(mode, info)
            return info

        if not self._should_activate_pubmed_for_query():
            info = {"enabled": False, "reason": "pubmed_feature_disabled"}
            self._record_pubmed_skip(mode, info)
            return info

        self._ensure_pubmed_components()
        if not self._pubmed_available():
            reason = self.component_health.get("pubmed_integration", {}).get("message") or "pubmed_unavailable"
            info = {"enabled": False, "reason": reason}
            self._record_pubmed_skip(mode, info)
            return info

        engine = self._pubmed_integration.pubmed_query_engine
        scraper = self._pubmed_integration.pubmed_scraper
        max_items = max_items if max_items is not None else self.config.max_external_results
        if max_items <= 0:
            max_items = self.config.max_external_results or 10

        start_time = time.perf_counter()
        try:
            response = engine.process_pharmaceutical_query(  # type: ignore[union-attr]
                normalized_query,
                max_items=max_items,
                filters=filters,
                allow_runtime_extraction_for_filters=allow_runtime_extraction_for_filters,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            info = {
                "enabled": True,
                "success": False,
                "error": str(exc),
                "latency_ms": latency_ms,
            }
            self._record_pubmed_failure(mode, info)
            info["fallback_to_local"] = bool(self.config.fallback_to_local_on_error)
            return info

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        cache_payload = response.get("cache") or {}
        cache_hit = bool(response.get("cache_hit") or cache_payload.get("cache_hit"))
        filtered_results = self._filter_pubmed_results(response)
        info = {
            "enabled": True,
            "success": True,
            "latency_ms": latency_ms,
            "payload": response,
            "results": filtered_results,
            "cache_hit": cache_hit,
        }
        self._record_pubmed_success(mode, info)

        if scraper is not None:
            with self._pubmed_lock:
                self._refresh_pubmed_component_health_locked(scraper)

        return info

    def _disabled_pubmed_payload(self, reason: str = "pubmed_disabled") -> Dict[str, Any]:
        return {"enabled": False, "success": False, "reason": reason, "results": []}

    def _render_pubmed_sources(self, results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        rendered: List[Dict[str, Any]] = []
        for idx, item in enumerate(results or []):
            # Normalize metadata to ensure pubmed_id is available for UI detection
            meta = dict(item)
            # Ensure pubmed_id is set for backward compatibility
            meta.setdefault("pubmed_id", meta.get("pmid") or meta.get("id"))

            rendered.append(
                {
                    "id": meta.get("pmid") or meta.get("id") or f"pubmed-{idx}",
                    "pmid": meta.get("pmid") or meta.get("id"),
                    "title": meta.get("title"),
                    "abstract": meta.get("abstract"),
                    "authors": meta.get("authors"),
                    "journal": meta.get("journal"),
                    "publication_date": meta.get("publication_date"),
                    "ranking_score": meta.get("ranking_score"),
                    "source_type": "pubmed",
                    "metadata": meta,
                }
            )
        return rendered

    def _combine_sources(
        self,
        local_payload: Optional[Dict[str, Any]],
        pubmed_payload: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        if local_payload:
            sources.extend(local_payload.get("sources", []))
        if pubmed_payload:
            sources.extend(self._render_pubmed_sources(pubmed_payload.get("results")))
        return sources

    def _build_hybrid_response(
        self,
        *,
        query: str,
        mode: str,
        local_payload: Optional[Dict[str, Any]],
        pubmed_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "local": local_payload,
            "pubmed": pubmed_payload,
            "timestamp": time.time(),
            "config": self.config.summarize_flags(),
        }

        # Add timing fields
        lat_ms = pubmed_payload.get("latency_ms")
        if local_payload and isinstance(local_payload.get("processing_time"), (int, float)):
            response["processing_time"] = float(local_payload["processing_time"])
        elif isinstance(lat_ms, (int, float)):
            response["processing_time"] = float(lat_ms) / 1000.0
        else:
            response["processing_time"] = 0.0
        response.setdefault("status", {})["pubmed_latency_ms"] = lat_ms

        response["sources"] = self._combine_sources(local_payload, pubmed_payload)
        if local_payload:
            response["answer"] = local_payload.get("answer")
            response["disclaimer"] = local_payload.get("disclaimer") or self.base_agent.get_disclaimer()
        else:
            response["answer"] = None
            response["disclaimer"] = self.base_agent.get_disclaimer()

            # For PubMed-only mode with successful results but no answer, provide a summary
            if mode == "pubmed_only" and pubmed_payload.get("success") and response["sources"]:
                article_count = len([s for s in response["sources"] if s.get("source_type") == "pubmed"])
                if article_count > 0:
                    response["answer"] = f"Found {article_count} PubMed article{'s' if article_count != 1 else ''} relevant to your query. See Sources for details."

        response["status"] = {
            "pubmed_enabled": pubmed_payload.get("enabled", False),
            "pubmed_success": pubmed_payload.get("success", False),
            "pubmed_reason": pubmed_payload.get("reason"),
            "cache_hit": pubmed_payload.get("cache_hit"),
            "latency_ms": pubmed_payload.get("latency_ms"),
        }

        # Add query_mode alias for backward compatibility with tests
        response["query_mode"] = mode

        return response

    def _classify_query_mode(self, query: str) -> str:
        """Classify query routing mode: local, pubmed, combined, or hybrid."""
        return self._pubmed_integration.classify_query_mode(query)

    def _classify_query(self, query: str) -> str:
        """Classify query as medical, general, or pharmaceutical.

        Args:
            query: The query string to classify

        Returns:
            Query type: "medical", "general", or "pharmaceutical"
        """
        return self._pubmed_integration.classify_query(query)

    # ------------------------------------------------------------------
    # Public PubMed-aware interfaces
    # ------------------------------------------------------------------

    def ask_question_with_pubmed(
        self,
        query: str,
        k: int = 4,
        *,
        max_external_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        allow_runtime_extraction_for_filters: bool = False,
        force_pubmed: bool = False,
    ) -> Dict[str, Any]:
        # Check rollout before proceeding
        if not self._should_activate_pubmed_for_query(force_pubmed=force_pubmed):
            # Return local-only response
            base_response = self._call_base_locked(query, k, None)
            local_payload = self._serialize_base_response(base_response)
            return self._build_hybrid_response(
                query=query,
                mode="local_only",
                local_payload=local_payload,
                pubmed_payload=self._disabled_pubmed_payload("rollout_not_active"),
            )

        base_response = self._call_base_locked(query, k, None)
        local_payload = self._serialize_base_response(base_response)
        pubmed_payload = self._execute_pubmed_query(
            query,
            max_items=max_external_results,
            filters=filters,
            allow_runtime_extraction_for_filters=allow_runtime_extraction_for_filters,
            mode="combined",
        )
        if "results" not in pubmed_payload:
            pubmed_payload["results"] = []
        return self._build_hybrid_response(
            query=query,
            mode="combined",
            local_payload=local_payload,
            pubmed_payload=pubmed_payload,
        )

    def ask_question_pubmed_only(
        self,
        query: str,
        *,
        max_external_results: Optional[int] = None,
        max_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        allow_runtime_extraction_for_filters: bool = False,
        fallback_to_local: Optional[bool] = None,
        k: int = 4,
        force_pubmed: bool = False,
        strict: bool = False,
    ) -> Dict[str, Any]:
        # Check rollout before proceeding
        if not self._should_activate_pubmed_for_query(force_pubmed=force_pubmed):
            # Return error response when rollout excludes user
            return {
                "query": query,
                "mode": "pubmed_only",
                "enabled": False,
                "success": False,
                "error": "PubMed integration not available in current rollout cohort",
                "results": [],
                "status": {
                    "pubmed_enabled": False,
                    "rollout_active": self.config.is_rollout_active(),
                    "rollout_percentage": self.config.rollout_percentage,
                    "in_rollout_cohort": False,
                },
            }

        # Handle parameter alias for backward compatibility
        effective_max_results = max_external_results
        if max_external_results is not None and max_results is not None:
            # Both provided, prefer max_external_results
            import warnings
            warnings.warn(
                "Both max_external_results and max_results provided to ask_question_pubmed_only. "
                "Using max_external_results. max_results is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        elif max_results is not None:
            effective_max_results = max_results
            warnings.warn(
                "max_results parameter is deprecated. Use max_external_results instead.",
                DeprecationWarning,
                stacklevel=2
            )

        # Check if strict mode is enabled (either via parameter or config)
        strict_mode = strict or os.getenv("RAG_PUBMED_STRICT_MODE", "false").lower() in ("true", "1", "yes", "on")

        fallback = self.config.fallback_to_local_on_error if fallback_to_local is None else bool(fallback_to_local)
        pubmed_payload = self._execute_pubmed_query(
            query,
            max_items=effective_max_results,
            filters=filters,
            allow_runtime_extraction_for_filters=allow_runtime_extraction_for_filters,
            mode="pubmed_only",
        )
        local_payload: Optional[Dict[str, Any]] = None

        # Only fallback if not in strict mode
        if not strict_mode and fallback and pubmed_payload.get("enabled") and not pubmed_payload.get("success"):
            base_response = self._call_base_locked(query, k, None)
            local_payload = self._serialize_base_response(base_response)
        if "results" not in pubmed_payload:
            pubmed_payload["results"] = []

        # Build the response
        response = self._build_hybrid_response(
            query=query,
            mode="pubmed_only",
            local_payload=local_payload,
            pubmed_payload=pubmed_payload,
        )

        # Add top-level error field when PubMed fails but was enabled
        if not pubmed_payload.get("success") and pubmed_payload.get("enabled"):
            response["error"] = {
                "message": pubmed_payload.get("error") or pubmed_payload.get("reason", "PubMed query failed"),
                "fallback": bool(local_payload)
            }

        return response

    def ask_question_hybrid(
        self,
        query: str,
        k: int = 4,
        *,
        max_external_results: Optional[int] = None,
        local_k: Optional[int] = None,
        pubmed_max: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        allow_runtime_extraction_for_filters: bool = False,
        force_pubmed: bool = False,
    ) -> Dict[str, Any]:
        # Handle parameter aliases for backward compatibility
        effective_k = k
        effective_max_external = max_external_results

        if local_k is not None:
            if k != 4:  # Default value
                import warnings
                warnings.warn(
                    "Both k and local_k provided to ask_question_hybrid. "
                    "Using k. local_k is deprecated.",
                    DeprecationWarning,
                    stacklevel=2
                )
            else:
                effective_k = local_k
                warnings.warn(
                    "local_k parameter is deprecated. Use k instead.",
                    DeprecationWarning,
                    stacklevel=2
                )

        if max_external_results is not None and pubmed_max is not None:
            warnings.warn(
                "Both max_external_results and pubmed_max provided to ask_question_hybrid. "
                "Using max_external_results. pubmed_max is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )
        elif pubmed_max is not None:
            effective_max_external = pubmed_max
            warnings.warn(
                "pubmed_max parameter is deprecated. Use max_external_results instead.",
                DeprecationWarning,
                stacklevel=2
            )

        routing_mode = self._classify_query_mode(query)
        if routing_mode == "local":
            base_response = self._call_base_locked(query, effective_k, None)
            local_payload = self._serialize_base_response(base_response)
            response = self._build_hybrid_response(
                query=query,
                mode="local_only",
                local_payload=local_payload,
                pubmed_payload=self._disabled_pubmed_payload("router_local_only"),
            )
            response.setdefault("status", {})["router_mode"] = routing_mode
            return response

        if routing_mode == "pubmed":
            response = self.ask_question_pubmed_only(
                query,
                max_external_results=effective_max_external,
                filters=filters,
                allow_runtime_extraction_for_filters=allow_runtime_extraction_for_filters,
                k=effective_k,
                force_pubmed=force_pubmed,
            )
            response.setdefault("status", {})["router_mode"] = routing_mode
            return response

        combined = self.ask_question_with_pubmed(
            query,
            k=effective_k,
            max_external_results=effective_max_external,
            filters=filters,
            allow_runtime_extraction_for_filters=allow_runtime_extraction_for_filters,
            force_pubmed=force_pubmed,
        )
        combined.setdefault("status", {})["router_mode"] = routing_mode
        combined["mode"] = "hybrid"
        return combined

    def _ensure_components_initialized(self) -> None:
        """Initialize components lazily on first use."""
        if not self._components_initialized:
            with self._components_lock:
                # Double-check pattern to prevent race conditions
                if not self._components_initialized:
                    self._initialize_components()

    def _rails_configured(self) -> bool:
        """Return True when NeMo Guardrails rails are available for execution."""
        guardrails = self.guardrails
        if not guardrails:
            return False
        if not getattr(guardrails, "enabled", False):
            return False

        # Check for validation methods instead of nemo_rails attribute
        has_input_rails = hasattr(guardrails, "run_input_validation_rails")
        has_output_rails = hasattr(guardrails, "run_output_validation_rails")
        has_retrieval_rails = hasattr(guardrails, "run_retrieval_validation_rails")

        return has_input_rails or has_output_rails or has_retrieval_rails

    def _inc_metric(self, key: str, value: int = 1) -> None:
        """Thread-safe increment of safety metrics counter."""
        with self._metrics_lock:
            self.safety_metrics[key] = self.safety_metrics.get(key, 0) + value

    def _set_metric(self, key: str, value: Any) -> None:
        """Thread-safe assignment to safety metrics."""
        with self._metrics_lock:
            self.safety_metrics[key] = value

    def _call_base_locked(self, query: str, k: int, disclaimer_override: Optional[bool]) -> RAGResponse:
        """Call base agent with thread safety."""
        with self._base_call_lock:
            return self.base_agent.ask_question(query, k, disclaimer_already_present=disclaimer_override)

    def _ensure_disclaimer(
        self,
        answer: str,
        output_metadata: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Centralize disclaimer enforcement logic to avoid duplication.

        Args:
            answer: Answer text to potentially modify
            output_metadata: Output metadata dictionary that may be updated

        Returns:
            Tuple of (updated_answer, updated_output_metadata)
        """
        # Check if disclaimer already exists to prevent duplication
        if contains_medical_disclaimer(answer) or output_metadata.get("disclaimer_added"):
            # Disclaimer already present, skip adding
            return answer, output_metadata

        # Check if we should force disclaimer regardless of rails configuration
        if self.force_disclaimer_in_answer is True:
            # Force apply disclaimer
            updated_answer = self.base_agent.apply_disclaimer(answer)
        elif self.force_disclaimer_in_answer is False:
            # Force skip disclaimer
            updated_answer = answer
        else:
            # Use default behavior through base agent
            updated_answer = self.base_agent.apply_disclaimer(answer)

        # Update metadata to track disclaimer addition
        if updated_answer != answer:
            output_metadata = dict(output_metadata)  # Make a copy to avoid mutation
            output_metadata["disclaimer_added"] = True

        return updated_answer, output_metadata

    def _run_coroutine_blocking(self, coro: "Awaitable[Any]", *, allow_nested: bool = True) -> Any:
        """Execute an async coroutine from synchronous code, tolerating active loops.

        This method handles event loop edge cases that can occur in Jupyter notebooks
        or other async environments where an event loop is already running.

        Limitations:
        - In some Jupyter/async environments, the coroutine may not have access to
          the same context (e.g., task-local storage) as the calling context
        - Performance overhead due to thread switching when nested execution is required
        - The executor-based approach maintains better loop affinity than creating
          a completely separate thread but still requires thread boundaries
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():  # pragma: no cover - runtime defensive path
            if not allow_nested:
                raise RuntimeError("Cannot execute coroutine while event loop is running")

            # Use the running loop's executor to run asyncio.run(coro) in a separate thread
            # This maintains better loop affinity compared to manually creating threads
            import concurrent.futures

            def _runner() -> Any:
                # Create a new event loop in the executor thread
                # This avoids conflicts with the main event loop
                return asyncio.run(coro)

            # Use the running loop's default executor for better context preservation
            future = loop.run_in_executor(None, _runner)

            # Block until the future completes, but in a thread-safe way
            result_container: Dict[str, Any] = {}
            error_container: Dict[str, BaseException] = {}

            def _waiter():
                try:
                    result_container["value"] = future.result()
                except BaseException as exc:
                    error_container["error"] = exc

            import threading
            wait_thread = threading.Thread(target=_waiter, daemon=True)
            wait_thread.start()
            wait_thread.join()

            if error_container:
                raise error_container["error"]
            return result_container.get("value")

        return asyncio.run(coro)

    async def ask_question_safe(
        self,
        query: str,
        k: int = 4,
        enable_analysis: bool = False,
        analysis_options: Optional[Dict[str, Any]] = None,
    ) -> EnhancedRAGPayload:
        """Async safe query entry point with validation and advanced analysis.

        Returns:
            EnhancedRAGPayload: Dictionary containing answer, sources, safety metadata,
                analysis results, and component health information. See EnhancedRAGPayload
                for the complete structure.
        """
        wall_start_time = time.time()
        start_perf = time.perf_counter()
        self._inc_metric("total_queries")
        analysis_options = analysis_options or {}
        safety_metadata: Dict[str, Any] = {
            "input_validation": None,
            "retrieval_validation": None,
            "output_validation": None,
            "analysis": {},
        }
        stage_timings: Dict[str, float] = {}

        try:
            stage_start = time.perf_counter()
            input_validation = await self._validate_input_safely(query)
            stage_timings["input_validation"] = time.perf_counter() - stage_start
            safety_metadata["input_validation"] = input_validation

            if not input_validation.get("is_valid", True):
                self._inc_metric("blocked_queries")
                self._inc_metric("input_validation_failures")

                # Use existing handler for consistency
                context = {
                    "validation": input_validation,
                    "stage": "input",
                    "query": query,
                    "base_payload": {
                        "analysis": {"status": "skipped", "reason": "input_validation_failed"},
                        "timings": {"input_validation": stage_timings["input_validation"]},
                    }
                }
                return self._handle_guardrails_exception(context)

            sanitized_query = input_validation.get("sanitized_query") or query
            rails_configured = self._rails_configured()
            base_append_enabled = bool(self.base_agent.append_disclaimer_in_answer)
            disclaimer_deferred = bool(rails_configured and base_append_enabled)
            disclaimer_override: Optional[bool] = True if disclaimer_deferred else None

            # Execute query without mutating shared base agent flags
            stage_start = time.perf_counter()
            base_response = await asyncio.to_thread(
                self._call_base_locked,
                sanitized_query,
                k,
                disclaimer_override,
            )
            stage_timings["base_rag"] = time.perf_counter() - stage_start
            stage_timings["base_rag_agent_processing"] = getattr(base_response, "processing_time", 0.0)

            base_payload = self._serialize_base_response(base_response)
            base_payload["sanitized_query"] = sanitized_query
            safety_metadata["base_processing_time"] = base_response.processing_time

            needs_rebuild = bool(getattr(base_response, "needs_rebuild", False))
            papers = base_payload.get("sources", [])
            analysis_results: Dict[str, Any] = {}
            analysis_metadata: Dict[str, Any] = {}
            stage_timings["analysis_total"] = 0.0
            stage_timings["analysis_synthesis"] = 0.0
            stage_timings["analysis_ddi_pk"] = 0.0
            stage_timings["retrieval_validation"] = 0.0
            stage_timings["output_validation"] = 0.0

            retrieval_validation: Optional[Dict[str, Any]] = None
            output_validation: Optional[Dict[str, Any]] = None
            output_metadata: Dict[str, Any] = {}
            output_rails_used = False
            disclaimer_enforced = False

            if needs_rebuild:
                # Create consistent payload using _package_enhanced_response
                base_payload = self._serialize_base_response(base_response)
                answer = self.base_agent.apply_disclaimer(
                    "The vector database needs to be rebuilt. Please reprocess your documents."
                )
                base_payload["answer"] = answer
                base_payload["needs_rebuild"] = True

                # Construct safety metadata with validation objects
                safety_metadata = {
                    "input_validation": input_validation,
                    "retrieval_validation": {"passed": True},
                    "output_validation": {
                        "is_valid": True,
                        "issues": [],
                        "recommendations": ["Vector database rebuild required"],
                        "validated_response": answer,
                        "nemo_rails_used": False,
                        "metadata": {
                            "guardrails_enabled": bool(self.guardrails and self.guardrails.enabled),
                            "warnings": ["Vector database rebuild required"]
                        }
                    },
                    "guardrails_triggered": False,
                    "timings": stage_timings,
                    "base_processing_time": base_response.processing_time,
                    "rails_usage": {"input": False, "output": False},
                }

                # Return consistent payload
                return self._package_enhanced_response(
                    base_payload,
                    analysis_results={"status": "skipped"},
                    safety_metadata=safety_metadata
                )

            if papers:
                stage_start = time.perf_counter()
                retrieval_validation = await self._validate_retrieval_safely(papers, sanitized_query)
                stage_timings["retrieval_validation"] = time.perf_counter() - stage_start
                if retrieval_validation:
                    safety_metadata["retrieval_validation"] = retrieval_validation
                    filtered_docs = retrieval_validation.get("filtered_documents")
                    if isinstance(filtered_docs, list):
                        papers = filtered_docs
                        base_payload["sources"] = filtered_docs
                else:
                    safety_metadata["retrieval_validation"] = None
            else:
                safety_metadata["retrieval_validation"] = None

            if (
                retrieval_validation
                and retrieval_validation.get("documents_valid") is False
                and self.safety_mode == "strict"
            ):
                self._inc_metric("blocked_queries")
                self._inc_metric("retrieval_validation_failures")

                # Use existing handler for consistency
                context = {
                    "validation": retrieval_validation,
                    "stage": "retrieval",
                    "query": query,
                    "base_payload": {
                        "analysis": {
                            "status": "skipped",
                            "reason": "retrieval_validation_failed",
                            "metadata": {
                                "analysis_type": self._determine_analysis_type(sanitized_query),
                                "timings": {"total": 0.0, "synthesis": 0.0, "ddi_pk": 0.0},
                                "synthesis_status": "skipped",
                                "ddi_status": "skipped",
                            },
                        },
                        "timings": stage_timings,
                        "retrieval_validation": retrieval_validation,
                    }
                }
                return self._handle_guardrails_exception(context)
            else:
                stage_start = time.perf_counter()
                output_validation = await self._validate_output_safely(
                    base_payload.get("answer", ""),
                    papers,
                )
                stage_timings["output_validation"] = time.perf_counter() - stage_start
                safety_metadata["output_validation"] = output_validation
                output_metadata = dict(output_validation.get("metadata", {}) or {})
                output_validation["metadata"] = output_metadata

                # Set base-agent guardrail metadata when available to keep disclaimer semantics unified
                if output_validation.get("metadata"):
                    self.base_agent.set_guardrail_metadata(output_validation.get("metadata"))

                if not output_validation.get("is_valid", True):
                    self._inc_metric("output_validation_failures")
                    if self.safety_mode == "strict":
                        self._inc_metric("blocked_queries")

                        # Use existing handler for consistency
                        context = {
                            "validation": output_validation,
                            "stage": "output",
                            "query": query,
                            "base_payload": {
                                "analysis": analysis_results or {"status": "skipped"},
                                "timings": stage_timings,
                                "retrieval_validation": retrieval_validation,
                            }
                        }
                        return self._handle_guardrails_exception(context)
                    elif self.safety_mode == "balanced":
                        warning = output_validation.get("issues") or ["Medical output validation concerns detected"]
                        base_payload["safety_warnings"] = warning

                validated_answer = output_validation.get("validated_response")
                if isinstance(validated_answer, str) and validated_answer.strip():
                    base_payload["answer"] = validated_answer

                output_rails_used = bool(output_validation.get("nemo_rails_used"))
                if disclaimer_deferred:
                    answer_text = base_payload.get("answer", "")
                    updated_answer, output_metadata = self._ensure_disclaimer(
                        answer_text,
                        output_metadata,
                    )
                    disclaimer_enforced = updated_answer != answer_text
                    base_payload["answer"] = updated_answer

                if enable_analysis and not needs_rebuild:
                    stage_start = time.perf_counter()
                    analysis_results = await self._perform_advanced_analysis(
                        papers,
                        sanitized_query,
                        analysis_options,
                    )
                    analysis_duration = time.perf_counter() - stage_start
                    stage_timings["analysis_total"] = analysis_duration
                    analysis_metadata = analysis_results.get("metadata", {}) or {}
                    timing_metadata = analysis_metadata.get("timings", {}) or {}
                    stage_timings["analysis_synthesis"] = timing_metadata.get("synthesis", 0.0)
                    stage_timings["analysis_ddi_pk"] = timing_metadata.get("ddi_pk", 0.0)
                    safety_metadata["analysis"] = analysis_metadata
                    if analysis_results.get("status") in {"error", "completed_with_errors"}:
                        self._inc_metric("analysis_failures")
            safety_metadata["timings"] = stage_timings

            enhanced_response = self._package_enhanced_response(
                base_payload,
                analysis_results,
                safety_metadata,
            )

            input_rails_used = bool(input_validation.get("nemo_rails_used"))
            rails_snapshot = {"input": input_rails_used, "output": output_rails_used}
            self._set_metric("rails_usage", rails_snapshot)
            safety_metadata["rails_usage"] = rails_snapshot

            # Update guardrails_active metric based on actual usage
            guardrails_actively_used = bool(
                input_validation.get("metadata", {}).get("guardrails_enabled")
                or (
                    output_validation
                    and output_validation.get("metadata", {}).get("guardrails_enabled")
                )
                or (
                    retrieval_validation
                    and retrieval_validation.get("metadata", {}).get("guardrails_enabled")
                )
            )
            self._set_metric("guardrails_active", guardrails_actively_used)

            def _flag_from_issues(issues: List[Any], keywords: Tuple[str, ...]) -> bool:
                for issue in issues:
                    if isinstance(issue, str):
                        lowered = issue.lower()
                        if any(keyword in lowered for keyword in keywords):
                            return True
                return False

            pii_detected = _flag_from_issues(input_validation.get("issues", []), ("pii", "phi", "personal data", "personal information"))
            jailbreak_detected = _flag_from_issues(input_validation.get("issues", []), ("jailbreak", "ignore", "bypass"))

            # Disclaimer enforcement metrics may undercount when guardrails add disclaimer text
            disclaimer_enforced = (
                disclaimer_enforced or  # From _ensure_disclaimer
                bool(output_metadata.get("disclaimer_added")) or  # From guardrails
                (output_validation.get("validated_response") and contains_medical_disclaimer(output_validation.get("validated_response")))  # Detection in validated response
            )

            if pii_detected:
                self._inc_metric("pii_blocks")
            if jailbreak_detected:
                self._inc_metric("jailbreak_detections")
            if disclaimer_enforced:
                self._inc_metric("disclaimer_enforcements")

            # Update guardrails_used metric based on actual rails usage
            guardrails_actually_used = bool(
                input_validation.get("nemo_rails_used") or
                (output_validation and output_validation.get("nemo_rails_used")) or
                (retrieval_validation and retrieval_validation.get("nemo_rails_used"))
            )
            self._set_metric("guardrails_used", guardrails_actually_used)

            self._set_metric("last_processing_time", time.perf_counter() - start_perf)
            self._set_metric("last_stage_timings", stage_timings)

            enhanced_response.setdefault("metrics", {})
            enhanced_response["metrics"].update(
                {
                    "stage_timings": stage_timings,
                    "rails_usage": rails_snapshot,
                    "pii_detected": pii_detected,
                    "jailbreak_detected": jailbreak_detected,
                    "disclaimer_enforced": disclaimer_enforced,
                }
            )

            enhanced_response.setdefault("safety", {})
            enhanced_response["safety"].setdefault("input_validation", input_validation)
            enhanced_response["safety"].setdefault("output_validation", output_validation)
            enhanced_response.setdefault("analysis", analysis_results or {})
            enhanced_response.setdefault("sanitized_query", sanitized_query)

            return enhanced_response

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Enhanced RAG query processing failed: %s", exc)
            self._set_metric("last_processing_time", time.perf_counter() - start_perf)
            return {
                "answer": "I encountered an unexpected error while processing the request.",
                "error": str(exc),
                "disclaimer": self.base_agent.get_disclaimer(),
                "safety": safety_metadata,
                "needs_rebuild": False,
                "metrics": {
                    "total_queries": self.safety_metrics.get("total_queries"),
                    "blocked_queries": self.safety_metrics.get("blocked_queries"),
                    "analysis_failures": self.safety_metrics.get("analysis_failures"),
                    "input_validation_failures": self.safety_metrics.get("input_validation_failures"),
                    "output_validation_failures": self.safety_metrics.get("output_validation_failures"),
                    "retrieval_validation_failures": self.safety_metrics.get("retrieval_validation_failures"),
                    "last_processing_time": self.safety_metrics.get("last_processing_time"),
                    "safety_mode": self.safety_metrics.get("safety_mode"),
                    "pii_blocks": self.safety_metrics.get("pii_blocks"),
                    "jailbreak_detections": self.safety_metrics.get("jailbreak_detections"),
                    "disclaimer_enforcements": self.safety_metrics.get("disclaimer_enforcements"),
                    "rails_usage": self.safety_metrics.get("rails_usage"),
                    "guardrails_enabled": self.safety_metrics.get("guardrails_enabled"),
                    "guardrails_active": self.safety_metrics.get("guardrails_active"),
                    "last_stage_timings": self.safety_metrics.get("last_stage_timings"),
                },
                "component_health": self.component_health,
            }

    def ask_question(self, query: str, k: int = 4) -> RAGResponse:
        """Return a ``RAGResponse`` from the base RAG agent.

        This method always returns a :class:`~src.rag_agent.RAGResponse` for consistent
        static typing. If you need the enhanced safety payload with validation and
        analysis results, use :meth:`ask_question_safe_sync` instead.

        Note: The ``default_safe`` constructor flag is deprecated. This method will
        always return a ``RAGResponse`` regardless of that flag's value.

        Args:
            query: User query to process
            k: Number of relevant documents to retrieve

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        if self._default_safe:
            logger.warning(
                "ask_question() called with default_safe=True. "
                "Use ask_question_safe_sync() for enhanced payload or initialize without default_safe."
            )
        return self.base_agent.ask_question(query, k)

    def ask_question_safe_sync(
        self,
        query: str,
        k: int = 4,
        enable_analysis: bool = False,
        analysis_options: Optional[Dict[str, Any]] = None,
    ) -> EnhancedRAGPayload:
        """Synchronous safe query entry point with validation and advanced analysis.

        This is the synchronous variant of ask_question_safe that executes the full
        enhanced pipeline including medical guardrails, validation, and analysis.
        Safe to use even inside Jupyter notebooks or other async environments.

        Args:
            query: User query to process
            k: Number of relevant documents to retrieve
            enable_analysis: Whether to run advanced analysis (synthesis, DDI/PK)
            analysis_options: Options for analysis pipeline

        Returns:
            EnhancedRAGPayload: Dictionary containing answer, sources, safety metadata,
                analysis results, and component health information. See EnhancedRAGPayload
                for the complete structure.
        """
        coroutine = self.ask_question_safe(
            query=query,
            k=k,
            enable_analysis=enable_analysis,
            analysis_options=analysis_options,
        )
        return self._run_coroutine_blocking(coroutine)

    def ask_question_payload(
        self,
        query: str,
        k: int = 4,
        enable_analysis: bool = False,
        analysis_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """DEPRECATED: Use ask_question_safe_sync instead for clarity."""
        return self.ask_question_safe_sync(query, k, enable_analysis, analysis_options)

    async def _validate_input_safely(self, query: str) -> Dict[str, Any]:
        """Run medical guardrails input validation when available."""
        self._ensure_components_initialized()
        guardrails = self.guardrails
        if not guardrails or not getattr(guardrails, "enabled", False):
            return {
                "is_valid": bool(query.strip()),
                "issues": [] if query.strip() else ["Empty query"],
                "recommendations": [],
                "sanitized_query": query,
                "nemo_rails_used": False,
                "metadata": {"guardrails_enabled": False},
            }
        rails_capable = self._rails_configured() and hasattr(guardrails, "run_input_validation_rails")
        fallback_validator = getattr(guardrails, "validate_medical_query", None)

        result: Optional[Dict[str, Any]] = None
        rails_exception: Optional[Exception] = None
        rails_used = False

        if rails_capable:
            try:
                result = await guardrails.run_input_validation_rails(query)
                rails_used = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Input validation rails failed; falling back to lightweight validator: %s", exc)
                rails_exception = exc

        if result is None:
            if callable(fallback_validator):
                result = await asyncio.to_thread(fallback_validator, query)
                if result is None:
                    return {
                        "is_valid": bool(query.strip()),
                        "issues": [] if query.strip() else ["Empty query"],
                        "recommendations": [],
                        "sanitized_query": query,
                        "nemo_rails_used": False,
                        "metadata": {
                            "guardrails_enabled": True,
                            "rails_available": rails_capable,
                        },
                    }
                if rails_exception is not None:
                    result.setdefault("metadata", {})
                    result["metadata"]["rails_exception"] = str(rails_exception)
            else:
                return {
                    "is_valid": bool(query.strip()),
                    "issues": [] if query.strip() else ["Empty query"],
                    "recommendations": [],
                    "sanitized_query": query,
                    "nemo_rails_used": False,
                    "metadata": {
                        "guardrails_enabled": True,
                        "rails_available": rails_capable,
                    },
                }

        result.setdefault("sanitized_query", query)
        result.setdefault("issues", [])
        result.setdefault("recommendations", [])
        result.setdefault("metadata", {})
        result.setdefault("nemo_rails_used", rails_used)
        result["metadata"].setdefault("guardrails_enabled", True)
        if rails_exception is not None and callable(fallback_validator):
            result["metadata"].setdefault("rails_exception", str(rails_exception))
        return result

    async def _validate_output_safely(self, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run medical guardrails output validation when available."""
        self._ensure_components_initialized()
        guardrails = self.guardrails
        if not guardrails or not getattr(guardrails, "enabled", False):
            return {
                "is_valid": True,
                "issues": [],
                "recommendations": [],
                "validated_response": response,
                "nemo_rails_used": False,
                "metadata": {"guardrails_enabled": False},
            }

        rails_capable = self._rails_configured() and hasattr(guardrails, "run_output_validation_rails")
        fallback_validator = getattr(guardrails, "validate_medical_response", None)

        result: Optional[Dict[str, Any]] = None
        rails_exception: Optional[Exception] = None
        rails_used = False

        if rails_capable:
            try:
                result = await guardrails.run_output_validation_rails(response, sources)
                rails_used = True
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Output validation rails failed; falling back to lightweight validator: %s", exc)
                rails_exception = exc

        if result is None:
            if callable(fallback_validator):
                result = await asyncio.to_thread(fallback_validator, response, sources)
                if result is None:
                    return {
                        "is_valid": True,
                        "issues": [],
                        "recommendations": [],
                        "validated_response": response,
                        "nemo_rails_used": False,
                        "metadata": {
                            "guardrails_enabled": True,
                            "rails_available": rails_capable,
                        },
                    }
                if rails_exception is not None:
                    result.setdefault("metadata", {})
                    result["metadata"]["rails_exception"] = str(rails_exception)
            else:
                return {
                    "is_valid": True,
                    "issues": [],
                    "recommendations": [],
                    "validated_response": response,
                    "nemo_rails_used": False,
                    "metadata": {
                        "guardrails_enabled": True,
                        "rails_available": rails_capable,
                    },
                }

        result.setdefault("issues", [])
        result.setdefault("recommendations", [])
        result.setdefault("validated_response", response)
        result.setdefault("metadata", {})
        result.setdefault("nemo_rails_used", rails_used)
        result["metadata"].setdefault("guardrails_enabled", True)
        if rails_exception is not None and callable(fallback_validator):
            result["metadata"].setdefault("rails_exception", str(rails_exception))
        return result

    async def _validate_retrieval_safely(
        self,
        papers: List[Dict[str, Any]],
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Run retrieval/document validation when supported by guardrails."""
        self._ensure_components_initialized()
        guardrails = self.guardrails
        if not guardrails or not getattr(guardrails, "enabled", False) or not papers:
            return None

        rails_capable = self._rails_configured() and hasattr(guardrails, "run_retrieval_validation_rails")
        fallback_callable = getattr(guardrails, "_validate_retrieved_documents_fallback", None)

        result: Optional[Dict[str, Any]] = None
        rails_used = False
        try:
            if rails_capable:
                result = await guardrails.run_retrieval_validation_rails(papers, query)
                rails_used = True
            elif callable(fallback_callable):
                result = await asyncio.to_thread(fallback_callable, papers, query)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Retrieval validation failed; using fallback guidance: %s", exc)
            result = {
                "documents_valid": False,
                "filtered_documents": papers,
                "issues": [f"Validation error: {exc}"],
                "recommendations": ["Manual document review recommended."],
                "nemo_rails_used": False,
                "metadata": {"error": str(exc), "guardrails_enabled": True},
            }

        if result is None:
            if callable(fallback_callable) and not rails_capable:
                # No result returned from fallback, treat as neutral pass-through
                result = await asyncio.to_thread(fallback_callable, papers, query)
            else:
                return None

        result.setdefault("documents_valid", True)
        result.setdefault("issues", [])
        result.setdefault("recommendations", [])
        result.setdefault("metadata", {})
        result["metadata"].setdefault("guardrails_enabled", True)
        result["metadata"]["sanitized_query"] = query
        if "filtered_documents" not in result or result["filtered_documents"] is None:
            result["filtered_documents"] = papers
        result.setdefault("nemo_rails_used", rails_used)
        return result

    async def _perform_advanced_analysis(
        self,
        papers: List[Dict[str, Any]],
        query: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run synthesis and DDI/PK analysis as configured."""
        self._ensure_components_initialized()
        analysis_type = self._determine_analysis_type(query)
        analysis_start = time.perf_counter()
        metadata: Dict[str, Any] = {"analysis_type": analysis_type, "timings": {}}
        timings: Dict[str, float] = {"total": 0.0, "synthesis": 0.0, "ddi_pk": 0.0}
        synthesis_available = bool(self.enable_synthesis and self.synthesis_engine)
        ddi_required = analysis_type in {"drug_interaction", "pharmacokinetic"}
        ddi_available = bool(self.enable_ddi_analysis and self.ddi_processor)
        synthesis_status = "pending" if synthesis_available else "disabled"
        if not ddi_required:
            ddi_status = "not_applicable"
        else:
            ddi_status = "pending" if ddi_available else "disabled"

        if not papers:
            if synthesis_status == "pending":
                synthesis_status = "skipped"
            if ddi_status == "pending":
                ddi_status = "skipped"
            metadata["synthesis_status"] = synthesis_status
            metadata["ddi_status"] = ddi_status
            metadata["timings"] = timings
            return {"status": "skipped", "reason": "no_sources", "metadata": metadata}

        results: Dict[str, Any] = {"status": "completed", "metadata": metadata}
        component_errors: Dict[str, str] = {}

        if synthesis_status == "pending":
            synthesis_start = time.perf_counter()
            try:
                meta_summary = await asyncio.to_thread(
                    self.synthesis_engine.generate_meta_summary,
                    papers,
                    query,
                )
                timings["synthesis"] = time.perf_counter() - synthesis_start
                synthesis_status = "completed"
                results["meta_summary"] = meta_summary
                metadata["meta_summary"] = {
                    "bullet_points": len(meta_summary.get("bullet_points", [])),
                }
            except Exception as exc:  # pragma: no cover - defensive logging
                timings["synthesis"] = time.perf_counter() - synthesis_start
                synthesis_status = "error"
                error_message = str(exc)
                component_errors["synthesis"] = error_message
                logger.error("Meta-summary generation failed: %s", exc)
                results.setdefault("errors", []).append(f"synthesis: {error_message}")
                metadata["meta_summary"] = {"status": "error", "message": error_message}

        if synthesis_status == "disabled":
            metadata.setdefault("meta_summary", {"status": "disabled"})
        elif synthesis_status == "skipped":
            metadata.setdefault("meta_summary", {"status": "skipped"})

        if ddi_status == "pending":
            primary_drug, secondary_drugs = self._infer_drug_targets(query, options)
            if primary_drug:
                metadata["ddi_request"] = {
                    "primary_drug": primary_drug,
                    "secondary_drugs": secondary_drugs,
                    "status": "pending",
                }
                ddi_start = time.perf_counter()
                try:
                    ddi_report = await asyncio.to_thread(
                        self.ddi_processor.analyze_drug_interactions,
                        papers,
                        primary_drug,
                        secondary_drugs,
                    )
                    timings["ddi_pk"] = time.perf_counter() - ddi_start
                    ddi_status = "completed"
                    metadata["ddi_request"]["status"] = "completed"
                    results["ddi_analysis"] = ddi_report
                except Exception as exc:  # pragma: no cover - defensive logging
                    timings["ddi_pk"] = time.perf_counter() - ddi_start
                    ddi_status = "error"
                    error_message = str(exc)
                    component_errors["ddi"] = error_message
                    logger.error("DDI/PK analysis failed: %s", exc)
                    metadata["ddi_request"]["status"] = "error"
                    metadata["ddi_request"]["message"] = error_message
                    results.setdefault("errors", []).append(f"ddi: {error_message}")
            else:
                ddi_status = "skipped"
                metadata["ddi_request"] = {"status": "skipped", "reason": "no_drug_entities"}
        elif ddi_status == "disabled":
            metadata["ddi_request"] = {"status": "disabled"}
        else:  # not_applicable
            metadata["ddi_request"] = {"status": "not_applicable"}

        metadata["synthesis_status"] = synthesis_status
        metadata["ddi_status"] = ddi_status
        if component_errors:
            metadata["component_errors"] = component_errors

        timings["total"] = time.perf_counter() - analysis_start
        metadata["timings"] = timings

        considered_statuses = [
            status
            for status in (synthesis_status, ddi_status)
            if status not in {"disabled", "not_applicable"}
        ]
        has_error = any(status == "error" for status in considered_statuses)
        has_completed = any(status == "completed" for status in considered_statuses)
        if has_error and has_completed:
            overall_status = "completed_with_errors"
        elif has_error:
            overall_status = "error"
        elif has_completed:
            overall_status = "completed"
        else:
            overall_status = "skipped"

        results["status"] = overall_status
        return results

    def _determine_analysis_type(self, query: str) -> str:
        """Classify the query to decide which analysis pipeline to emphasise."""
        primary_drug, secondary_drugs = self._infer_drug_targets(query, {})
        if primary_drug and secondary_drugs:
            return "drug_interaction"
        lowered = (query or "").lower()
        if any(keyword in lowered for keyword in ["interaction", "contraindicated", "combination", "co-admin"]):
            return "drug_interaction"
        if any(keyword in lowered for keyword in ["pharmacokinetic", "auc", "cmax", "clearance", "half-life"]):
            return "pharmacokinetic"
        if any(keyword in lowered for keyword in ["compare", "vs", "difference", "efficacy"]):
            return "comparative"
        return "general_research"

    def _infer_drug_targets(self, query: str, options: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """Infer primary and secondary drug entities from options or query heuristics.

        Uses pharmaceutical entity extractor when available, falling back to
        heuristics if extraction fails or no extractor is available.
        """
        primary_option = options.get("primary_drug")
        secondary_option = options.get("secondary_drugs") or []
        if primary_option:
            secondary_list = secondary_option if isinstance(secondary_option, list) else [secondary_option]
            return str(primary_option), [str(drug) for drug in secondary_list if drug]

        # Try pharmaceutical entity extraction if DDI processor is available
        if self.ddi_processor and hasattr(self.ddi_processor, "extract_drug_entities"):
            try:
                entities = self.ddi_processor.extract_drug_entities(query)
                if entities and len(entities) >= 1:
                    primary_drug = entities[0]
                    secondary_drugs = entities[1:] if len(entities) > 1 else []
                    logger.debug("Extracted drug entities: primary=%s, secondary=%s", primary_drug, secondary_drugs)
                    return primary_drug, secondary_drugs
            except Exception as extraction_error:
                logger.debug("Drug entity extraction failed, falling back to heuristics: %s", extraction_error)

        # Fallback to basic heuristic: split on keywords to detect drug pairs
        lowered = (query or "").lower()
        delimiters = [" and ", " with ", " vs ", " versus ", " + ", " interaction", " combined with "]
        primary_candidate: Optional[str] = None
        secondary_candidates: List[str] = []
        for delimiter in delimiters:
            if delimiter in lowered:
                parts = lowered.split(delimiter)
                if len(parts) >= 2:
                    primary_candidate = parts[0].strip().split()[-1]
                    secondary_candidates = [segment.strip().split()[0] for segment in parts[1].split(",") if segment.strip()]
                    break

        logger.debug("Heuristic drug inference: primary=%s, secondary=%s", primary_candidate, secondary_candidates)
        return primary_candidate, secondary_candidates

    def _package_enhanced_response(
        self,
        base_response: Dict[str, Any],
        analysis_results: Dict[str, Any],
        safety_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine base response, analysis, and safety metadata into final payload."""
        payload: Dict[str, Any] = {
            "answer": base_response.get("answer"),
            "sources": base_response.get("sources", []),
            "confidence_scores": base_response.get("confidence_scores"),
            "query": base_response.get("query"),
            "sanitized_query": base_response.get("sanitized_query", base_response.get("query")),
            "processing_time": base_response.get("processing_time"),
            "disclaimer": base_response.get("disclaimer"),
            "needs_rebuild": base_response.get("needs_rebuild", False),
            "safety": safety_metadata,
            "analysis": analysis_results or {},
        }

        if base_response.get("safety_warnings"):
            payload.setdefault("safety", {}).setdefault("output_validation", {}).setdefault(
                "warnings",
                base_response["safety_warnings"],
            )

        # NOTE: This method is only invoked from the async ask_question_safe flow, which already serialises access
        # before leaving the event loop. Metrics are read without awaiting the lock; external callers should prefer
        # get_safety_metrics()/get_safety_metrics_async, which provide locked snapshots.
        payload["metrics"] = {
            "total_queries": self.safety_metrics.get("total_queries"),
            "blocked_queries": self.safety_metrics.get("blocked_queries"),
            "analysis_failures": self.safety_metrics.get("analysis_failures"),
            "input_validation_failures": self.safety_metrics.get("input_validation_failures"),
            "output_validation_failures": self.safety_metrics.get("output_validation_failures"),
            "retrieval_validation_failures": self.safety_metrics.get("retrieval_validation_failures"),
            "last_processing_time": self.safety_metrics.get("last_processing_time"),
            "safety_mode": self.safety_metrics.get("safety_mode"),
            "pii_blocks": self.safety_metrics.get("pii_blocks"),
            "jailbreak_detections": self.safety_metrics.get("jailbreak_detections"),
            "disclaimer_enforcements": self.safety_metrics.get("disclaimer_enforcements"),
            "rails_usage": self.safety_metrics.get("rails_usage"),
            "guardrails_enabled": self.safety_metrics.get("guardrails_enabled"),
            "guardrails_active": self.safety_metrics.get("guardrails_active"),
            "last_stage_timings": self.safety_metrics.get("last_stage_timings"),
        }
        payload["component_health"] = self.component_health
        return payload

    def _serialize_base_response(self, response: Union[RAGResponse, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert the base RAGResponse or dict into a JSON-serialisable dictionary.

        Maps common fields from document metadata to top-level keys to ensure
        compatibility with medical analysis modules.
        """
        # Handle dict responses
        if isinstance(response, dict):
            # Transform dict to expected structure
            sources = response.get('sources', [])
            # Normalize sources if they're not in the expected format
            normalized_sources = []
            for idx, doc in enumerate(sources):
                if isinstance(doc, Document):
                    page_content = doc.page_content
                    metadata = dict(doc.metadata or {})
                elif isinstance(doc, dict):
                    page_content = doc.get("page_content", "")
                    metadata = dict(doc.get("metadata", {}) or {})
                else:
                    # Unknown format, create minimal entry
                    page_content = getattr(doc, "page_content", "")
                    metadata = getattr(doc, "metadata", {}) or {}

                source_entry = {
                    "id": metadata.get("id") or metadata.get("document_id") or f"doc-{idx}",
                    "page_content": page_content,
                    "metadata": metadata,
                }

                # Map common fields from metadata to top-level
                for field in ["pmid", "title", "url", "doi", "authors", "journal", "publication_date"]:
                    if field in metadata:
                        source_entry[field] = metadata[field]

                normalized_sources.append(source_entry)

            return {
                "answer": response.get("answer"),
                "sources": normalized_sources,
                "confidence_scores": response.get("confidence_scores"),
                "query": response.get("query"),
                "processing_time": response.get("processing_time"),
                "disclaimer": response.get("disclaimer"),
                "needs_rebuild": response.get("needs_rebuild", False),
            }

        # Handle RAGResponse objects (existing behavior)
        sources_payload: List[Dict[str, Any]] = []
        for idx, doc in enumerate(response.source_documents or []):
            if isinstance(doc, Document):
                page_content = doc.page_content
                metadata = dict(doc.metadata or {})
            else:
                page_content = getattr(doc, "page_content", "")
                metadata = dict(getattr(doc, "metadata", {}) or {})

            # Create base source entry with required schema
            source_entry = {
                "id": metadata.get("id") or metadata.get("document_id") or f"doc-{idx}",
                "page_content": page_content,
                "metadata": metadata,
            }

            # Map common fields from metadata to top-level for medical module compatibility
            # These fields may be expected by guardrails and analysis modules
            if "pmid" in metadata:
                source_entry["pmid"] = metadata["pmid"]
            if "title" in metadata:
                source_entry["title"] = metadata["title"]
            if "url" in metadata:
                source_entry["url"] = metadata["url"]
            if "doi" in metadata:
                source_entry["doi"] = metadata["doi"]
            if "authors" in metadata:
                source_entry["authors"] = metadata["authors"]
            if "journal" in metadata:
                source_entry["journal"] = metadata["journal"]
            if "publication_date" in metadata:
                source_entry["publication_date"] = metadata["publication_date"]

            sources_payload.append(source_entry)

        return {
            "answer": response.answer,
            "sources": sources_payload,
            "confidence_scores": response.confidence_scores,
            "query": response.query,
            "processing_time": response.processing_time,
            "disclaimer": response.disclaimer,
            "needs_rebuild": response.needs_rebuild,
        }

    def get_safety_metrics(self) -> Dict[str, Any]:
        """Expose aggregated safety metrics via the locked async accessor."""
        return self._run_coroutine_blocking(self.get_safety_metrics_async())

    async def get_safety_metrics_async(self) -> Dict[str, Any]:
        """Async version of get_safety_metrics with proper locking."""
        self._ensure_components_initialized()
        self._ensure_pubmed_components()

        def _read_metrics():
            with self._metrics_lock:
                metrics = dict(self.safety_metrics)
                metrics["component_health"] = dict(self.component_health)
                metrics["pubmed"] = dict(self.pubmed_metrics)
            return metrics

        return await asyncio.to_thread(_read_metrics)

    def get_system_status(self) -> Dict[str, Any]:
        """Return consolidated status for Streamlit dashboards and monitoring."""
        self._ensure_components_initialized()
        self._ensure_pubmed_components()

        # Use PubMedIntegrationManager for component health
        pubmed_health = self._pubmed_integration.get_pubmed_health()

        with self._metrics_lock:
            safety_snapshot = dict(self.safety_metrics)
            pubmed_snapshot = dict(self.pubmed_metrics)
            component_health = dict(self.component_health)

        return {
            "timestamp": time.time(),
            "safety": safety_snapshot,
            "component_health": component_health,
            "pubmed": {
                "enabled": self.config.should_enable_pubmed(),
                "metrics": pubmed_snapshot,
                "config": self.config.summarize_flags(),
                "warnings": list(self.config.warnings),
                "errors": list(self.config.errors),
                "component_health": pubmed_health,
            },
        }

    def _handle_guardrails_exception(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured response when guardrails block a query."""
        # Track guardrails usage
        with self._metrics_lock:
            self.safety_metrics["guardrails_used"] = True

        validation = context.get("validation", {})
        stage = context.get("stage", "unknown")
        issues = validation.get("issues") or ["Safety policy violation detected"]
        recommendations = validation.get("recommendations") or []
        sanitized_query = validation.get("sanitized_query")

        # Build the base response
        response = {
            "answer": None,
            "error": {
                "type": "guardrails_block",
                "message": "; ".join(issues),
                "details": {
                    "stage": stage,
                    "issues": issues,
                    "recommendations": recommendations,
                },
                "stage": stage,
            },
            "query": context.get("query"),
            "sanitized_query": sanitized_query,
            "disclaimer": self.base_agent.get_disclaimer(),
            "safety": {
                "input_validation": validation if stage == "input" else None,
                "output_validation": validation if stage == "output" else None,
                "retrieval_validation": context.get("base_payload", {}).get("retrieval_validation"),
                "analysis": context.get("base_payload", {}).get("analysis", {"status": "skipped"}),
                "timings": context.get("base_payload", {}).get("timings", {}),
            },
            "needs_rebuild": bool(context.get("base_payload", {}).get("needs_rebuild")),
        }

        # Add analysis field with skipped status
        response["analysis"] = {"status": "skipped"}

        # Add complete metrics
        response["metrics"] = {
            "total_queries": self.safety_metrics.get("total_queries", 0),
            "blocked_queries": self.safety_metrics.get("blocked_queries", 0),
            "input_validation_failures": self.safety_metrics.get("input_validation_failures", 0),
            "output_validation_failures": self.safety_metrics.get("output_validation_failures", 0),
            "analysis_failures": self.safety_metrics.get("analysis_failures", 0),
            "last_processing_time": self.safety_metrics.get("last_processing_time", 0.0),
            "safety_mode": self.safety_metrics.get("safety_mode", "balanced"),
            "guardrails_enabled": self.safety_metrics.get("guardrails_enabled", False),
            "guardrails_configured": self.safety_metrics.get("guardrails_configured", False),
            "guardrails_used": self.safety_metrics.get("guardrails_used", False),
            "pii_blocks": self.safety_metrics.get("pii_blocks", 0),
            "jailbreak_detections": self.safety_metrics.get("jailbreak_detections", 0),
            "disclaimer_enforcements": self.safety_metrics.get("disclaimer_enforcements", 0),
            "rails_usage": self.safety_metrics.get("rails_usage", {"input": False, "output": False}),
            "last_stage_timings": self.safety_metrics.get("last_stage_timings", {}),
        }

        # Add component health
        response["component_health"] = self.component_health

        return response
