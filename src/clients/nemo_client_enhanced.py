"""
Enhanced NeMo Retriever Client with Cloud-First Integration

Integrates OpenAI SDK wrapper with existing NeMo architecture using composition pattern.
Provides seamless fallback mechanisms while preserving pharmaceutical optimizations.

Architecture:
- Composition over inheritance for clean separation
- Cloud-first with intelligent fallback to self-hosted NeMo
- Pharmaceutical domain optimization maintained
- Comprehensive error handling and monitoring

Integration Points:
- OpenAI SDK for NVIDIA Build cloud endpoints
- Existing NeMo client for self-hosted fallback
- Enhanced configuration for decision logic
- Performance monitoring and cost tracking
"""
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests  # noqa: F401

# Initialize logger early for use in imports
logger = logging.getLogger(__name__)

# Local imports
try:
    from .openai_wrapper import NVIDIABuildConfig, NVIDIABuildError, OpenAIWrapper

    OPENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenAI SDK not available: {e}")
    OpenAIWrapper = None
    NVIDIABuildConfig = None
    NVIDIABuildError = None
    OPENAI_AVAILABLE = False

from .model_normalization import normalize_model as _normalize_shared_model
from .ollama_client import OllamaClient, OllamaConfig

try:
    from ..enhanced_config import EnhancedRAGConfig
except ImportError:
    from src.enhanced_config import EnhancedRAGConfig

if TYPE_CHECKING:
    # Only for type checking to avoid circular imports at runtime
    try:
        from ..nemo_retriever_client import NeMoRetrieverClient  # pragma: no cover
    except Exception:  # pragma: no cover
        from src.nemo_retriever_client import NeMoRetrieverClient  # type: ignore


def _to_dict_safe(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(obj, dict):
        return obj
    try:
        return obj.__dict__
    except Exception:
        return {"repr": repr(obj)}


class EndpointType(Enum):
    """Endpoint types for fallback strategy."""

    CLOUD = "cloud"
    SELF_HOSTED = "self_hosted"


@dataclass
class ClientResponse:
    """Standardized response wrapper for both cloud and self-hosted clients."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    endpoint_type: Optional[EndpointType] = None
    endpoint_used: Optional[str] = None
    model_used: Optional[str] = None
    response_time_ms: Optional[int] = None
    cost_tier: Optional[str] = None  # "free_tier" or "infrastructure"


class EnhancedNeMoClient:
    """
    Enhanced NeMo client with cloud-first architecture and seamless fallback.

    Uses composition pattern to integrate OpenAI SDK wrapper with existing
    NeMo client, providing pharmaceutical optimizations and cost monitoring.
    """

    def __init__(
        self,
        config: Optional[EnhancedRAGConfig] = None,
        enable_fallback: bool = True,
        pharmaceutical_optimized: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize enhanced NeMo client with cloud-first configuration.

        Args:
            config: Enhanced RAG configuration (auto-loaded if None)
            enable_fallback: Enable automatic fallback to self-hosted
            pharmaceutical_optimized: Enable pharmaceutical domain optimizations
            api_key: NVIDIA API key (takes precedence over environment variable)
        """
        # Load configuration
        self.config = config or EnhancedRAGConfig.from_env()
        self.enable_fallback = enable_fallback
        self.pharmaceutical_optimized = pharmaceutical_optimized

        # Store explicit API key for precedence over environment variable
        self._explicit_api_key = api_key

        # Initialize clients
        self.cloud_client: Optional[OpenAIWrapper] = None
        self.ollama_client: Optional[OllamaClient] = None
        self.nemo_client: Optional["NeMoRetrieverClient"] = None

        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "cloud_requests": 0,
            "ollama_requests": 0,
            "fallback_requests": 0,
            "self_hosted_requests": 0,
            "failed_requests": 0,
            "total_response_time_ms": 0.0,
        }

        # Background event loop for safe coroutine execution when already inside a loop
        self._bg_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_loop_thread: Optional[threading.Thread] = None

        # Pharma-centric metrics and simple budgeting counters
        self.pharma_metrics: Dict[str, Any] = {
            "query_types": {"drug_interaction": 0, "pharmacokinetics": 0, "clinical_trial": 0, "general": 0},
            "quality_failures": 0,
            "cache_hits": 0,
        }

        # Lightweight in-memory cache for common queries
        self._cache_max_entries = 256
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

        # Validate pharmaceutical configuration for compliance
        self._validate_pharmaceutical_config()

        # Initialize clients based on configuration
        self._initialize_clients()

        # Validate configuration compatibility
        if self.config.get_cloud_first_strategy()["cloud_first_enabled"] and not OPENAI_AVAILABLE:
            logger.warning(
                "Cloud-first strategy requested but OpenAI SDK not available. "
                "Only NeMo/Ollama endpoints will be used. Install OpenAI SDK with: pip install openai"
            )

        # Log configuration decisions
        self.config.log_configuration_decisions()

    # ---------------------------- Async helpers ----------------------------
    def _ensure_bg_loop(self) -> None:
        if self._bg_loop is not None and self._bg_loop_thread is not None and self._bg_loop_thread.is_alive():
            return
        self._bg_loop = asyncio.new_event_loop()

        def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._bg_loop_thread = threading.Thread(target=_run_loop, args=(self._bg_loop,), daemon=True)
        self._bg_loop_thread.start()

    def run_coro_sync(self, coro: "asyncio.Future[Any]") -> Any:
        try:
            # Detect if there's an active event loop in this thread
            asyncio.get_running_loop()
            # Prefer background loop to avoid nest_asyncio quirks
            self._ensure_bg_loop()
            fut = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)  # type: ignore[arg-type]
            return fut.result()
        except RuntimeError:
            # No running loop, safe to run directly
            return asyncio.run(coro)

    # ---------------------------- Pharma utils ----------------------------
    def _classify_pharma_query(self, text: str) -> str:
        """Classify pharmaceutical query using centralized classifier."""
        try:
            from ..utils.pharma import classify_pharma_query_safe

            return classify_pharma_query_safe(text)
        except ImportError:
            try:
                from src.utils.pharma import classify_pharma_query_safe

                return classify_pharma_query_safe(text)
            except ImportError:
                # Log warning once and default to general classification
                if not hasattr(self, "_pharma_import_warned"):
                    logger.warning("Pharma query classification utilities unavailable, defaulting to 'general'")
                    self._pharma_import_warned = True
                return "general"

    # ---------------------------- Cache utils ----------------------------
    def _generate_secure_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        query_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate collision-resistant cache key using SHA256 with full conversation context.

        This replaces the vulnerable cache key generation that used truncated content
        and weak 8-character MD5 hashes, which could cause dangerous cache collisions
        in pharmaceutical conversations.

        Key Components:
        1. Full conversation hash (all messages, roles, content, sequence)
        2. Parameter hash (model, temperature, max_tokens, query_type)
        3. Configuration hash (pharmaceutical flags, compliance settings)
        4. Timestamp bucket (for cache expiry control)

        Args:
            messages: List of conversation messages with role/content structure
            model: Model name used for completion
            temperature: Temperature parameter for completion
            max_tokens: Max tokens parameter for completion
            query_type: Pharmaceutical query classification
            conversation_id: Optional conversation identifier
            **kwargs: Additional parameters

        Returns:
            Collision-resistant cache key string with 'pharma_chat::' prefix
        """
        try:
            # 1. Full conversation context hash
            conversation_data = {
                "messages": [
                    {"role": msg.get("role", ""), "content": str(msg.get("content", "")), "sequence": idx}
                    for idx, msg in enumerate(messages or [])
                ],
                "conversation_id": conversation_id or "single_turn",
                "message_count": len(messages or []),
                "has_system_prompt": any(m.get("role") == "system" for m in (messages or [])),
                "user_message_count": sum(1 for m in (messages or []) if m.get("role") == "user"),
                "assistant_message_count": sum(1 for m in (messages or []) if m.get("role") == "assistant"),
            }

            conversation_json = json.dumps(conversation_data, sort_keys=True, separators=(",", ":"))
            conversation_hash = hashlib.sha256(conversation_json.encode("utf-8")).hexdigest()

            # 2. Parameters hash
            params_data = {
                "model": model or "",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "query_type": query_type or "general",
                "additional_params": sorted(kwargs.items()) if kwargs else [],
            }
            params_json = json.dumps(params_data, sort_keys=True, separators=(",", ":"))
            params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()[:16]

            # 3. Pharmaceutical configuration hash
            config_data = {
                "disclaimer": bool(getattr(self.config, "pharma_require_disclaimer", False)),
                "compliance": bool(getattr(self.config, "pharma_compliance_mode", False)),
                "workflows": bool(getattr(self.config, "pharma_workflow_templates_enabled", False)),
                "qa_enabled": bool(getattr(self.config, "pharma_quality_assurance_enabled", False)),
                "region": getattr(self.config, "pharma_region", "US"),
                "pharmaceutical_optimized": bool(self.pharmaceutical_optimized),
            }
            config_json = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
            config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]

            # 4. Time bucket for cache expiry (1-hour buckets for pharmaceutical safety)
            time_bucket = str(int(time.time() // 3600))

            # Combine into collision-resistant key
            final_key_data = f"{conversation_hash}::{params_hash}::{config_hash}::{time_bucket}"
            final_key = hashlib.sha256(final_key_data.encode("utf-8")).hexdigest()

            # Return with pharmaceutical prefix for identification
            return f"pharma_chat::{final_key[:32]}"

        except Exception as e:
            logger.warning(f"Error generating secure cache key: {e}, falling back to simple key")
            # Fallback to a simple but unique key
            simple_data = f"{str(messages)}::{model}::{temperature}::{max_tokens}::{query_type}::{time.time()}"
            simple_hash = hashlib.sha256(simple_data.encode("utf-8")).hexdigest()
            return f"pharma_fallback::{simple_hash[:32]}"

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get item from cache with hit tracking."""
        if cache_key in self._cache:
            # Move to end (LRU behavior)
            value = self._cache.pop(cache_key)
            self._cache[cache_key] = value
            self.pharma_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for key: {cache_key[:20]}...")
            return value
        return None

    def _set_to_cache(self, cache_key: str, value: Any) -> None:
        """Set item in cache with size management."""
        # Remove oldest entries if cache is full
        while len(self._cache) >= self._cache_max_entries:
            self._cache.popitem(last=False)  # Remove oldest (FIFO)

        self._cache[cache_key] = value
        logger.debug(f"Cached result for key: {cache_key[:20]}...")

    def _extract_pharmaceutical_safety_markers(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract pharmaceutical safety markers from conversation messages."""
        safety_markers = {
            "contains_drug_names": False,
            "contains_dosage_info": False,
            "contains_interaction_warnings": False,
            "contains_medical_disclaimers": False,
            "high_risk_content": False,
        }

        try:
            full_text = " ".join(str(msg.get("content", "")) for msg in (messages or []))
            text_lower = full_text.lower()

            # Basic pharmaceutical content detection
            drug_keywords = ["medication", "drug", "prescription", "dosage", "mg", "tablet", "pill"]
            interaction_keywords = ["interaction", "contraindication", "side effect", "adverse", "warning"]
            disclaimer_keywords = ["consult", "physician", "doctor", "medical advice", "disclaimer"]

            safety_markers["contains_drug_names"] = any(keyword in text_lower for keyword in drug_keywords)
            safety_markers["contains_interaction_warnings"] = any(
                keyword in text_lower for keyword in interaction_keywords
            )
            safety_markers["contains_medical_disclaimers"] = any(
                keyword in text_lower for keyword in disclaimer_keywords
            )
            safety_markers["high_risk_content"] = (
                safety_markers["contains_drug_names"] and safety_markers["contains_interaction_warnings"]
            )

        except Exception as e:
            logger.debug(f"Error extracting pharmaceutical safety markers: {e}")

        return safety_markers

    def _build_workflow_prompt(self, query_type: str) -> str:
        if query_type == "drug_interaction":
            return (
                "You are a clinical pharmacology assistant. Focus on known and plausible drug-drug interactions, "
                "mechanisms (CYP enzymes, transporters), contraindications, and monitoring guidance. Be precise and conservative."
            )
        if query_type == "pharmacokinetics":
            return (
                "You are a PK specialist. Summarize half-life, clearance, protein binding, Cmax/Tmax, and key factors "
                "(renal/hepatic impairment, genetic polymorphisms). Use concise numbers where possible."
            )
        if query_type == "clinical_trial":
            return (
                "You are a clinical trials analyst. Summarize study design, phase, arms, endpoints, and key outcomes. "
                "Note limitations and population details where relevant."
            )
        return "You are a medical research assistant. Provide concise, accurate, and source-aware responses."

    def _validate_medical_terminology(self, text: str, query_type: str) -> bool:
        if not text:
            return False
        t = text.lower()
        if query_type == "drug_interaction":
            return any(
                k in t for k in ["interaction", "contraindication", "monitor", "cyp", "p-gp", "transport"]
            )  # simple heuristic
        if query_type == "pharmacokinetics":
            return any(k in t for k in ["half-life", "clearance", "cmax", "tmax", "auc"])  # heuristic
        if query_type == "clinical_trial":
            return any(k in t for k in ["phase", "endpoint", "random", "placebo", "cohort", "arm"])  # heuristic
        return True

    def _cache_key(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        query_type: Optional[str] = None,
        endpoint_label: Optional[str] = None,
    ) -> str:
        """
        Generate comprehensive cache key including all parameters that affect response.

        Args:
            messages: Chat messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            query_type: Query type classification

        Returns:
            Cache key string incorporating all relevant parameters
        """
        # Extract user content (existing logic)
        last_user = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break

        # Hash system prompt content if present
        system_prompt_hash = ""
        for m in messages or []:
            if m.get("role") == "system":
                import hashlib

                system_prompt_hash = hashlib.md5(str(m.get("content", "")).encode()).hexdigest()[:8]
                break

        # Include pharma flags to avoid stale reuse across compliance settings
        try:
            disclaimer = bool(getattr(self.config, "pharma_require_disclaimer", False))
            compliance = bool(getattr(self.config, "pharma_compliance_mode", False))
            workflows = bool(getattr(self.config, "pharma_workflow_templates_enabled", False))
        except Exception:
            disclaimer = compliance = workflows = False

        # Build comprehensive cache key
        key_parts = [
            (model or "").strip(),
            str(temperature or "auto"),
            str(max_tokens or "auto"),
            system_prompt_hash,
            query_type or "general",
            (endpoint_label or "auto"),
            f"disc={int(disclaimer)}",
            f"comp={int(compliance)}",
            f"wf={int(workflows)}",
            last_user.strip()[:400],  # Reduced to fit other params
        ]
        return "::".join(key_parts)

    def _cache_get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.pharma_metrics["cache_hits"] += 1
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_max_entries:
            self._cache.popitem(last=False)

    def _normalize_model_name(self, model: str, prefer_full_name: bool = True) -> str:
        """
        Normalize model identifiers to consistent format.

        Args:
            model: Model name to normalize
            prefer_full_name: If True, return full names (e.g., "nvidia/nv-embedqa-e5-v5")
                             If False, return short names (e.g., "nv-embedqa-e5-v5")

        Returns:
            Normalized model name
        """
        return _normalize_shared_model(model, prefer_full_name)

    def _get_effective_api_key(self) -> Optional[str]:
        """
        Get effective API key with explicit preference over environment variable.

        PHARMACEUTICAL SECURITY: Implements secure API key resolution for pharmaceutical
        research environments. Explicit keys take precedence to support programmatic
        authentication in research pipelines while maintaining environment fallback
        for operational flexibility.

        AUDIT TRAIL: Key resolution logic ensures pharmaceutical applications can
        track authentication sources for compliance. No actual key values are logged
        to maintain security in pharmaceutical research environments.

        Returns:
            API key with precedence: explicit key -> environment variable -> None
            None indicates no authentication available for cloud services.

        Security Note:
            Empty string is treated as None for clean environment variable fallback.
            This prevents accidental authentication bypass in pharmaceutical systems.
        """
        # Handle empty string as None for clean fallback
        explicit_key = self._explicit_api_key
        if explicit_key == "":
            explicit_key = None

        # Prefer explicit key, fall back to environment
        return explicit_key or os.getenv("NVIDIA_API_KEY")

    def _validate_pharmaceutical_config(self) -> None:
        """
        Validate pharmaceutical-specific configuration for regulatory compliance.

        PHARMACEUTICAL COMPLIANCE: Ensures configuration settings meet pharmaceutical
        research requirements for batch processing, API usage, and safety controls.

        AUDIT TRAIL: Logs configuration warnings for pharmaceutical audit requirements.
        Critical configuration errors raise PharmaceuticalConfigError for immediate
        attention in medical research environments.

        Raises:
            PharmaceuticalConfigError: For critical pharmaceutical configuration issues
        """
        # Import pharmaceutical error types
        try:
            from .openai_wrapper import PharmaceuticalConfigError
        except ImportError:
            # Define local error type if import fails
            class PharmaceuticalConfigError(Exception):
                pass

        # Validate pharmaceutical batch settings
        batch_size = getattr(self.config, "pharma_batch_max_size", 16)
        batch_latency = getattr(self.config, "pharma_batch_max_latency_ms", 500)

        if batch_size > 100:
            logger.warning(
                "Large pharmaceutical batch size (%d) may impact drug interaction query accuracy. "
                "Consider reducing to ≤100 for optimal pharmaceutical research precision.",
                batch_size,
            )

        if batch_latency > 2000:
            logger.warning(
                "High pharmaceutical batch latency (%dms) may delay safety-critical queries. "
                "Consider reducing to ≤2000ms for responsive pharmaceutical research.",
                batch_latency,
            )

        # Validate pharmaceutical safety settings
        require_disclaimer = getattr(self.config, "pharma_require_disclaimer", True)
        compliance_mode = getattr(self.config, "pharma_compliance_mode", True)

        if not require_disclaimer:
            logger.warning(
                "Medical disclaimer disabled in pharmaceutical configuration. "
                "Ensure regulatory compliance for medical research applications."
            )

        if not compliance_mode:
            logger.warning(
                "Pharmaceutical compliance mode disabled. "
                "This may impact regulatory audit requirements for medical research."
            )

        # Validate pharmaceutical cost controls
        budget_limit = getattr(self.config, "research_project_budget_limit_usd", 0.0)
        cost_tracking = getattr(self.config, "cost_per_query_tracking", True)

        if budget_limit > 0 and not cost_tracking:
            raise PharmaceuticalConfigError(
                "Pharmaceutical budget limit set but cost tracking disabled. "
                "Enable cost_per_query_tracking for pharmaceutical budget compliance."
            )

        logger.debug("Pharmaceutical configuration validation completed successfully")

    def _initialize_clients(self) -> None:
        """Initialize cloud and fallback clients based on configuration."""
        strategy = self.config.get_cloud_first_strategy()

        # Initialize cloud client if cloud-first is enabled AND OpenAI is available
        if strategy["cloud_first_enabled"]:
            if OPENAI_AVAILABLE:
                try:
                    # Budgeting configuration validation
                    research_project_budgeting = getattr(self.config, "research_project_budgeting", False)
                    budget_limit = getattr(self.config, "research_project_budget_limit_usd", 0.0)

                    if research_project_budgeting and not budget_limit:
                        logger.warning(
                            "Budgeting enabled but PHARMA_BUDGET_LIMIT_USD is 0.0; cost guards will be inert"
                        )

                    enabled_eit, eit_value = (False, None)
                    try:
                        enabled_eit, eit_value = self.config.get_embedding_input_type_config()
                    except Exception:
                        enabled_eit, eit_value = (
                            getattr(self.config, "enable_nvidia_build_embedding_input_type", True),
                            getattr(self.config, "nvidia_build_embedding_input_type", None),
                        )

                    # Determine batch optimization from feature flags (not budgeting)
                    try:
                        batch_opt_enabled = bool(
                            self.config.get_feature_flags().get("batch_optimization_enabled", True)
                        )
                    except Exception:
                        batch_opt_enabled = True

                    # Map EnhancedRAGConfig request policy knobs into NVIDIA Build config
                    backoff_base = getattr(self.config, "rerank_retry_backoff_base", 0.5)
                    jitter_enabled = getattr(self.config, "rerank_retry_jitter", True)
                    jitter_amp = float(backoff_base) if jitter_enabled else 0.0

                    endpoints = self.config.get_effective_endpoints()
                    nvidia_config = NVIDIABuildConfig(
                        api_key=self._get_effective_api_key(),
                        base_url=self.config.nvidia_build_base_url,
                        pharmaceutical_optimized=self.pharmaceutical_optimized,
                        embedding_input_type=(eit_value if enabled_eit else None),
                        research_project_id=getattr(self.config, "pharma_project_id", None),
                        research_project_budget_limit_usd=budget_limit,
                        enable_cost_per_query_tracking=getattr(self.config, "cost_per_query_tracking", True),
                        enable_batch_optimization=batch_opt_enabled,
                        prefer_embedding_model=getattr(self.config, "nvidia_build_embedding_model", None),
                        prefer_chat_model=getattr(self.config, "nvidia_build_llm_model", None),
                        prefer_responses_api=getattr(self.config, "prefer_responses_api", True),
                        batch_max_size=getattr(self.config, "pharma_batch_max_size", 16),
                        batch_max_latency_ms=getattr(self.config, "pharma_batch_max_latency_ms", 500),
                        # Rerank retry/backoff tuning
                        request_backoff_base=backoff_base,
                        request_backoff_jitter=jitter_amp,
                        rerank_retry_max_attempts=getattr(self.config, "rerank_retry_max_attempts", 3),
                        # Rerank ordering and endpoints
                        cloud_first_rerank_enabled=getattr(self.config, "enable_cloud_first_rerank", True),
                        build_rerank_enabled=True,
                        enable_rerank_model_mapping=False,
                        nemo_reranking_endpoint=endpoints.get("reranking"),
                    )
                    self.cloud_client = OpenAIWrapper(nvidia_config)
                    logger.info("Cloud client initialized successfully (NVIDIA Build)")

                except Exception as e:
                    logger.error(f"Cloud client initialization failed: {str(e)}")
                    if not self.enable_fallback:
                        raise
            else:
                logger.warning("Cloud-first enabled but OpenAI SDK unavailable. Continuing with NeMo/Ollama only.")
                self.cloud_client = None

        # Initialize NeMo self-hosted client for fallback
        if self.enable_fallback or not strategy["cloud_first_enabled"]:
            try:
                # Use effective API key with explicit preference over environment variable
                api_key = self._get_effective_api_key()
                # Local import to avoid circular dependency at module import time
                from ..nemo_retriever_client import NeMoRetrieverClient  # type: ignore

                self.nemo_client = NeMoRetrieverClient(api_key=api_key)
                logger.info("NeMo self-hosted client initialized successfully")

            except Exception as e:
                logger.error(f"NeMo client initialization failed: {str(e)}")
                if not strategy["cloud_first_enabled"]:
                    raise

        # Validate at least one client is available
        # Optional: initialize Ollama local client if enabled
        if getattr(self.config, "enable_ollama", False):
            try:
                ocfg = self.config.get_ollama_config()
                self.ollama_client = OllamaClient(
                    OllamaConfig(
                        base_url=ocfg["base_url"],
                        chat_model=ocfg["chat_model"],
                        embed_model=ocfg["embed_model"],
                        timeout_seconds=int(ocfg.get("timeout_seconds", 60)),
                    )
                )
                logger.info("Ollama local client initialized (%s)", ocfg["base_url"])
            except Exception as e:
                logger.warning("Ollama client initialization failed: %s", e)

        if not self.cloud_client and not self.ollama_client and not self.nemo_client:
            raise RuntimeError("No clients available - both cloud and self-hosted initialization failed")

    def _execute_with_fallback(
        self,
        operation: str,
        cloud_func: callable,
        nemo_func: callable,
        ollama_func: Optional[callable] = None,
        force_endpoint: Optional[EndpointType] = None,
        **kwargs,
    ) -> ClientResponse:
        """
        Execute operation with intelligent fallback logic respecting configured order.

        Args:
            operation: Operation name for logging
            cloud_func: Function to call on cloud client
            nemo_func: Function to call on NeMo client
            ollama_func: Function to call on Ollama client (optional)
            force_endpoint: Force specific endpoint (CLOUD or SELF_HOSTED), bypasses fallback logic
            **kwargs: Arguments to pass to functions

        Returns:
            ClientResponse with results and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1

        # Get fallback order from configuration or force specific endpoint
        if force_endpoint == EndpointType.CLOUD:
            fallback_order = ["nvidia_build"]
        elif force_endpoint == EndpointType.SELF_HOSTED:
            fallback_order = ["nemo", "ollama"]
        else:
            fallback_order = self.config.get_fallback_order()

        # Map endpoint names to client functions and availability
        endpoint_mappings = {
            "nvidia_build": (cloud_func, self.cloud_client, "cloud_requests", EndpointType.CLOUD, "free_tier"),
            "nemo": (nemo_func, self.nemo_client, "fallback_requests", EndpointType.SELF_HOSTED, "infrastructure"),
            "ollama": (ollama_func, self.ollama_client, "ollama_requests", EndpointType.SELF_HOSTED, "infrastructure"),
        }

        errors = []

        # Try endpoints in configured order
        for endpoint_name in fallback_order:
            endpoint_config = endpoint_mappings.get(endpoint_name)
            if not endpoint_config:
                logger.warning("Unknown endpoint '%s' in fallback order; skipping", endpoint_name)
                continue

            func, client, metrics_key, endpoint_type, cost_tier = endpoint_config

            # Skip if client not available or function not provided
            if not client or not func:
                continue

            # Skip if fallback disabled and this is not the primary
            if not self.enable_fallback and endpoint_name != fallback_order[0]:
                continue

            try:
                logger.debug(f"Attempting {operation} on {endpoint_name} endpoint")
                result = func(client, **kwargs)
                response_time = int((time.time() - start_time) * 1000)

                self.metrics[metrics_key] += 1
                # Also update self_hosted_requests for nemo and ollama endpoints
                if endpoint_name in ("nemo", "ollama"):
                    self.metrics["self_hosted_requests"] += 1
                self.metrics["total_response_time_ms"] += response_time

                logger.info(f"{operation} succeeded on {endpoint_name} endpoint ({response_time}ms)")
                return ClientResponse(
                    success=True,
                    data=result,
                    endpoint_type=endpoint_type,
                    response_time_ms=response_time,
                    cost_tier=cost_tier,
                )

            except Exception as e:
                error_msg = f"{operation} failed on {endpoint_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)

        # All endpoints failed
        self.metrics["failed_requests"] += 1
        combined_error = "; ".join(errors) if errors else "No clients available"
        return ClientResponse(
            success=False, error=combined_error, response_time_ms=int((time.time() - start_time) * 1000)
        )

    def create_embeddings(
        self, texts: List[str], model: Optional[str] = None, is_query: bool = False
    ) -> ClientResponse:
        """
        Create embeddings with cloud-first fallback.

        Args:
            texts: List of texts to embed
            model: Model name (auto-selected if None)

        Returns:
            ClientResponse with embedding results. For empty inputs, returns
            standardized response with empty embeddings list, zero usage counts,
            and selected model string to ensure consistent consumer expectations
            across all client implementations.
        """

        def _cloud_embedding(client: OpenAIWrapper, **kwargs) -> Any:
            texts_in = kwargs["texts"]
            model_in = kwargs.get("model")
            is_query_in = bool(kwargs.get("is_query", False))
            # Normalize to full model name for cloud calls
            normalized_model = self._normalize_model_name(
                model_in or client.PHARMA_MODELS["embedding"]["preferred"],
                prefer_full_name=True,
            )
            # Heuristic auto-detection when not explicitly set
            if not is_query_in:
                try:
                    if isinstance(texts_in, list) and texts_in:
                        avg_len = sum(len(t or "") for t in texts_in) / max(1, len(texts_in))
                        has_q = any(isinstance(t, str) and ("?" in t) for t in texts_in)
                        is_query_in = len(texts_in) <= 3 and avg_len < 200 and has_q
                except Exception:
                    pass
            # Respect config gate for input_type forwarding
            try:
                eit_enabled = bool(getattr(self.config, "enable_nvidia_build_embedding_input_type", True))
            except Exception:
                eit_enabled = True
            # Batch optimization
            enable_batch = False
            try:
                enable_batch = bool(self.config.get_feature_flags().get("batch_optimization_enabled", True))
            except Exception:
                enable_batch = True
            batch_threshold = getattr(self.config, "pharma_batch_max_size", 16) or 16
            if enable_batch and len(texts_in) > batch_threshold:
                batched = client.create_embeddings_batched(
                    texts_in,
                    model=normalized_model,
                    # If enabled, choose query/pass based on detection
                    input_type=("query" if (eit_enabled and is_query_in) else ("passage" if eit_enabled else None)),
                )
                # Sort by index and extract embeddings
                batched_sorted = sorted(batched, key=lambda x: x.get("index", 0))
                embeddings = [item["embedding"] for item in batched_sorted]
                dims = len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else None
                return {
                    "embeddings": embeddings,
                    "model": normalized_model,
                    "usage": None,
                    "dimensions": dims,
                }
            # Non-batched path
            response = client.create_embeddings(
                texts_in,
                model=normalized_model,
                input_type=("query" if (eit_enabled and is_query_in) else ("passage" if eit_enabled else None)),
                is_query=(is_query_in if eit_enabled else None),
            )
            embs = [data.embedding for data in response.data]
            dims = len(embs[0]) if embs and isinstance(embs[0], list) else None
            return {
                "embeddings": embs,
                "model": self._normalize_model_name(getattr(response, "model", None) or normalized_model),
                "usage": _to_dict_safe(getattr(response, "usage", None)) or None,
                "dimensions": dims,
            }

        def _ollama_embedding(client: OllamaClient, **kwargs) -> Any:
            embs = client.embed(kwargs["texts"], model=self.config.ollama_embed_model)
            return {"embeddings": embs, "model": self.config.ollama_embed_model or "nomic-embed-text"}

        def _nemo_embedding(client: "NeMoRetrieverClient", **kwargs) -> Any:
            async def _run():
                # Local import to avoid circular dependency
                from ..nemo_retriever_client import NeMoRetrieverClient  # type: ignore  # noqa: F401

                resp = await client.embed_texts(kwargs["texts"], model="nv-embedqa-e5-v5")
                if resp.success:
                    data = resp.data or {}
                    embs = data.get("embeddings") or data.get("data") or []
                    dims = len(embs[0]) if embs and isinstance(embs[0], list) else None
                    return {
                        "embeddings": embs,
                        "model": self._normalize_model_name("nvidia/nv-embedqa-e5-v5"),
                        "usage": None,
                        "dimensions": dims,
                    }
                raise RuntimeError(resp.error or "NeMo embedding failed")

            return self.run_coro_sync(_run())

        # Early return for empty inputs - standardized format matching OpenAI wrapper
        if not texts:
            # Select appropriate default model
            selected_model = model or (
                self.config.nvidia_build_embedding_model
                or "nvidia/nv-embedqa-e5-v5"  # Pharmaceutical-optimized default
            )

            return ClientResponse(
                success=True,
                data={
                    "embeddings": [],
                    "model": selected_model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "dimensions": 0,
                },
                response_time_ms=0,
                endpoint_used="empty_input",
                cost_tier="free",
            )

        # Prefer configured model when not explicitly provided
        if model is None:
            try:
                model = self.config.nvidia_build_embedding_model or model
            except Exception:
                pass

        return self._execute_with_fallback(
            "embedding_creation",
            _cloud_embedding,
            _nemo_embedding,
            _ollama_embedding if self.ollama_client else None,
            texts=texts,
            model=model,
            is_query=is_query,
        )

    async def embed_texts(self, texts: List[str], model: Optional[str] = None) -> "NeMoAPIResponse":
        """
        Legacy compatibility adapter for embed_texts method.

        This method provides backward compatibility for consumers expecting the
        legacy embed_texts interface. It internally calls create_embeddings and
        converts the response format to NeMoAPIResponse.

        Args:
            texts: List of texts to embed
            model: Embedding model (auto-selected if None)

        Returns:
            NeMoAPIResponse with legacy format compatibility
        """
        start_time = time.time()

        try:
            # Normalize model to full form for cloud paths when provided
            model_full = self._normalize_model_name(model, True) if model else None
            # Call the internal create_embeddings method
            client_response = self.create_embeddings(texts, model_full, is_query=True)
            response_time_ms = (time.time() - start_time) * 1000

            # Convert ClientResponse to NeMoAPIResponse format
            if client_response.success and client_response.data:
                # Extract embeddings data in expected format
                embeddings = client_response.data.get("embeddings", [])
                model_used = self._normalize_model_name(client_response.data.get("model", model))

                # Import NeMoAPIResponse here to avoid circular imports
                from ..nemo_retriever_client import NeMoAPIResponse

                return NeMoAPIResponse(
                    success=True,
                    data={
                        "embeddings": embeddings,
                        "model": model_used,
                        "dimensions": client_response.data.get("dimensions"),
                        "usage": client_response.data.get("usage"),
                    },
                    error=None,
                    response_time_ms=response_time_ms,
                    service="enhanced_nemo_client",
                    model=model_used,
                )
            else:
                # Handle error case
                from ..nemo_retriever_client import NeMoAPIResponse

                return NeMoAPIResponse(
                    success=False,
                    data=None,
                    error=client_response.error or "Embedding creation failed",
                    response_time_ms=response_time_ms,
                    service="enhanced_nemo_client",
                    model=model,
                )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"embed_texts adapter failed: {str(e)}")

            # Import NeMoAPIResponse here to avoid circular imports
            try:
                from ..nemo_retriever_client import NeMoAPIResponse
            except ImportError:
                from src.nemo_retriever_client import NeMoAPIResponse

            return NeMoAPIResponse(
                success=False,
                data=None,
                error=str(e),
                response_time_ms=response_time_ms,
                service="enhanced_nemo_client",
                model=model,
            )

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        force_endpoint: Optional[EndpointType] = None,
    ) -> ClientResponse:
        """
        Create chat completion with cloud-first fallback.

        Args:
            messages: List of message dictionaries
            model: Model name (auto-selected if None)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            force_endpoint: Force specific endpoint (CLOUD or SELF_HOSTED), bypasses fallback logic

        Returns:
            ClientResponse with chat completion results
        """
        # Query classification and routing
        user_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        query_type = self._classify_pharma_query(user_text) if self.pharmaceutical_optimized else "general"
        self.pharma_metrics["query_types"][query_type] += 1

        # Prefer pharma-specific model if configured
        if model is None:
            try:
                model = self.config.get_pharma_model_for_query_type(query_type)
            except Exception:
                pass

        # Add workflow system prompt for quality assurance
        if self.config.pharma_workflow_templates_enabled:
            sys_prompt = self._build_workflow_prompt(query_type)
            if not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": sys_prompt}] + list(messages)

        # Cache check
        endpoint_label = (
            "cloud"
            if force_endpoint == EndpointType.CLOUD
            else ("self_hosted" if force_endpoint == EndpointType.SELF_HOSTED else "auto")
        )
        cache_key = self._cache_key(messages, model, temperature, max_tokens, query_type, endpoint_label=endpoint_label)
        cached = self._cache_get(cache_key)
        if cached is not None:
            payload = cached.get("payload", cached) if isinstance(cached, dict) else cached
            ep_value = None
            if isinstance(cached, dict):
                ep_value = cached.get("endpoint_type")
            endpoint = None
            if isinstance(ep_value, str):
                endpoint = (
                    EndpointType.CLOUD
                    if ep_value == "cloud"
                    else (EndpointType.SELF_HOSTED if ep_value == "self_hosted" else None)
                )
            return ClientResponse(
                success=True,
                data=payload,
                endpoint_type=endpoint,
                model_used=model,
                response_time_ms=0,
                cost_tier=(cached.get("cost_tier") if isinstance(cached, dict) else None),
            )

        def _cloud_chat(client: OpenAIWrapper, **kwargs) -> Any:
            response = client.create_chat_completion(
                kwargs["messages"],
                model=kwargs.get("model"),
                max_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
            )
            content = response.choices[0].message.content
            data = {
                "content": content,
                "model": self._normalize_model_name(response.model),
                "usage": _to_dict_safe(getattr(response, "usage", None)),
                "quality_ok": (
                    self._validate_medical_terminology(content, query_type)
                    if self.config.pharma_quality_assurance_enabled
                    else True
                ),
                "query_type": query_type,
            }
            # Append or flag disclaimer for pharma compliance
            try:
                require_disclaimer = bool(getattr(self.config, "pharma_require_disclaimer", False))
            except Exception:
                require_disclaimer = False
            append_disclaimer = os.getenv("APPEND_DISCLAIMER_IN_ANSWER", "true").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if require_disclaimer:
                disclaimer_text = os.getenv(
                    "MEDICAL_DISCLAIMER", "This information is for educational purposes and not medical advice."
                )
                data["disclaimer_added"] = True
                data["disclaimer_text"] = disclaimer_text
                if append_disclaimer and isinstance(data.get("content"), str):
                    data["content"] = f"{data['content']}\n\n{disclaimer_text}"
            return data

        def _ollama_chat(client: OllamaClient, **kwargs) -> Any:
            data = client.chat(
                kwargs["messages"],
                model=kwargs.get("model") or self.config.ollama_chat_model,
                max_tokens=kwargs.get("max_tokens"),
                temperature=kwargs.get("temperature"),
            )
            content = (data.get("message", {}) or {}).get("content", "")
            payload = {
                "content": content,
                "model": self._normalize_model_name(
                    kwargs.get("model") or self.config.ollama_chat_model or "llama3.1:8b"
                ),
                "usage": None,
            }
            # Pharma disclaimer flag for local responses as well
            try:
                require_disclaimer = bool(getattr(self.config, "pharma_require_disclaimer", False))
            except Exception:
                require_disclaimer = False
            append_disclaimer = os.getenv("APPEND_DISCLAIMER_IN_ANSWER", "true").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if require_disclaimer:
                disclaimer_text = os.getenv(
                    "MEDICAL_DISCLAIMER", "This information is for educational purposes and not medical advice."
                )
                payload["disclaimer_added"] = True
                payload["disclaimer_text"] = disclaimer_text
                if append_disclaimer and isinstance(payload.get("content"), str):
                    payload["content"] = f"{payload['content']}\n\n{disclaimer_text}"
            return payload

        def _nemo_chat(client: "NeMoRetrieverClient", **kwargs) -> Any:
            # Local import to avoid circular dependency
            from ..nemo_retriever_client import NeMoRetrieverClient  # type: ignore  # noqa: F401

            # NeMo Retriever does not provide chat; raise to continue fallback chain
            raise RuntimeError("NeMo chat is not supported in Retriever client")

        response = self._execute_with_fallback(
            "chat_completion",
            _cloud_chat,
            _nemo_chat,
            _ollama_chat if self.ollama_client else None,
            force_endpoint=force_endpoint,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Post-process for QA and potential single retry for critical queries
        if response.success and isinstance(response.data, dict):
            data = response.data
            quality_ok = data.get("quality_ok", True)
            critical = query_type in {"drug_interaction", "pharmacokinetics", "clinical_trial"}
            if critical and self.config.pharma_quality_assurance_enabled and not quality_ok:
                self.pharma_metrics["quality_failures"] += 1
                try:
                    # One guided retry with more conservative settings
                    guided = list(messages)
                    guided.insert(
                        0,
                        {
                            "role": "system",
                            "content": "Ensure terminology appropriate to the query type and include key details.",
                        },
                    )
                    response = self._execute_with_fallback(
                        "chat_completion",
                        _cloud_chat,
                        _nemo_chat,
                        _ollama_chat if self.ollama_client else None,
                        force_endpoint=force_endpoint,
                        messages=guided,
                        model=model,
                        max_tokens=(max_tokens or 1000),
                        temperature=0.0,
                    )
                except Exception:
                    pass

        # Cache store on success (regenerate key with final parameter values)
        if response.success and isinstance(response.data, dict):
            final_cache_key = self._cache_key(
                messages, model, temperature, max_tokens, query_type, endpoint_label=endpoint_label
            )
            self._cache_put(
                final_cache_key,
                {
                    "payload": response.data,
                    "endpoint_type": response.endpoint_type.value if response.endpoint_type else None,
                    "cost_tier": response.cost_tier,
                },
            )

        return response

    def test_pharmaceutical_capabilities(self) -> Dict[str, Any]:
        """
        Test pharmaceutical research capabilities across endpoints.

        Returns:
            Comprehensive test results for pharmaceutical use cases
        """
        results = {
            "pharmaceutical_optimized": self.pharmaceutical_optimized,
            "cloud_test": None,
            "nemo_test": None,
            "embedding_test": None,
            "chat_test": None,
            "overall_status": "unknown",
        }

        # Test cloud client if available
        if self.cloud_client:
            try:
                cloud_validation = self.cloud_client.validate_pharmaceutical_setup()
                results["cloud_test"] = cloud_validation
            except Exception as e:
                results["cloud_test"] = {"success": False, "error": str(e)}

        # Test NeMo client if available
        if self.nemo_client:
            try:
                # Basic health check for NeMo
                results["nemo_test"] = {"success": True, "message": "NeMo client available"}
            except Exception as e:
                results["nemo_test"] = {"success": False, "error": str(e)}

        # Test integrated embedding functionality
        embedding_response = self.create_embeddings(
            ["metformin pharmacokinetics and drug interactions with ACE inhibitors"]
        )
        results["embedding_test"] = {
            "success": embedding_response.success,
            "endpoint_type": embedding_response.endpoint_type.value if embedding_response.endpoint_type else None,
            "response_time_ms": embedding_response.response_time_ms,
            "cost_tier": embedding_response.cost_tier,
            "error": embedding_response.error,
        }

        # Test integrated chat functionality
        chat_response = self.create_chat_completion(
            [{"role": "user", "content": "Explain the mechanism of action of metformin in type 2 diabetes treatment."}]
        )
        results["chat_test"] = {
            "success": chat_response.success,
            "endpoint_type": chat_response.endpoint_type.value if chat_response.endpoint_type else None,
            "response_time_ms": chat_response.response_time_ms,
            "cost_tier": chat_response.cost_tier,
            "error": chat_response.error,
        }

        # Determine overall status
        embedding_ok = results["embedding_test"]["success"]
        chat_ok = results["chat_test"]["success"]

        if embedding_ok and chat_ok:
            results["overall_status"] = "success"
        elif embedding_ok or chat_ok:
            results["overall_status"] = "partial"
        else:
            results["overall_status"] = "failed"

        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and cost analysis."""
        avg_response_time = self.metrics["total_response_time_ms"] / max(self.metrics["total_requests"], 1)

        # Compute self_hosted_requests as sum of fallback_requests + ollama_requests for clarity
        computed_self_hosted = self.metrics["fallback_requests"] + self.metrics["ollama_requests"]

        return {
            "total_requests": self.metrics["total_requests"],
            "cloud_requests": self.metrics["cloud_requests"],
            "ollama_requests": self.metrics["ollama_requests"],
            "fallback_requests": self.metrics["fallback_requests"],
            "self_hosted_requests": self.metrics["self_hosted_requests"],
            "computed_self_hosted_requests": computed_self_hosted,  # For verification
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": (
                (self.metrics["total_requests"] - self.metrics["failed_requests"])
                / max(self.metrics["total_requests"], 1)
            ),
            "avg_response_time_ms": int(avg_response_time),
            "cloud_usage_percentage": (self.metrics["cloud_requests"] / max(self.metrics["total_requests"], 1) * 100),
            "cost_optimization": {
                "free_tier_requests": self.metrics["cloud_requests"],
                "infrastructure_requests": self.metrics["self_hosted_requests"],
                "estimated_monthly_projection": self.metrics["cloud_requests"] * 30,
            },
        }

    def get_endpoint_status(self) -> Dict[str, Any]:
        """Get current status of all endpoints."""
        status = {
            "cloud_available": self.cloud_client is not None,
            "nemo_available": self.nemo_client is not None,
            "fallback_enabled": self.enable_fallback,
            "pharmaceutical_optimized": self.pharmaceutical_optimized,
        }

        # Test cloud endpoint if available
        if self.cloud_client:
            try:
                cloud_test = self.cloud_client.test_connection()
                status["cloud_status"] = cloud_test
            except Exception as e:
                status["cloud_status"] = {"success": False, "error": str(e)}

        # Get configuration strategy
        status["strategy"] = self.config.get_cloud_first_strategy()
        status["endpoint_priority"] = self.config.get_endpoint_priority_order()
        # Surface pharma mode and settings for quick inspection
        try:
            status["pharmaceutical_research_mode"] = bool(getattr(self.config, "pharmaceutical_research_mode", True))
        except Exception:
            status["pharmaceutical_research_mode"] = True
        try:
            status["pharma_settings"] = self.config.get_pharma_settings()
        except Exception:
            status["pharma_settings"] = None
        status["pharma_metrics"] = {
            **self.pharma_metrics,
            "budget": self.config.get_cost_monitoring_config(),
        }

        return status

    # ---------------------------- Reranking ----------------------------
    def rerank_passages(
        self, query: str, passages: List[str], model: Optional[str] = None, top_n: Optional[int] = None
    ) -> ClientResponse:
        """Rerank passages using cloud (NVIDIA Build) with fallback to NeMo.

        Returns data in a normalized schema:
            {"reranked_passages": [{"text": str, "score": float}, ...]}
        sorted by descending score.
        """

        def _cloud_rerank(client: OpenAIWrapper, **kwargs) -> Any:
            # Respect configured model preference when not explicitly provided
            default_model = (
                kwargs.get("model")
                or self.config.get_effective_models().get("reranking")
                or "meta/llama-3_2-nemoretriever-500m-rerank-v2"
            )
            # Normalize to full name for cloud call
            normalized_model = self._normalize_model_name(default_model, prefer_full_name=True)
            rankings = client.rerank(
                query=kwargs["query"],
                candidates=kwargs["passages"],
                top_n=kwargs.get("top_n"),
                model=normalized_model,
            )
            # Normalize and sort desc
            norm = []
            for item in rankings or []:
                txt = item.get("text") or item.get("passage") or ""
                try:
                    score = float(item.get("score") if item.get("score") is not None else 0.0)
                except Exception:
                    score = 0.0
                norm.append({"text": txt, "score": score})
            norm.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return {
                "reranked_passages": norm,
                "model": normalized_model,
            }

        def _nemo_rerank(client: "NeMoRetrieverClient", **kwargs) -> Any:
            async def _run():
                # Local import to avoid circular dependency
                from ..nemo_retriever_client import NeMoRetrieverClient  # type: ignore  # noqa: F401

                # Use short-name for legacy NeMo client, but report full name in response
                default_model = (
                    kwargs.get("model")
                    or self.config.get_effective_models().get("reranking")
                    or "meta/llama-3_2-nemoretriever-500m-rerank-v2"
                )
                default_model_short = self._normalize_model_name(default_model, prefer_full_name=False)
                resp = await client.rerank_passages(
                    query=kwargs["query"],
                    passages=kwargs["passages"],
                    model=default_model_short,
                    top_k=kwargs.get("top_n"),
                )
                if resp.success:
                    data = resp.data or {}
                    ranks = data.get("reranked_passages", [])
                    # Normalize and sort desc
                    norm = []
                    for item in ranks:
                        score = item.get("score") or item.get("relevance") or item.get("relevance_score", 0.0)
                        txt = item.get("text") or item.get("passage") or item.get("document", "")
                        norm.append({"text": txt, "score": float(score)})
                    norm.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                    return {"reranked_passages": norm, "model": self._normalize_model_name(default_model)}
                raise RuntimeError(resp.error or "NeMo rerank failed")

            return self.run_coro_sync(_run())

        return self._execute_with_fallback(
            "reranking",
            _cloud_rerank,
            _nemo_rerank,
            query=query,
            passages=passages,
            model=model,
            top_n=top_n,
        )

    async def rerank_passages_async(
        self, query: str, passages: List[str], model: Optional[str] = None, top_n: Optional[int] = None
    ) -> ClientResponse:
        """Async-compatible alias matching legacy client signature."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.rerank_passages(query, passages, model, top_n))


# Advanced pharmaceutical client factory
def create_pharmaceutical_client(config: Optional[EnhancedRAGConfig] = None) -> EnhancedNeMoClient:
    """
    Advanced pharmaceutical-optimized client factory.

    For most use cases, prefer create_client() in src.nemo_retriever_client.
    This factory provides enhanced pharmaceutical domain features for advanced applications.

    Args:
        config: Optional enhanced RAG configuration (auto-loaded if None)

    Returns:
        EnhancedNeMoClient configured for pharmaceutical research with cloud-first strategy
    """
    return EnhancedNeMoClient(
        config=config or EnhancedRAGConfig.from_env(), pharmaceutical_optimized=True, enable_fallback=True
    )


if __name__ == "__main__":
    # Quick pharmaceutical capability test
    import json

    try:
        client = create_pharmaceutical_client()
        results = client.test_pharmaceutical_capabilities()
        print("Pharmaceutical Client Test Results:")
        print(json.dumps(results, indent=2))

        # Performance metrics
        metrics = client.get_performance_metrics()
        print("\nPerformance Metrics:")
        print(json.dumps(metrics, indent=2))

    except Exception as e:
        print(f"Client test failed: {str(e)}")
