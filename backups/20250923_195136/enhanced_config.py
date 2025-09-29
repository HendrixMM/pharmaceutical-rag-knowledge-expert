"""Central configuration management for enhanced RAG and PubMed features."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

_BOOL_TRUE = {"1", "true", "yes", "on", "enabled"}
_BOOL_FALSE = {"0", "false", "no", "off", "disabled"}
_ERROR_HANDLING_MODES = {"graceful", "strict", "silent"}


def _get_env(env: Mapping[str, str], key: str) -> Optional[str]:
    value = env.get(key)
    return value.strip() if isinstance(value, str) else value


def _as_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _get_env(env, key)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    raise ValueError(f"Environment variable {key} must be boolean-like, got: {raw}")


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _get_env(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer, got: {raw}") from exc


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _get_env(env, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be numeric, got: {raw}") from exc


@dataclass
class EnhancedRAGConfig:
    """Configuration snapshot for hybrid RAG + PubMed functionality."""

    enhanced_features_enabled: bool = False
    enable_pubmed_integration: bool = False
    pubmed_hybrid_mode: bool = False
    pubmed_fallback_enabled: bool = True
    pubmed_cache_integration: bool = True

    max_external_results: int = 10
    relevance_threshold: float = 0.7
    query_timeout_seconds: int = 30
    enable_query_enhancement: bool = True

    enable_status_reporting: bool = True
    enable_metrics_collection: bool = True
    enable_performance_logging: bool = False
    enable_dashboard_integration: bool = True

    gradual_rollout_enabled: bool = False
    rollout_percentage: float = 0.0
    beta_mode: bool = False

    safe_mode: bool = True
    error_handling_strategy: str = "graceful"
    fallback_to_local_on_error: bool = True

    enable_enhanced_pubmed_scraper: bool = False
    enable_advanced_caching: bool = True
    use_normalized_cache_keys: bool = False
    mirror_legacy_cache: bool = False
    prune_legacy_after_migration: bool = False

    enable_rate_limiting: bool = True
    rate_limit_window_seconds: int = 1
    rate_limit_max_requests: int = 3

    cache_prefetch_on_startup: bool = False
    cache_cleanup_on_startup: bool = False

    # NIM-specific configuration
    enable_nim_health_gate: bool = True
    health_check_timeout_seconds: int = 10
    max_health_latency_ms: float = 5000.0
    health_check_retry_attempts: int = 2

    # Model overrides
    embedding_model_override: Optional[str] = None
    reranking_model_override: Optional[str] = None
    extraction_model_override: Optional[str] = None

    # Endpoint overrides
    custom_embedding_endpoint: Optional[str] = None
    custom_reranking_endpoint: Optional[str] = None
    custom_extraction_endpoint: Optional[str] = None

    # NVIDIA Build platform integration
    enable_nvidia_build_fallback: bool = False
    nvidia_build_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_build_embedding_model: Optional[str] = None
    nvidia_build_llm_model: Optional[str] = None

    config_loaded_at: float = field(default_factory=time.time)
    source_env: Dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "EnhancedRAGConfig":
        """Build a configuration snapshot from environment variables."""
        env_map: Mapping[str, str] = env or os.environ

        try:
            enhanced_features_enabled = _as_bool(env_map, "RAG_ENHANCED_FEATURES_ENABLED", False)
            enable_pubmed_integration = _as_bool(env_map, "ENABLE_RAG_PUBMED_INTEGRATION", False)
            pubmed_hybrid_mode = _as_bool(env_map, "RAG_PUBMED_HYBRID_MODE", False)
            pubmed_fallback_enabled = _as_bool(env_map, "RAG_PUBMED_FALLBACK_ENABLED", True)
            pubmed_cache_integration = _as_bool(env_map, "RAG_PUBMED_CACHE_INTEGRATION", True)

            max_external_results = _as_int(env_map, "RAG_PUBMED_MAX_EXTERNAL_RESULTS", 10)
            relevance_threshold = _as_float(env_map, "RAG_PUBMED_RELEVANCE_THRESHOLD", 0.7)
            query_timeout_seconds = _as_int(env_map, "RAG_PUBMED_QUERY_TIMEOUT_SECONDS", 30)
            enable_query_enhancement = _as_bool(env_map, "RAG_PUBMED_ENABLE_QUERY_ENHANCEMENT", True)

            enable_status_reporting = _as_bool(env_map, "ENABLE_RAG_PUBMED_STATUS_REPORTING", True)
            enable_metrics_collection = _as_bool(env_map, "RAG_PUBMED_METRICS_COLLECTION", True)
            enable_performance_logging = _as_bool(env_map, "RAG_PUBMED_PERFORMANCE_LOGGING", False)
            enable_dashboard_integration = _as_bool(env_map, "RAG_PUBMED_DASHBOARD_INTEGRATION", True)

            gradual_rollout_enabled = _as_bool(env_map, "RAG_PUBMED_GRADUAL_ROLLOUT", False)
            rollout_percentage = _as_float(env_map, "RAG_PUBMED_ROLLOUT_PERCENTAGE", 0.0)
            beta_mode = _as_bool(env_map, "RAG_PUBMED_BETA_MODE", False)

            safe_mode = _as_bool(env_map, "RAG_PUBMED_SAFE_MODE", True)
            error_handling_strategy = _get_env(env_map, "RAG_PUBMED_ERROR_HANDLING") or "graceful"
            fallback_to_local_on_error = _as_bool(env_map, "RAG_PUBMED_FALLBACK_TO_LOCAL", True)

            enable_enhanced_pubmed_scraper = _as_bool(env_map, "ENABLE_ENHANCED_PUBMED_SCRAPER", False)
            enable_advanced_caching = _as_bool(env_map, "ENABLE_ADVANCED_CACHING", True)
            use_normalized_cache_keys = _as_bool(env_map, "USE_NORMALIZED_CACHE_KEYS", False)
            mirror_legacy_cache = _as_bool(env_map, "MIRROR_TO_LEGACY_ROOT", False)
            prune_legacy_after_migration = _as_bool(env_map, "PRUNE_LEGACY_AFTER_MIGRATION", False)

            # Support both ENABLE_PUBMED_RATE_LIMITING and ENABLE_RATE_LIMITING for compatibility
            enable_rate_limiting = _as_bool(env_map, "ENABLE_RATE_LIMITING",
                                  _as_bool(env_map, "ENABLE_PUBMED_RATE_LIMITING", True))
            rate_limit_window_seconds = _as_int(env_map, "PUBMED_RATE_LIMIT_WINDOW_SECONDS", 1)
            # Support both PUBMED_RATE_LIMIT_MAX_REQUESTS and MAX_REQUESTS_PER_SECOND for compatibility
            rate_limit_max_requests = _as_int(env_map, "MAX_REQUESTS_PER_SECOND",
                                     _as_int(env_map, "PUBMED_RATE_LIMIT_MAX_REQUESTS", 3))

            cache_prefetch_on_startup = _as_bool(env_map, "RAG_PUBMED_CACHE_PREFETCH", False)
            cache_cleanup_on_startup = _as_bool(env_map, "RAG_PUBMED_CACHE_CLEANUP", False)

            # NIM / NEMO specific
            enable_nim_health_gate = _as_bool(env_map, "ENABLE_NIM_HEALTH_GATE", True)
            health_check_timeout_seconds = _as_int(env_map, "HEALTH_CHECK_TIMEOUT_SECONDS", 10)
            # Support both HEALTH_LATENCY_MS and HEALTH_LATENCY_MS (legacy naming)
            max_health_latency_ms = _as_float(env_map, "HEALTH_LATENCY_MS", 5000.0)
            health_check_retry_attempts = _as_int(env_map, "HEALTH_CHECK_RETRY_ATTEMPTS", 2)

            embedding_model_override = _get_env(env_map, "EMBEDDING_MODEL")
            reranking_model_override = _get_env(env_map, "RERANK_MODEL")
            extraction_model_override = _get_env(env_map, "EXTRACTION_MODEL")

            custom_embedding_endpoint = _get_env(env_map, "NEMO_EMBEDDING_ENDPOINT")
            custom_reranking_endpoint = _get_env(env_map, "NEMO_RERANKING_ENDPOINT")
            custom_extraction_endpoint = _get_env(env_map, "NEMO_EXTRACTION_ENDPOINT")

            # NVIDIA Build platform integration
            enable_nvidia_build_fallback = _as_bool(env_map, "ENABLE_NVIDIA_BUILD_FALLBACK", False)
            nvidia_build_base_url = _get_env(env_map, "NVIDIA_BUILD_BASE_URL") or "https://integrate.api.nvidia.com/v1"
            nvidia_build_embedding_model = _get_env(env_map, "NVIDIA_BUILD_EMBEDDING_MODEL")
            nvidia_build_llm_model = _get_env(env_map, "NVIDIA_BUILD_LLM_MODEL")
        except ValueError as exc:
            config = cls()
            config.errors.append(str(exc))
            return config

        config = cls(
            enhanced_features_enabled=enhanced_features_enabled,
            enable_pubmed_integration=enable_pubmed_integration,
            pubmed_hybrid_mode=pubmed_hybrid_mode,
            pubmed_fallback_enabled=pubmed_fallback_enabled,
            pubmed_cache_integration=pubmed_cache_integration,
            max_external_results=max_external_results,
            relevance_threshold=relevance_threshold,
            query_timeout_seconds=query_timeout_seconds,
            enable_query_enhancement=enable_query_enhancement,
            enable_status_reporting=enable_status_reporting,
            enable_metrics_collection=enable_metrics_collection,
            enable_performance_logging=enable_performance_logging,
            enable_dashboard_integration=enable_dashboard_integration,
            gradual_rollout_enabled=gradual_rollout_enabled,
            rollout_percentage=rollout_percentage,
            beta_mode=beta_mode,
            safe_mode=safe_mode,
            error_handling_strategy=error_handling_strategy,
            fallback_to_local_on_error=fallback_to_local_on_error,
            enable_enhanced_pubmed_scraper=enable_enhanced_pubmed_scraper,
            enable_advanced_caching=enable_advanced_caching,
            use_normalized_cache_keys=use_normalized_cache_keys,
            mirror_legacy_cache=mirror_legacy_cache,
            prune_legacy_after_migration=prune_legacy_after_migration,
            enable_rate_limiting=enable_rate_limiting,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_max_requests=rate_limit_max_requests,
            cache_prefetch_on_startup=cache_prefetch_on_startup,
            cache_cleanup_on_startup=cache_cleanup_on_startup,
            source_env=dict(env_map),
            enable_nim_health_gate=enable_nim_health_gate,
            health_check_timeout_seconds=health_check_timeout_seconds,
            max_health_latency_ms=max_health_latency_ms,
            health_check_retry_attempts=health_check_retry_attempts,
            embedding_model_override=embedding_model_override,
            reranking_model_override=reranking_model_override,
            extraction_model_override=extraction_model_override,
            custom_embedding_endpoint=custom_embedding_endpoint,
            custom_reranking_endpoint=custom_reranking_endpoint,
            custom_extraction_endpoint=custom_extraction_endpoint,
            enable_nvidia_build_fallback=enable_nvidia_build_fallback,
            nvidia_build_base_url=nvidia_build_base_url,
            nvidia_build_embedding_model=nvidia_build_embedding_model,
            nvidia_build_llm_model=nvidia_build_llm_model,
        )
        config._validate()
        return config

    # ------------------------------------------------------------------
    # Validation and mutation helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        self.warnings.clear()
        self.errors.clear()

        if not 0.0 <= self.relevance_threshold <= 1.0:
            self.warnings.append(
                "Relevance threshold must be between 0 and 1; clamping to valid range."
            )
            self.relevance_threshold = max(0.0, min(1.0, self.relevance_threshold))

        if self.max_external_results < 0:
            self.warnings.append("Max external results cannot be negative; using 0.")
            self.max_external_results = 0

        if self.rollout_percentage < 0:
            self.warnings.append("Rollout percentage cannot be negative; using 0.")
            self.rollout_percentage = 0.0
        elif self.rollout_percentage > 100.0:
            self.warnings.append("Rollout percentage cannot exceed 100; capping to 100.")
            self.rollout_percentage = 100.0

        if self.query_timeout_seconds <= 0:
            self.warnings.append("Query timeout must be positive; defaulting to 30 seconds.")
            self.query_timeout_seconds = 30

        if self.rate_limit_window_seconds <= 0:
            self.warnings.append("Rate limit window must be positive; defaulting to 1 second.")
            self.rate_limit_window_seconds = 1

        if self.rate_limit_max_requests <= 0:
            self.warnings.append("Rate limit max requests must be positive; defaulting to 1.")
            self.rate_limit_max_requests = 1

        if self.error_handling_strategy not in _ERROR_HANDLING_MODES:
            self.warnings.append(
                "Unknown error handling mode '%s'; defaulting to 'graceful'." % self.error_handling_strategy
            )
            self.error_handling_strategy = "graceful"

        # Basic sanity for health check settings
        if self.health_check_timeout_seconds <= 0:
            self.warnings.append("Health check timeout must be positive; defaulting to 10 seconds.")
            self.health_check_timeout_seconds = 10
        if self.max_health_latency_ms < 0:
            self.warnings.append("Max health latency must be non-negative; defaulting to 5000ms.")
            self.max_health_latency_ms = 5000.0
        if self.health_check_retry_attempts < 0:
            self.warnings.append("Health check retries cannot be negative; using 0.")
            self.health_check_retry_attempts = 0

        if not self.enable_pubmed_integration:
            if self.pubmed_hybrid_mode:
                self.warnings.append(
                    "Hybrid mode requested without PubMed integration; disabling hybrid mode."
                )
                self.pubmed_hybrid_mode = False
            if self.pubmed_cache_integration:
                self.warnings.append(
                    "Cache integration requested without PubMed integration; disabling cache integration."
                )
                self.pubmed_cache_integration = False

        if not self.enable_enhanced_pubmed_scraper and self.enable_advanced_caching:
            self.warnings.append(
                "Advanced caching requested but enhanced scraper disabled; disabling advanced caching."
            )
            self.enable_advanced_caching = False
            self.use_normalized_cache_keys = False
            self.mirror_legacy_cache = False
            self.prune_legacy_after_migration = False

        if self.enable_advanced_caching and not self.pubmed_cache_integration:
            self.warnings.append(
                "Advanced caching enabled without RAG cache integration; consider enabling integration for best results."
            )

        if not self.enhanced_features_enabled and (
            self.enable_pubmed_integration
            or self.pubmed_hybrid_mode
            or self.beta_mode
        ):
            self.warnings.append(
                "Specific PubMed features requested while master switch is off; features remain disabled."
            )

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    def to_dict(self, include_env: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_env:
            data.pop("source_env", None)
        return data

    def export_public_view(self) -> Dict[str, Any]:
        data = self.to_dict(include_env=False)
        data.pop("warnings", None)
        data.pop("errors", None)
        data.pop("_lock", None)
        return data

    def should_enable_pubmed(self) -> bool:
        return bool(
            self.enhanced_features_enabled
            and self.enable_pubmed_integration
            and not self.errors
        )

    def should_use_hybrid_mode(self) -> bool:
        return bool(self.should_enable_pubmed() and self.pubmed_hybrid_mode)

    def is_rollout_active(self) -> bool:
        return bool(self.gradual_rollout_enabled and self.rollout_percentage > 0)

    def reload(self, env: Optional[Mapping[str, str]] = None) -> None:
        new_config = self.from_env(env)
        with self._lock:
            for key, value in asdict(new_config).items():
                if key == "_lock":
                    continue
                setattr(self, key, value)
            self.config_loaded_at = time.time()

    def apply_overrides(self, overrides: MutableMapping[str, Any]) -> None:
        with self._lock:
            for key, value in overrides.items():
                if not hasattr(self, key):
                    raise AttributeError(f"Unknown configuration field: {key}")
                setattr(self, key, value)
            self._validate()
            self.config_loaded_at = time.time()

    def summarize_flags(self) -> Dict[str, Any]:
        return {
            "enabled": self.should_enable_pubmed(),
            "hybrid": self.should_use_hybrid_mode(),
            "gradual_rollout": self.is_rollout_active(),
            "beta_mode": self.beta_mode,
            "safe_mode": self.safe_mode,
            "status_reporting": self.enable_status_reporting,
        }

    # ------------------------------------------------------------------
    # NIM-native helpers
    # ------------------------------------------------------------------
    def validate_nim_configuration(self) -> list[str]:
        errs: list[str] = []
        try:
            from src.nemo_retriever_client import NeMoRetrieverClient
        except Exception:
            # If client import fails, skip strict model validation here
            return errs
        if self.embedding_model_override and not NeMoRetrieverClient.validate_model_availability("embedding", self.embedding_model_override):
            errs.append(f"Invalid embedding model override: {self.embedding_model_override}")
        if self.reranking_model_override and not NeMoRetrieverClient.validate_model_availability("reranking", self.reranking_model_override):
            errs.append(f"Invalid reranking model override: {self.reranking_model_override}")
        return errs

    def get_effective_models(self) -> Dict[str, str]:
        # Defaults aligned with client
        eff = {
            "embedding": self.embedding_model_override or "nvidia/nv-embedqa-e5-v5",
            "reranking": self.reranking_model_override or "llama-3_2-nemoretriever-500m-rerank-v2",
            "extraction": self.extraction_model_override or "nvidia/nv-ingest",
        }
        return eff

    def get_effective_endpoints(self) -> Dict[str, str]:
        try:
            from src.nemo_retriever_client import NeMoRetrieverClient
            defaults = NeMoRetrieverClient.DEFAULT_ENDPOINTS
        except Exception:
            defaults = {
                "embedding": os.getenv("NEMO_EMBEDDING_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"),
                "reranking": os.getenv("NEMO_RERANKING_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"),
                "extraction": os.getenv("NEMO_EXTRACTION_ENDPOINT", "https://ai.api.nvidia.com/v1/retrieval/nvidia/extraction"),
            }
        return {
            "embedding": self.custom_embedding_endpoint or defaults.get("embedding", ""),
            "reranking": self.custom_reranking_endpoint or defaults.get("reranking", ""),
            "extraction": self.custom_extraction_endpoint or defaults.get("extraction", ""),
        }

    def get_health_check_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enable_nim_health_gate,
            "timeout_seconds": self.health_check_timeout_seconds,
            "max_latency_ms": self.max_health_latency_ms,
            "retries": self.health_check_retry_attempts,
        }

    def get_nim_overrides(self) -> Dict[str, Any]:
        return {
            "models": {
                "embedding": self.embedding_model_override,
                "reranking": self.reranking_model_override,
                "extraction": self.extraction_model_override,
            },
            "endpoints": {
                "embedding": self.custom_embedding_endpoint,
                "reranking": self.custom_reranking_endpoint,
                "extraction": self.custom_extraction_endpoint,
            },
        }

    def get_nvidia_build_config(self) -> Dict[str, Any]:
        """Get NVIDIA Build platform configuration."""
        return {
            "enabled": self.enable_nvidia_build_fallback,
            "base_url": self.nvidia_build_base_url,
            "embedding_model": self.nvidia_build_embedding_model or "nvidia/nv-embed-v1",
            "llm_model": self.nvidia_build_llm_model or "meta/llama-3.1-8b-instruct",
            "endpoints": {
                "embeddings": f"{self.nvidia_build_base_url}/embeddings",
                "chat_completions": f"{self.nvidia_build_base_url}/chat/completions",
            }
        }

    def should_use_nvidia_build_fallback(self) -> bool:
        """Check if NVIDIA Build fallback should be used."""
        return self.enable_nvidia_build_fallback


__all__ = ["EnhancedRAGConfig"]
