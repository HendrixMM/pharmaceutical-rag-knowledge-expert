"""PubMed-specific functionality for enhanced RAG agent.

This module contains PubMed-specific logic extracted from EnhancedRAGAgent
to keep the core agent focused on safety validation while providing
optional PubMed integration through minimal hooks.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

from .enhanced_config import EnhancedRAGConfig
from .enhanced_pubmed_scraper import EnhancedPubMedScraper
from .query_engine import EnhancedQueryEngine

logger = logging.getLogger(__name__)


class PubMedIntegrationManager:
    """Manages PubMed integration components and functionality for RAG agents."""

    def __init__(
        self,
        config: EnhancedRAGConfig,
        pubmed_query_engine: EnhancedQueryEngine | None = None,
        pubmed_scraper: EnhancedPubMedScraper | None = None,
    ) -> None:
        """Initialize PubMed integration manager.

        Args:
            config: Enhanced RAG configuration
            pubmed_query_engine: Optional pre-configured query engine
            pubmed_scraper: Optional pre-configured scraper
        """
        self.config = config
        self._pubmed_query_engine: EnhancedQueryEngine | None = pubmed_query_engine
        self._pubmed_scraper: EnhancedPubMedScraper | None = pubmed_scraper
        self._pubmed_components_initialized = False
        self._pubmed_lock = threading.RLock()

        # PubMed-specific metrics
        self.pubmed_metrics: dict[str, Any] = {
            "total_queries": 0,
            "cache_hits": 0,
            "error_count": 0,
            "last_latency_ms": None,
            "last_mode": None,
            "last_result_count": 0,
            "last_error": None,
            "last_updated": None,
            "rollout_active": self.config.is_rollout_active(),
        }

        # Component health tracking
        self.component_health: dict[str, dict[str, Any]] = {
            "pubmed_integration": {
                "status": "disabled" if not self.config.should_enable_pubmed() else "pending",
                "enabled_via_config": self.config.should_enable_pubmed(),
                "hybrid_mode": self.config.pubmed_hybrid_mode,
                "cache_integration": self.config.pubmed_cache_integration,
            }
        }

    @property
    def pubmed_scraper(self) -> EnhancedPubMedScraper | None:
        """Get the PubMed scraper instance. Read-only access."""
        return self._pubmed_scraper

    @property
    def pubmed_query_engine(self) -> EnhancedQueryEngine | None:
        """Get the PubMed query engine instance. Read-only access."""
        return self._pubmed_query_engine

    def initialize_pubmed_components(self) -> None:
        """Initialize PubMed components with health checks."""
        with self._pubmed_lock:
            if self._pubmed_components_initialized:
                return

            pubmed_health: dict[str, Any] = self.component_health.get("pubmed_integration", {}) or {}

            # Check if PubMed is enabled
            if not self.config.should_enable_pubmed():
                pubmed_health.update(
                    status="disabled",
                    message="PubMed integration disabled via configuration",
                    last_checked=time.time(),
                )
                self.component_health["pubmed_integration"] = pubmed_health
                self._pubmed_components_initialized = True
                return

            # Ensure scraper is available
            if self._pubmed_query_engine is not None and self._pubmed_scraper is None:
                self._pubmed_scraper = getattr(self._pubmed_query_engine, "scraper", None)

            if self._pubmed_scraper is None:
                try:
                    # Configure scraper based on settings
                    scraper_kwargs = {
                        "enable_rate_limiting": self.config.enable_rate_limiting,
                        "enable_advanced_caching": self.config.enable_advanced_caching,
                        "use_normalized_cache_keys": self.config.use_normalized_cache_keys,
                    }
                    if not self.config.enable_enhanced_pubmed_scraper:
                        scraper_kwargs["enable_advanced_caching"] = False
                        scraper_kwargs["use_normalized_cache_keys"] = False

                    self._pubmed_scraper = EnhancedPubMedScraper(**scraper_kwargs)
                    pubmed_health.update(
                        status="ready",
                        scraper_type="enhanced" if self.config.enable_enhanced_pubmed_scraper else "standard",
                        last_initialized=time.time(),
                    )
                except Exception as exc:
                    logger.error("Failed to initialize PubMed scraper: %s", exc)
                    pubmed_health.update(
                        status="error",
                        error=str(exc),
                        last_error=time.time(),
                    )
                    self.component_health["pubmed_integration"] = pubmed_health
                    self._pubmed_components_initialized = True
                    self._pubmed_scraper = None
                    self._pubmed_query_engine = None
                    return

            # Ensure query engine is available
            scraper = self._pubmed_scraper
            if self._pubmed_query_engine is None and scraper is not None:
                try:
                    engine_kwargs = {
                        "enable_query_enhancement": self.config.enable_query_enhancement,
                        "cache_filtered_results": self.config.pubmed_cache_integration,
                    }
                    self._pubmed_query_engine = EnhancedQueryEngine(scraper, **engine_kwargs)
                    pubmed_health.update(
                        status="ready",
                        query_engine_type="enhanced",
                        last_initialized=time.time(),
                    )
                except Exception as exc:
                    logger.error("Failed to initialize PubMed query engine: %s", exc)
                    pubmed_health.update(
                        status="error",
                        error=str(exc),
                        last_error=time.time(),
                    )
                    self.component_health["pubmed_integration"] = pubmed_health
                    self._pubmed_components_initialized = True
                    self._pubmed_query_engine = None
                    return

            # Final validation
            if self._pubmed_query_engine is None or self._pubmed_scraper is None:
                pubmed_health.update(
                    status="error",
                    error="Required components failed to initialize",
                    last_error=time.time(),
                )
                self._pubmed_components_initialized = True
                return

            # Update health with component details
            pubmed_health.update(
                status="ready",
                components={
                    "scraper": type(self._pubmed_scraper).__name__,
                    "query_engine": type(self._pubmed_query_engine).__name__,
                },
                enhanced_scraper=self.config.enable_enhanced_pubmed_scraper,
                cache_integration=self.config.pubmed_cache_integration,
                rate_limiting=self.config.enable_rate_limiting,
                last_health_check=time.time(),
            )

            # Refresh component health
            self._refresh_pubmed_component_health_locked(scraper)

            self.component_health["pubmed_integration"] = pubmed_health
            self._pubmed_components_initialized = True

    def _refresh_pubmed_component_health_locked(self, scraper: EnhancedPubMedScraper) -> None:
        """Refresh PubMed component health status."""
        try:
            status_report = scraper.combined_status_report()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.component_health.setdefault("pubmed_integration", {})["telemetry_error"] = str(exc)
            return

        health = self.component_health.setdefault("pubmed_integration", {})
        health.update(
            {
                "cache": status_report.get("cache", {}),
                "rate_limit": status_report.get("rate_limit", {}),
                "last_telemetry_update": time.time(),
            }
        )

    def is_pubmed_available(self) -> bool:
        """Check if PubMed integration is available."""
        return bool(
            self._pubmed_query_engine
            and self._pubmed_scraper
            and self.component_health.get("pubmed_integration", {}).get("status") == "ready"
        )

    def should_activate_pubmed_for_query(self, *, force_pubmed: bool = False) -> bool:
        """Determine if PubMed should be activated for a query."""
        if not self.config.should_enable_pubmed():
            return False
        if force_pubmed:
            return True

        # Check rollout status
        is_rollout_active = getattr(self.config, "is_rollout_active", None)
        if not callable(is_rollout_active) or not is_rollout_active():
            return True

        # Get rollout percentage
        try:
            rollout_pct = float(getattr(self.config, "rollout_percentage", 0.0))
        except (ValueError, TypeError):
            rollout_pct = 0.0

        probability = max(0.0, min(1.0, rollout_pct / 100.0))
        selected = random.random() < probability

        # Update metrics
        self.pubmed_metrics["rollout_active"] = True
        self.pubmed_metrics["last_rollout_selected"] = selected
        self.pubmed_metrics["last_rollout_probability"] = probability

        return selected

    def filter_pubmed_results(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Filter and format PubMed results from query engine response."""
        results = payload.get("results", [])
        if not results:
            return []

        filtered = []
        for result in results:
            # Skip results without required fields
            if not result.get("pmid") or not result.get("title"):
                continue

            # Apply result filtering based on configuration
            if self.config.enable_query_enhancement:
                # Enhanced filtering for pharmaceutical queries
                if result.get("abstract") and len(result["abstract"]) < 50:
                    continue  # Skip very short abstracts

            filtered.append(result)

        # Update metrics
        self.pubmed_metrics["last_result_count"] = len(filtered)
        self.pubmed_metrics["total_queries"] += 1

        return filtered

    def classify_query_mode(self, query: str) -> str:
        """Classify query routing mode (local, pubmed, combined, hybrid)."""
        normalized = (query or "").lower().strip()

        # Explicit routing terms
        local_only_terms = (
            "local documents",
            "local files",
            "my documents",
            "uploaded files",
            "corporate",
            "internal",
            "proprietary",
            "company",
            "organization",
        )
        pubmed_only_terms = (
            "pubmed",
            "medical literature",
            "clinical trials",
            "research studies",
            "recent studies",
            "pubmed search",
            "medical database",
        )

        if any(term in normalized for term in local_only_terms):
            return "local"
        if any(term in normalized for term in pubmed_only_terms):
            return "pubmed"

        # Default behavior based on configuration
        if not self.config.should_use_hybrid_mode():
            return "combined"

        # For short queries, prefer local
        if len(normalized.split()) <= 3:
            return "local"

        return "combined"

    def classify_query(self, query: str) -> str:
        """Classify query as medical, general, or pharmaceutical."""
        normalized = (query or "").lower()

        # Pharmaceutical classification terms
        pharma_terms = (
            "drug interaction",
            "pharmacokinetic",
            "pharmacodynamic",
            "adverse reaction",
            "contraindication",
            "dosage",
            "administration",
            "bioavailability",
            "metabolism",
            "drug metabolism",
            "cytochrome",
            "cyp",
            "induction",
            "inhibition",
            "therapeutic",
            "half-life",
            "clearance",
            "protein binding",
            "distribution",
            "elimination",
        )

        # Medical classification terms
        medical_terms = (
            "disease",
            "symptom",
            "treatment",
            "diagnosis",
            "therapy",
            "clinical",
            "patient",
            "medicine",
            "health",
            "medical",
            "hospital",
            "healthcare",
        )

        if any(term in normalized for term in pharma_terms):
            return "pharmaceutical"
        elif any(term in normalized for term in medical_terms):
            return "medical"
        else:
            return "general"

    def get_pubmed_health(self) -> dict[str, Any]:
        """Get current PubMed integration health status."""
        return self.component_health.get("pubmed_integration", {})

    def get_pubmed_metrics(self) -> dict[str, Any]:
        """Get PubMed metrics."""
        return self.pubmed_metrics.copy()
