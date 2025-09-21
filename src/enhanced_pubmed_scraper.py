"""Enhanced PubMed scraper integrating advanced caching with rate limiting."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .cache_management import CacheLookupResult, NCBICacheManager
from .pubmed_scraper import PubMedScraper
from .rate_limiting import NCBIRateLimiter, RateLimitStatus

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


BatchRequest = Union[str, Mapping[str, Any]]


class EnhancedPubMedScraper(PubMedScraper):
    """PubMed scraper with intelligent caching and rate limiting integration."""

    def __init__(
        self,
        *args: Any,
        cache_manager: Optional[NCBICacheManager] = None,
        rate_limiter: Optional[NCBIRateLimiter] = None,
        enable_rate_limiting: Optional[bool] = None,
        enable_advanced_caching: Optional[bool] = None,
        use_normalized_cache_keys: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self._advanced_caching_requested = (
            True
            if cache_manager is not None
            else (
                enable_advanced_caching
                if enable_advanced_caching is not None
                else _env_flag("ENABLE_ADVANCED_CACHING", "true")
            )
        )
        self._cache_manager: Optional[NCBICacheManager] = cache_manager if self._advanced_caching_requested else None
        self._cache_context = threading.local()
        self._use_normalized_cache_keys = (
            use_normalized_cache_keys
            if use_normalized_cache_keys is not None
            else _env_flag("USE_NORMALIZED_CACHE_KEYS", "false")
        )
        self._mirror_legacy_root = _env_flag("MIRROR_TO_LEGACY_ROOT", "false")
        self._prune_legacy_after_migration = _env_flag("PRUNE_LEGACY_AFTER_MIGRATION", "false")

        super().__init__(
            *args,
            rate_limiter=rate_limiter,
            enable_rate_limiting=enable_rate_limiting,
            **kwargs,
        )

        if self._advanced_caching_requested and self._cache_manager is None:
            self._cache_manager = NCBICacheManager.from_env()

        if not self._advanced_caching_requested or self._cache_manager is None:
            self._use_normalized_cache_keys = False
            self._mirror_legacy_root = False
            self._prune_legacy_after_migration = False

        if self._cache_manager:
            logger.info(
                "Advanced PubMed caching enabled (backend=%s, dir=%s)",
                self._cache_manager.cache_backend,
                self._cache_manager.cache_dir,
            )
        else:
            logger.info("Advanced PubMed caching disabled; falling back to legacy caching")

    # ------------------------------------------------------------------
    # Cache context helpers
    # ------------------------------------------------------------------

    def _advanced_cache_active(self) -> bool:
        return bool(self._cache_manager)

    def _get_cache_context(self) -> Dict[str, Any]:
        ctx = getattr(self._cache_context, "value", None)
        if not isinstance(ctx, dict):
            ctx = {}
            self._set_cache_context(ctx)
        return ctx

    def _set_cache_context(self, ctx: Dict[str, Any]) -> None:
        self._cache_context.value = ctx

    def _update_cache_context(self, **updates: Any) -> None:
        ctx = self._get_cache_context()
        ctx.update(updates)
        self._set_cache_context(ctx)

    def _clear_cache_context(self) -> None:
        self._set_cache_context({})

    # ------------------------------------------------------------------
    # PubMedScraper overrides
    # ------------------------------------------------------------------

    def _enhance_pharmaceutical_query(self, query: str, enhancement_enabled: bool) -> str:
        enhanced = super()._enhance_pharmaceutical_query(query, enhancement_enabled)
        if self._advanced_cache_active():
            self._set_cache_context(
                {
                    "original_query": query,
                    "enhanced_query": enhanced,
                }
            )
        return enhanced

    def _get_cache_key(
        self,
        enhanced_query: str,
        max_items: int,
        apply_ranking: bool,
        pharma_enhance_enabled: bool,
        *,
        include_tags_effective: bool,
        include_abstract_effective: bool,
        preserve_order: bool,
    ) -> str:
        cache_key = super()._get_cache_key(
            enhanced_query,
            max_items,
            apply_ranking,
            pharma_enhance_enabled,
            include_tags_effective=include_tags_effective,
            include_abstract_effective=include_abstract_effective,
            preserve_order=preserve_order,
        )
        if not self._advanced_cache_active():
            return cache_key

        context = self._get_cache_context()
        context.setdefault("enhanced_query", enhanced_query)
        context.setdefault("original_query", context.get("original_query", enhanced_query))
        parameters = {
            "max_items": max_items,
            "apply_ranking": apply_ranking,
            "pharma_enhance": pharma_enhance_enabled,
            "include_tags": include_tags_effective,
            "include_abstract": include_abstract_effective,
            "preserve_order": preserve_order,
        }
        context["parameters"] = parameters
        context["legacy_cache_key"] = cache_key

        if self._use_normalized_cache_keys and self._cache_manager:
            normalized_key = self._cache_manager.build_cache_key(enhanced_query, parameters)
            context["normalized_cache_key"] = normalized_key
            context["cache_key"] = normalized_key
            self._set_cache_context(context)
            return normalized_key

        context["cache_key"] = cache_key
        self._set_cache_context(context)
        return cache_key

    def _get_cached_results(self, cache_key: str, apply_ranking: bool):
        if not self._advanced_cache_active():
            return super()._get_cached_results(cache_key, apply_ranking)

        context = self._get_cache_context()
        legacy_cache_key = context.get("legacy_cache_key")
        allow_stale = self._cache_manager.cache_allow_stale_within_grace  # type: ignore[union-attr]
        lookup = self._cache_manager.get(cache_key, allow_stale=allow_stale)  # type: ignore[union-attr]

        if not lookup and self._use_normalized_cache_keys and legacy_cache_key and legacy_cache_key != cache_key:
            legacy_lookup = self._cache_manager.get(legacy_cache_key, allow_stale=allow_stale)  # type: ignore[union-attr]
            if legacy_lookup:
                logger.info(
                    "Migrating legacy cache entry %s to normalized key %s",
                    legacy_cache_key,
                    cache_key,
                )
                try:
                    migrated_metadata = dict(legacy_lookup.metadata)
                    migrated_metadata["cache_key"] = cache_key
                    migrated_metadata["legacy_cache_key"] = legacy_cache_key
                    self._cache_manager.set(  # type: ignore[union-attr]
                        cache_key,
                        legacy_lookup.payload,
                        metadata=migrated_metadata,
                        status=str(legacy_lookup.metadata.get("status", "success")),
                        preserve_expiry=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Failed to migrate legacy cache entry %s: %s", legacy_cache_key, exc)
                else:
                    lookup = self._cache_manager.get(cache_key, allow_stale=allow_stale)  # type: ignore[union-attr]
                    if lookup and self._prune_legacy_after_migration:
                        try:
                            self._cache_manager.prune_legacy_entry(legacy_cache_key)  # type: ignore[union-attr]
                        except Exception as exc:  # pragma: no cover - defensive logging
                            logger.debug(
                                "Failed to prune legacy cache entry %s after migration: %s",
                                legacy_cache_key,
                                exc,
                            )

        if not lookup:
            logger.debug("Advanced cache miss for key %s", cache_key)
            legacy_key = legacy_cache_key or cache_key
            legacy_results: Optional[List[Dict[str, Any]]] = None
            if legacy_key:
                try:
                    legacy_results = super()._get_cached_results(legacy_key, apply_ranking)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Legacy cache lookup failed for %s: %s", legacy_key, exc)
                    legacy_results = None

            if legacy_results is not None:
                logger.info("Recovered legacy cache entry for key %s via %s", cache_key, legacy_key)
                try:
                    status = "success" if legacy_results else "empty"
                    metadata = self._prepare_cache_metadata(len(legacy_results))
                    metadata["legacy_cache_key"] = legacy_key
                    metadata["migration_source"] = "legacy_file_cache"
                    expires_override: Optional[datetime] = None
                    grace_override: Optional[datetime] = None
                    legacy_path = self.cache_dir / f"{legacy_key}.json"
                    if legacy_path.exists():
                        try:
                            legacy_mtime = legacy_path.stat().st_mtime
                        except OSError as stat_exc:  # pragma: no cover - defensive logging
                            logger.debug(
                                "Failed to stat legacy cache file %s for TTL preservation: %s",
                                legacy_path,
                                stat_exc,
                            )
                        else:
                            expires_override = datetime.fromtimestamp(
                                legacy_mtime + float(self.cache_ttl),
                                tz=UTC,
                            )
                            grace_override = expires_override + self._cache_manager.ttl_calculator.grace_period  # type: ignore[union-attr]
                    self._cache_manager.set(  # type: ignore[union-attr]
                        cache_key,
                        legacy_results,
                        metadata=metadata,
                        status=status,
                        explicit_expiry=expires_override,
                        explicit_grace_expiry=grace_override,
                    )
                    lookup = self._cache_manager.get(cache_key, allow_stale=allow_stale)  # type: ignore[union-attr]
                    if lookup and self._prune_legacy_after_migration:
                        try:
                            self._cache_manager.prune_legacy_entry(legacy_key)  # type: ignore[union-attr]
                        except Exception as exc:  # pragma: no cover - defensive logging
                            logger.debug(
                                "Failed to prune legacy cache entry %s after legacy fallback migration: %s",
                                legacy_key,
                                exc,
                            )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed to migrate legacy cache entry %s to advanced cache %s: %s",
                        legacy_key,
                        cache_key,
                        exc,
                    )
                    lookup = None

            if not lookup:
                return None

        results = lookup.payload
        if isinstance(results, list) and apply_ranking and results:
            needs_ranking = any("ranking_score" not in result for result in results if isinstance(result, dict))
            if needs_ranking:
                logger.info("Recomputing ranking for cached results missing ranking scores")
                for result in results:
                    if isinstance(result, dict) and "ranking_score" not in result:
                        self._apply_study_ranking(result)
                results.sort(key=lambda r: r.get("ranking_score", 0) if isinstance(r, dict) else 0, reverse=True)
                try:
                    self._cache_manager.set(
                        cache_key,
                        results,
                        metadata=lookup.metadata,
                        status=str(lookup.metadata.get("status", "success")),
                        preserve_expiry=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Failed to persist updated ranking scores for %s: %s", cache_key, exc)

        if lookup.stale:
            logger.info("Serving stale cached results for key %s; scheduled for warming", cache_key)

        return results

    def _cache_results(self, cache_key: str, results: List[Dict]) -> None:  # type: ignore[override]
        if not self._advanced_cache_active():
            super()._cache_results(cache_key, results)
            return

        context = self._get_cache_context()
        status = "success" if results else "empty"
        metadata = self._prepare_cache_metadata(len(results))

        try:
            self._cache_manager.set(cache_key, results, metadata=metadata, status=status)  # type: ignore[union-attr]
            if (
                self._use_normalized_cache_keys
                and context.get("legacy_cache_key")
                and context.get("legacy_cache_key") != cache_key
            ):
                legacy_key = str(context.get("legacy_cache_key"))
                try:
                    legacy_metadata = dict(metadata)
                    legacy_metadata["cache_key"] = legacy_key
                    self._cache_manager.set(  # type: ignore[union-attr]
                        legacy_key,
                        results,
                        metadata=legacy_metadata,
                        status=status,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Failed to mirror cache entry to legacy key %s: %s", legacy_key, exc)
            elif self._mirror_legacy_root:
                try:
                    super()._cache_results(cache_key, results)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Failed to mirror cache entry to legacy root %s: %s", cache_key, exc)
            self._cache_manager.maybe_run_cleanup()  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to store advanced cache entry for %s: %s", cache_key, exc)
            super()._cache_results(cache_key, results)
        finally:
            self._clear_cache_context()

    # ------------------------------------------------------------------
    # Cache integrations and monitoring
    # ------------------------------------------------------------------

    def _prepare_cache_metadata(self, result_count: int) -> Dict[str, Any]:
        context = self._get_cache_context()
        metadata = {
            "original_query": context.get("original_query", ""),
            "enhanced_query": context.get("enhanced_query", context.get("original_query", "")),
            "parameters": context.get("parameters", {}),
            "result_count": result_count,
        }
        metadata["status"] = "empty" if result_count == 0 else "success"
        if "cache_key" in context:
            metadata["cache_key"] = context["cache_key"]
        if "legacy_cache_key" in context:
            metadata["legacy_cache_key"] = context["legacy_cache_key"]
        return metadata

    def get_cache_manager(self) -> Optional[NCBICacheManager]:
        return self._cache_manager

    def cache_status_report(self) -> Dict[str, Any]:
        if not self._advanced_cache_active():
            return {"enabled": False}
        manager = self._cache_manager  # keep local for mypy
        return {
            "enabled": True,
            "backend": manager.cache_backend,
            "directory": str(manager.cache_dir),
            "entry_directory": str(manager.advanced_cache_dir),
            "statistics": manager.get_statistics(),
            "warm_queue_size": manager.warming_scheduler.pending_count() if manager.cache_warming_enabled else 0,
        }

    def get_rate_limit_status(self) -> Optional[RateLimitStatus]:  # type: ignore[override]
        status = super().get_rate_limit_status()
        return status

    def combined_status_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "cache": self.cache_status_report(),
        }
        status = self.get_rate_limit_status()
        if isinstance(status, RateLimitStatus):
            report["rate_limit"] = asdict(status)
        else:
            report["rate_limit"] = None
        return report

    def warm_cache_entries(self) -> Dict[str, Dict[str, Any]]:
        if not self._advanced_cache_active():
            return {}
        return self._cache_manager.warmable_entries()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Batch and preload helpers
    # ------------------------------------------------------------------

    def _normalize_batch_request(
        self,
        request: BatchRequest,
        *,
        default_max_items: Optional[int],
        default_rank: Optional[bool],
        default_pharma_enhance: Optional[bool],
    ) -> Tuple[str, Optional[int], Optional[bool], Optional[bool]]:
        if isinstance(request, str):
            return request, default_max_items, default_rank, default_pharma_enhance
        if not isinstance(request, Mapping):
            raise TypeError("Batch requests must be strings or mappings with a 'query' field")
        if "query" not in request:
            raise ValueError("Batch request mappings must contain a 'query' key")
        query = str(request["query"])
        return (
            query,
            request.get("max_items", default_max_items),
            request.get("rank", default_rank),
            request.get("pharma_enhance", default_pharma_enhance),
        )

    @staticmethod
    def _make_batch_key(
        query: str,
        max_items: Optional[int],
        rank: Optional[bool],
        pharma_enhance: Optional[bool],
    ) -> Tuple[Any, ...]:
        return (
            query,
            max_items,
            rank,
            pharma_enhance,
        )

    def search_pubmed_batch(
        self,
        requests: Sequence[BatchRequest],
        *,
        max_items: Optional[int] = None,
        rank: Optional[bool] = None,
        pharma_enhance: Optional[bool] = None,
        reuse_cached_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute multiple PubMed queries with intelligent cache reuse."""

        if not requests:
            return []

        batch_results: List[Dict[str, Any]] = []
        dedup_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for request in requests:
            query, resolved_max_items, resolved_rank, resolved_pharma = self._normalize_batch_request(
                request,
                default_max_items=max_items,
                default_rank=rank,
                default_pharma_enhance=pharma_enhance,
            )
            dedup_key = self._make_batch_key(query, resolved_max_items, resolved_rank, resolved_pharma)

            if reuse_cached_results and dedup_key in dedup_cache:
                cached_info = dedup_cache[dedup_key]
                batch_results.append(
                    {
                        "query": query,
                        "results": cached_info["results"],
                        "from_cache": cached_info["from_cache"],
                        "reused": True,
                        "result_count": len(cached_info["results"]),
                    }
                )
                continue

            stats_before = self._cache_manager.get_statistics() if self._cache_manager else None
            payload = self.search_pubmed(
                query,
                max_items=resolved_max_items,
                rank=resolved_rank,
                pharma_enhance=resolved_pharma,
            )
            stats_after = self._cache_manager.get_statistics() if self._cache_manager else None
            from_cache = False
            if stats_before and stats_after:
                from_cache = stats_after["hits"] > stats_before["hits"]

            record = {
                "query": query,
                "results": payload,
                "from_cache": from_cache,
                "reused": False,
                "result_count": len(payload),
            }
            batch_results.append(record)

            if reuse_cached_results:
                dedup_cache[dedup_key] = {
                    "results": payload,
                    "from_cache": from_cache,
                }

        return batch_results

    def preload_cache(
        self,
        requests: Sequence[BatchRequest],
        *,
        max_items: Optional[int] = None,
        rank: Optional[bool] = None,
        pharma_enhance: Optional[bool] = None,
        reuse_cached_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Warm cache entries for a batch of queries without returning payloads."""

        batch_results = self.search_pubmed_batch(
            requests,
            max_items=max_items,
            rank=rank,
            pharma_enhance=pharma_enhance,
            reuse_cached_results=reuse_cached_results,
        )

        cache_active = self._advanced_cache_active()
        summaries: List[Dict[str, Any]] = []

        for entry in batch_results:
            if not cache_active:
                status = "executed"
            elif entry["from_cache"]:
                status = "hit"
            elif entry["reused"]:
                status = "reused"
            else:
                status = "stored"

            summaries.append(
                {
                    "query": entry["query"],
                    "status": status,
                    "result_count": entry["result_count"],
                    "from_cache": entry["from_cache"],
                    "reused": entry["reused"],
                }
            )

        return summaries

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return combined cache and rate limiting analytics."""

        metrics: Dict[str, Any] = {}
        cache_stats: Optional[Dict[str, int]]
        if self._cache_manager:
            cache_stats = self._cache_manager.get_statistics()
            metrics["cache"] = cache_stats
            total_events = cache_stats["hits"] + cache_stats["stale_hits"] + cache_stats["misses"]
            metrics["cache_hit_rate"] = (
                (cache_stats["hits"] + cache_stats["stale_hits"]) / total_events if total_events else None
            )
            metrics["cache_miss_rate"] = (cache_stats["misses"] / total_events if total_events else None)
            metrics["cache_rate_limit_skips"] = cache_stats["skipped_rate_limit"]
            metrics["cache_write_count"] = cache_stats["writes"]
            metrics["cache_eviction_count"] = cache_stats["evictions"]
        else:
            cache_stats = None
            metrics["cache"] = None
            metrics["cache_hit_rate"] = None
            metrics["cache_miss_rate"] = None
            metrics["cache_rate_limit_skips"] = None
            metrics["cache_write_count"] = None
            metrics["cache_eviction_count"] = None

        status = self.get_rate_limit_status()
        metrics["rate_limit"] = asdict(status) if isinstance(status, RateLimitStatus) else None
        if metrics["rate_limit"] is not None:
            metrics["rate_limit"]["skipped_due_to_cache"] = metrics["cache_rate_limit_skips"] or 0

        return metrics
