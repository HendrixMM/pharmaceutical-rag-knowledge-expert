"""Advanced caching layer integrating NCBI compliance, metadata, and analytics.

This module introduces :class:`NCBICacheManager`, a drop-in replacement for the
legacy caching helpers inside ``pubmed_scraper`` with richer metadata tracking,
policy enforcement, and monitoring.  The design is intentionally backend-agnostic;
the initial implementation ships a file-based backend while exposing hooks for
future alternatives (for example, Redis or in-memory caches).
"""
from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import threading
from functools import partial
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .utils.cache_utils import (
    CacheAnalytics,
    CacheCleanupScheduler,
    CacheCompressionUtils,
    CacheConfigValidator,
    CacheExportImportUtils,
    CacheKeyNormalizer,
    CacheMetadataExtractor,
    CacheOptimizer,
    CacheTTLCalculator,
    CacheValidationUtils,
    CacheWarmingScheduler,
    PharmaceuticalCacheOptimizer,
    RateLimitCacheIntegrator,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric") from exc
    if value < 0:
        raise ValueError(f"Environment variable {name} must be non-negative")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"Environment variable {name} must be non-negative")
    return value


@dataclass(frozen=True)
class CacheLookupResult:
    payload: Any
    metadata: Mapping[str, Any]
    stale: bool


class CacheBackendNotSupported(RuntimeError):
    """Raised when a requested cache backend is not available."""


class NCBICacheManager:
    """Manage PubMed cache entries with metadata-rich bookkeeping.

    The manager focuses on three pillars:

    * NCBI compliance – enforcing 24-hour TTL guidance by default
    * Observability – collecting hit/miss metrics plus cache warming hints
    * Extensibility – simple file format compatible with future backends
    """

    SCHEMA_VERSION = "1"

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        cache_backend: str = "file",
        enable_metadata: bool = True,
        enable_statistics: bool = True,
        enable_cleanup: bool = True,
        cleanup_interval_hours: float = 6,
        default_ttl_hours: float = 24,
        grace_period_hours: float = 2,
        empty_results_ttl_minutes: float = 30,
        error_results_ttl_minutes: float = 5,
        cache_max_size_mb: float = 1000,
        cache_max_entries: int = 10_000,
        compression_enabled: bool = True,
        cache_warming_enabled: bool = False,
        rate_limit_integration: bool = True,
        pharmaceutical_optimization: bool = True,
        pharma_ttl_bonus_hours: float = 12.0,
        cache_allow_stale_within_grace: Optional[bool] = None,
        cache_write_on_access: bool = True,
        cleanup_run_on_start: bool = False,
        cache_pretty_json: bool = False,
        strict_ncbi_ttl: bool = False,
        cleanup_daemon_enabled: bool = False,
        rate_limiter: Optional[Any] = None,  # Optional rate limiter instance
    ) -> None:
        self.cache_backend = cache_backend
        if cache_backend != "file":
            raise CacheBackendNotSupported("Only the 'file' backend is currently implemented")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.advanced_cache_dir = self.cache_dir / "advanced"
        self.advanced_cache_dir.mkdir(parents=True, exist_ok=True)

        self.enable_metadata = enable_metadata
        self.enable_statistics = enable_statistics
        self.enable_cleanup = enable_cleanup
        self.compression_enabled = compression_enabled
        self.cache_warming_enabled = cache_warming_enabled
        self.rate_limit_integration = rate_limit_integration
        self.pharmaceutical_optimization = pharmaceutical_optimization
        if cache_allow_stale_within_grace is None:
            cache_allow_stale_within_grace = _env_flag("CACHE_ALLOW_STALE_WITHIN_GRACE", "true")
        self.cache_allow_stale_within_grace = cache_allow_stale_within_grace
        self.cache_write_on_access = cache_write_on_access
        self.cache_pretty_json = cache_pretty_json
        self.strict_ncbi_ttl = strict_ncbi_ttl

        self.cache_max_size_bytes = int(cache_max_size_mb * 1024 * 1024)
        self.cache_max_entries = int(cache_max_entries)

        self._lock = threading.RLock()

        self.analytics = CacheAnalytics() if enable_statistics else None
        self._rate_limit_integrator = (
            RateLimitCacheIntegrator(self.analytics) if (self.analytics and rate_limit_integration) else None
        )
        self.ttl_calculator = CacheTTLCalculator(
            default_ttl_hours=default_ttl_hours,
            grace_period_hours=grace_period_hours,
            empty_results_ttl_minutes=empty_results_ttl_minutes,
            error_results_ttl_minutes=error_results_ttl_minutes,
            pharma_ttl_bonus_hours=pharma_ttl_bonus_hours,
            strict_ncbi_ttl=self.strict_ncbi_ttl,
        )
        self.optimizer = CacheOptimizer()
        self.warming_scheduler = CacheWarmingScheduler(rate_limiter=rate_limiter)
        self.cleanup_scheduler = CacheCleanupScheduler(cleanup_interval_hours, run_on_start=cleanup_run_on_start)
        self.pharma_optimizer = PharmaceuticalCacheOptimizer()
        self.cleanup_daemon_enabled = bool(cleanup_daemon_enabled and self.enable_cleanup)
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_atexit_registered = False

        if self.cleanup_daemon_enabled:
            self.start_cleanup_daemon()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "NCBICacheManager":
        """Create a cache manager from environment configuration."""
        cache_dir = os.getenv("PUBMED_CACHE_DIR", "./pubmed_cache")
        config: Dict[str, Any] = {
            "cache_backend": os.getenv("CACHE_BACKEND", "file"),
            "enable_metadata": _env_flag("CACHE_METADATA_ENABLED", "true"),
            "enable_statistics": _env_flag("CACHE_STATISTICS_ENABLED", "true"),
            "enable_cleanup": _env_flag("CACHE_CLEANUP_ENABLED", "true"),
            "cleanup_interval_hours": _env_float("CACHE_CLEANUP_INTERVAL_HOURS", 6.0),
            "default_ttl_hours": _env_float("NCBI_CACHE_TTL_HOURS", 24.0),
            "grace_period_hours": _env_float("NCBI_CACHE_GRACE_PERIOD_HOURS", 2.0),
            "empty_results_ttl_minutes": _env_float("CACHE_EMPTY_RESULTS_TTL_MINUTES", 30.0),
            "error_results_ttl_minutes": _env_float("CACHE_ERROR_RESULTS_TTL_MINUTES", 5.0),
            "cache_max_size_mb": _env_float("CACHE_MAX_SIZE_MB", 1000.0),
            "cache_max_entries": _env_int("CACHE_MAX_ENTRIES", 10_000),
            "compression_enabled": _env_flag("CACHE_COMPRESSION_ENABLED", "true"),
            "cache_warming_enabled": _env_flag("CACHE_WARMING_ENABLED", "false"),
            "rate_limit_integration": _env_flag("CACHE_RATE_LIMIT_INTEGRATION", "true"),
            "pharmaceutical_optimization": _env_flag("CACHE_PHARMACEUTICAL_OPTIMIZATION", "true"),
            "pharma_ttl_bonus_hours": _env_float("PHARMA_TTL_BONUS_HOURS", 12.0),
            "cache_allow_stale_within_grace": _env_flag("CACHE_ALLOW_STALE_WITHIN_GRACE", "true"),
            "cache_write_on_access": _env_flag("CACHE_WRITE_ON_ACCESS", "true"),
            "cache_pretty_json": _env_flag("CACHE_PRETTY_JSON", "false"),
            "strict_ncbi_ttl": _env_flag("STRICT_NCBI_TTL", "false"),
            "cleanup_daemon_enabled": _env_flag("CACHE_CLEANUP_DAEMON_ENABLED", "false"),
        }
        CacheConfigValidator.validate(
            {
                "default_ttl_hours": config["default_ttl_hours"],
                "grace_period_hours": config["grace_period_hours"],
                "empty_results_ttl_minutes": config["empty_results_ttl_minutes"],
                "error_results_ttl_minutes": config["error_results_ttl_minutes"],
                "cache_max_size_mb": config["cache_max_size_mb"],
                "cache_max_entries": config["cache_max_entries"],
                "cache_backend": config["cache_backend"],
                "pharma_ttl_bonus_hours": config["pharma_ttl_bonus_hours"],
            }
        )
        return cls(cache_dir, **config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_cache_key(self, query: str, params: Mapping[str, Any], *, version: str | int = "1") -> str:
        return CacheKeyNormalizer.normalize(query, params, version=version or self.SCHEMA_VERSION)

    def get(self, cache_key: str, *, allow_stale: bool = True) -> Optional[CacheLookupResult]:
        """Return cached payload for ``cache_key`` if available."""
        with self._lock:
            path = self._entry_path(cache_key)
            entry = self._read_entry(path)
            if not entry:
                self._record_miss()
                return None

            stale = self._is_stale(entry)
            grace_expired = self._is_grace_expired(entry)
            if grace_expired:
                logger.debug(
                    "Cache entry for %s expired beyond grace window; evicting.",
                    cache_key,
                )
                self._evict_path(path)
                self._record_miss()
                return None
            if stale and not allow_stale:
                logger.debug(
                    "Cache entry for %s stale but stale results disallowed; treating as miss.",
                    cache_key,
                )
                if self.cache_warming_enabled:
                    metadata = dict(entry["metadata"]) if self.enable_metadata else {}
                    self.warming_scheduler.schedule(cache_key, metadata)
                self._record_miss()
                return None

            metadata = dict(entry["metadata"]) if self.enable_metadata else {}
            payload = CacheCompressionUtils.decompress_json(entry["payload"], entry["payload_encoding"])

            if self.cache_write_on_access:
                entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
                entry["last_accessed_at"] = datetime.now(UTC).isoformat()
                self._write_entry(path, entry)
            if stale:
                if self.cache_warming_enabled:
                    self.warming_scheduler.schedule(cache_key, metadata)
                self._record_hit(stale=True)
                if self._rate_limit_integrator:
                    self._rate_limit_integrator.notify_cache_hit(stale=True)
            else:
                self._record_hit(stale=False)
                if self._rate_limit_integrator:
                    self._rate_limit_integrator.notify_cache_hit(stale=False)

            return CacheLookupResult(payload=payload, metadata=metadata, stale=stale)

    async def aget(self, cache_key: str, *, allow_stale: bool = True) -> Optional[CacheLookupResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.get, cache_key, allow_stale=allow_stale))

    def set(
        self,
        cache_key: str,
        payload: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        status: str = "success",
        preserve_expiry: bool = False,
        explicit_expiry: Optional[datetime] = None,
        explicit_grace_expiry: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            now = datetime.now(UTC)

            path = self._entry_path(cache_key)
            existing_entry: Optional[Dict[str, Any]] = None
            if preserve_expiry or explicit_expiry or explicit_grace_expiry:
                existing_entry = self._read_entry(path)

            if self.enable_metadata:
                base_metadata = dict(metadata or {})
                results_payload: Optional[Sequence[Any]] = None
                if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                    results_payload = payload  # type: ignore[assignment]

                computed_metadata = CacheMetadataExtractor.extract(
                    original_query=base_metadata.get("original_query", ""),
                    enhanced_query=base_metadata.get("enhanced_query", base_metadata.get("original_query", "")),
                    parameters=base_metadata.get("parameters", {}),
                    results=results_payload,
                    status=status,
                )

                # Preserve any custom fields supplied by callers while keeping the canonical keys.
                for key, value in base_metadata.items():
                    if key not in computed_metadata:
                        computed_metadata[key] = value
                metadata_payload = computed_metadata
                if self.pharmaceutical_optimization:
                    metadata_payload = self.pharma_optimizer.enrich_metadata(metadata_payload)
            else:
                metadata_payload = {}

            ttl, grace = self.ttl_calculator.calculate(metadata_payload)
            expires_at = now + ttl
            grace_expires_at = expires_at + grace

            if explicit_expiry:
                if explicit_expiry.tzinfo is None:
                    explicit_expiry = explicit_expiry.replace(tzinfo=UTC)
                expires_at = explicit_expiry
            if explicit_grace_expiry:
                if explicit_grace_expiry.tzinfo is None:
                    explicit_grace_expiry = explicit_grace_expiry.replace(tzinfo=UTC)
                grace_expires_at = explicit_grace_expiry
            elif explicit_expiry:
                grace_expires_at = expires_at + grace

            if preserve_expiry and existing_entry and not (explicit_expiry or explicit_grace_expiry):
                expires_at = datetime.fromisoformat(existing_entry["expires_at"])
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                grace_expires_at = datetime.fromisoformat(existing_entry["grace_expires_at"])
                if grace_expires_at.tzinfo is None:
                    grace_expires_at = grace_expires_at.replace(tzinfo=UTC)

            encoded_payload, encoding = CacheCompressionUtils.compress_json(payload, self.compression_enabled)
            entry = {
                "schema_version": self.SCHEMA_VERSION,
                "created_at": existing_entry.get("created_at") if existing_entry else now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "grace_expires_at": grace_expires_at.isoformat(),
                "hit_count": existing_entry.get("hit_count", 0) if existing_entry else 0,
                "payload": encoded_payload,
                "payload_encoding": encoding,
                "metadata": metadata_payload,
            }
            if existing_entry and "last_accessed_at" in existing_entry:
                entry["last_accessed_at"] = existing_entry["last_accessed_at"]
            else:
                entry["last_accessed_at"] = now.isoformat()
            CacheValidationUtils.validate_entry(entry)

            self._write_entry(path, entry)
            self._record_write()
            self._enforce_limits()

    async def aset(
        self,
        cache_key: str,
        payload: Any,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        status: str = "success",
        preserve_expiry: bool = False,
        explicit_expiry: Optional[datetime] = None,
        explicit_grace_expiry: Optional[datetime] = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.set,
                cache_key,
                payload,
                metadata=metadata,
                status=status,
                preserve_expiry=preserve_expiry,
                explicit_expiry=explicit_expiry,
                explicit_grace_expiry=explicit_grace_expiry,
            ),
        )

    def update_cache_stats(self, *, hit: bool = False, miss: bool = False) -> None:
        """Compatibility helper for legacy code paths."""
        if hit:
            self._record_hit(stale=False)
        if miss:
            self._record_miss()

    def cleanup_expired_entries(self) -> int:
        """Remove entries whose grace period has elapsed."""
        removed = 0
        for path, entry in self._iter_entries():
            if self._is_grace_expired(entry):
                self._evict_path(path)
                removed += 1
        if removed:
            logger.info("Removed %s expired cache entries", removed)
        return removed

    def maybe_run_cleanup(self) -> int:
        """Run cleanup if the configured interval has elapsed."""
        if not self.enable_cleanup:
            return 0
        if not self.cleanup_scheduler.should_run():
            return 0
        removed = self.cleanup_expired_entries()
        self.cleanup_scheduler.mark_ran()
        return removed

    def warmable_entries(self) -> Dict[str, Dict[str, Any]]:
        """Get warmable entries using rate limiter optimal timing.

        Returns entries marked as 'due' for warming based on:
        - Rate limiter optimal timing windows
        - Entry priority (pharmaceutical queries, result count, recency)
        - Current rate limit status

        Defer decisions are kept for future warming cycles.
        """
        if hasattr(self.warming_scheduler, 'get_warming_schedule'):
            # Use new rate limiter-aware timing method
            schedule = self.warming_scheduler.get_warming_schedule()
            due_entries = {}

            # Extract entries that are due now
            for cache_key, entry_info in schedule.items():
                if entry_info.get('due_now', False):
                    due_entries[cache_key] = entry_info['metadata']

            if due_entries and self.analytics:
                self.analytics.record_warming(len(due_entries))

            # Log detailed decision summary
            due_count = len(due_entries)
            defer_count = sum(1 for info in schedule.values() if not info.get('due_now', False))
            logger.debug(
                "Cache warming decisions - Due: %d, Defer: %d, Total: %d",
                due_count,
                defer_count,
                len(schedule)
            )

            # Update decision summary for compatibility with existing tests
            if hasattr(self.warming_scheduler, '_decisions'):
                with self.warming_scheduler._lock:
                    # Clear old decisions
                    self.warming_scheduler._decisions.clear()
                    # Add new decisions based on timing results
                    for cache_key, info in schedule.items():
                        from src.utils.cache_utils import WarmingDecision
                        decision = "due" if info.get('due_now', False) else "defer"
                        reason = info.get('reason', 'Unknown')
                        priority = info.get('priority', 0)
                        defer_until = info.get('defer_until')
                        suggested_defer_time = (
                            (defer_until - datetime.now(UTC)).total_seconds()
                            if defer_until else None
                        )

                        self.warming_scheduler._decisions[cache_key] = WarmingDecision(
                            cache_key=cache_key,
                            metadata=info['metadata'],
                            decision=decision,
                            reason=reason,
                            priority=priority,
                            suggested_defer_time=suggested_defer_time
                        )

            # Log sample of deferred entries for debugging
            if defer_count > 0 and logger.isEnabledFor(logging.DEBUG):
                deferred_samples = [
                    (key, info.get('reason', 'unknown'), info.get('defer_until'))
                    for key, info in list(schedule.items())[:3]
                    if not info.get('due_now', False)
                ]
                for key, reason, defer_time in deferred_samples:
                    if defer_time:
                        logger.debug("Deferred entry %s: %s until %s", key, reason, defer_time)
                    else:
                        logger.debug("Deferred entry %s: %s", key, reason)

            return due_entries
        elif hasattr(self.warming_scheduler, 'pop_due_entries'):
            # Fallback to older rate limiter-aware method
            due_entries = self.warming_scheduler.pop_due_entries()
            if due_entries and self.analytics:
                self.analytics.record_warming(len(due_entries))

            # Log decision summary if available
            if hasattr(self.warming_scheduler, 'get_decision_summary'):
                summary = self.warming_scheduler.get_decision_summary()
                logger.debug(
                    "Cache warming decisions - Due: %d, Defer: %d, Skip: %d",
                    summary.get('due', 0),
                    summary.get('defer', 0),
                    summary.get('skip', 0)
                )

            return due_entries
        else:
            # Fallback to legacy behavior
            pending = self.warming_scheduler.pop_all()
            if pending and self.analytics:
                self.analytics.record_warming(len(pending))
            return pending

    def start_cleanup_daemon(self) -> None:
        """Start the background cleanup daemon when enabled."""
        if not self.cleanup_daemon_enabled:
            return
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        self._cleanup_stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_daemon_loop,
            name="ncbi-cache-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()
        if not self._cleanup_atexit_registered:
            atexit.register(self.stop_cleanup_daemon, wait=False)
            self._cleanup_atexit_registered = True

    def stop_cleanup_daemon(self, *, wait: bool = True) -> None:
        """Signal the cleanup daemon to stop and optionally wait for completion."""
        if not self._cleanup_thread:
            return
        self._cleanup_stop_event.set()
        if wait and self._cleanup_thread.is_alive():
            self._cleanup_thread.join()
        self._cleanup_thread = None

    def _cleanup_daemon_loop(self) -> None:
        interval = max(self.cleanup_scheduler.interval.total_seconds(), 1.0)
        while not self._cleanup_stop_event.wait(interval):
            try:
                self.maybe_run_cleanup()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Background cleanup execution failed: %s", exc)

    def prune_legacy_entry(self, cache_key: str) -> bool:
        """Remove the legacy cache file associated with ``cache_key`` if present."""
        legacy_path = self.cache_dir / f"{cache_key}.json"
        if not legacy_path.exists():
            return False
        try:
            legacy_path.unlink()
            if self.analytics:
                self.analytics.record_eviction()
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to prune legacy cache entry %s: %s", cache_key, exc)
            return False

    def get_statistics(self) -> Dict[str, int]:
        if not self.analytics:
            return {
                "hits": 0,
                "stale_hits": 0,
                "misses": 0,
                "writes": 0,
                "evictions": 0,
                "warmed": 0,
                "skipped_rate_limit": 0,
                "exported": 0,
                "imported": 0,
            }
        return self.analytics.snapshot()

    def export_cache(self, destination: Path | str) -> int:
        path = Path(destination)
        count = CacheExportImportUtils.export_entries(self.advanced_cache_dir, path)
        if self.analytics:
            self.analytics.record_export(count)
        logger.info("Exported %s cache entries to %s", count, path)
        return count

    def import_cache(self, source: Path | str) -> int:
        path = Path(source)
        count = CacheExportImportUtils.import_entries(path, self.advanced_cache_dir, target_schema_version=self.SCHEMA_VERSION)
        if self.analytics:
            self.analytics.record_import(count)
        logger.info("Imported %s cache entries from %s", count, path)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry_path(self, cache_key: str) -> Path:
        return self.advanced_cache_dir / f"{cache_key}.json"

    def _read_entry(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                entry = json.load(fh)
            CacheValidationUtils.validate_entry(entry)
            return entry
        except Exception as exc:
            logger.warning("Failed to read cache entry %s: %s", path.name, exc)
            self._evict_path(path)
            return None

    def _write_entry(self, path: Path, entry: Mapping[str, Any]) -> None:
        tmp_path = path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            if self.cache_pretty_json:
                json.dump(entry, fh, ensure_ascii=False, indent=2)
            else:
                json.dump(entry, fh, ensure_ascii=False, separators=(',', ':'))
        tmp_path.replace(path)

    def _evict_path(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Failed to remove cache entry %s: %s", path, exc)
        else:
            self._record_eviction()

    def _iter_entries(self) -> Iterable[Tuple[Path, Dict[str, Any]]]:
        for entry_path in self.advanced_cache_dir.glob("*.json"):
            entry = self._read_entry(entry_path)
            if entry:
                yield entry_path, entry

    def _is_expired(self, entry: Mapping[str, Any]) -> bool:
        expires_at = datetime.fromisoformat(entry["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        return datetime.now(UTC) >= expires_at

    def _is_stale(self, entry: Mapping[str, Any]) -> bool:
        return self._is_expired(entry)

    def _is_grace_expired(self, entry: Mapping[str, Any]) -> bool:
        grace_expires_at = datetime.fromisoformat(entry["grace_expires_at"])
        if grace_expires_at.tzinfo is None:
            grace_expires_at = grace_expires_at.replace(tzinfo=UTC)
        return datetime.now(UTC) >= grace_expires_at

    def _enforce_limits(self) -> None:
        entries = list(self._iter_entries())
        if len(entries) <= self.cache_max_entries and self._total_size(entries) <= self.cache_max_size_bytes:
            return
        to_remove = self.optimizer.suggest_evictions(
            entries,
            max_entries=self.cache_max_entries,
            max_size_bytes=self.cache_max_size_bytes,
        )
        for path in to_remove:
            self._evict_path(path)

    @staticmethod
    def _total_size(entries: Sequence[Tuple[Path, Mapping[str, Any]]]) -> int:
        total = 0
        for path, _entry in entries:
            if path.exists():
                total += path.stat().st_size
        return total

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def _record_hit(self, *, stale: bool) -> None:
        if self.analytics:
            self.analytics.record_hit(stale=stale)

    def _record_miss(self) -> None:
        if self.analytics:
            self.analytics.record_miss()

    def _record_write(self) -> None:
        if self.analytics:
            self.analytics.record_write()

    def _record_eviction(self) -> None:
        if self.analytics:
            self.analytics.record_eviction()


__all__ = [
    "CacheBackendNotSupported",
    "CacheLookupResult",
    "NCBICacheManager",
]
