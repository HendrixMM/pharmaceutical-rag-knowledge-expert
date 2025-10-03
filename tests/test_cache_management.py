import asyncio
import json
import threading
import types
from datetime import UTC, datetime, timedelta

import pytest

from src.cache_management import CacheLookupResult, NCBICacheManager
from src.rate_limiting import RateLimitStatus
from src.utils.cache_utils import CacheKeyNormalizer


@pytest.fixture
def frozen_datetime(monkeypatch):
    class FrozenDateTime(datetime):
        _now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        @classmethod
        def utcnow(cls):
            return cls._now

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls._now
            return cls._now.astimezone(tz)

        @classmethod
        def advance(cls, **kwargs):
            cls._now += timedelta(**kwargs)

    monkeypatch.setattr("src.cache_management.datetime", FrozenDateTime)
    return FrozenDateTime


@pytest.fixture
def cache_manager(tmp_path, frozen_datetime):
    manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=1 / 60,  # 1 minute default TTL for tests
        grace_period_hours=1 / 60,  # 1 minute grace period
        empty_results_ttl_minutes=1,
        error_results_ttl_minutes=0.5,
        cache_max_entries=100,
        cache_max_size_mb=10,
        cache_warming_enabled=True,
    )
    return manager


def test_cache_key_normalization_is_stable():
    key1 = CacheKeyNormalizer.normalize("Test Query", {"order": 1, "flag": True})
    key2 = CacheKeyNormalizer.normalize(" test   query ", {"flag": True, "order": 1})
    assert key1 == key2


def test_cache_round_trip(cache_manager):
    payload = [{"title": "primary"}]
    cache_manager.set(
        "key-1",
        payload,
        metadata={
            "original_query": "primary",
            "enhanced_query": "primary",
            "parameters": {"max_items": 5},
        },
    )

    result = cache_manager.get("key-1")
    assert isinstance(result, CacheLookupResult)
    assert result.payload == payload
    assert result.metadata["result_count"] == 1
    stats = cache_manager.get_statistics()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_cache_stale_within_grace_period(cache_manager, frozen_datetime):
    payload = [{"title": "stale"}]
    cache_manager.set("key-2", payload)

    frozen_datetime.advance(seconds=90)
    lookup = cache_manager.get("key-2")
    assert isinstance(lookup, CacheLookupResult)
    assert lookup.payload == payload
    assert lookup.stale is True
    # Entry should be scheduled for warming when stale hits occur.
    assert cache_manager.warming_scheduler.pending_count() == 1


def test_warmable_entries_updates_analytics(cache_manager):
    # Schedule with high priority metadata to ensure it's marked as due
    cache_manager.warming_scheduler.schedule("warm-key", {"pharmaceutical_focus": True})
    warmed_before = cache_manager.get_statistics()["warmed"]
    pending = cache_manager.warmable_entries()
    assert "warm-key" in pending
    warmed_after = cache_manager.get_statistics()["warmed"]
    assert warmed_after == warmed_before + 1


def test_cache_miss_after_grace(cache_manager, frozen_datetime, tmp_path):
    payload = [{"title": "expire"}]
    cache_manager.set("key-3", payload)

    frozen_datetime.advance(minutes=3)
    # Grace period elapsed (TTL + grace == 2 minutes)
    assert cache_manager.get("key-3") is None
    assert not (cache_manager.advanced_cache_dir / "key-3.json").exists()


def test_cleanup_removes_expired_entries(cache_manager, frozen_datetime):
    cache_manager.set("key-4", [])

    frozen_datetime.advance(minutes=5)
    removed = cache_manager.cleanup_expired_entries()
    assert removed == 1
    assert cache_manager.get_statistics()["evictions"] >= 1


def test_cache_export_import(cache_manager, tmp_path):
    cache_manager.set(
        "key-5",
        [{"title": "export"}],
        metadata={"original_query": "export", "parameters": {}},
    )
    export_dir = tmp_path / "export"
    exported = cache_manager.export_cache(export_dir)
    assert exported == 1

    imported = cache_manager.import_cache(export_dir)
    assert imported == 1
    assert (cache_manager.advanced_cache_dir / "key-5.json").exists()


def test_async_wrappers_preserve_kwargs(cache_manager):
    payload = [{"title": "async"}]

    async def exercise() -> None:
        await cache_manager.aset(
            "async-key",
            payload,
            metadata={"original_query": "async", "parameters": {}},
            status="success",
        )

        lookup = await cache_manager.aget("async-key", allow_stale=False)
        assert isinstance(lookup, CacheLookupResult)
        assert lookup.payload == payload

    asyncio.run(exercise())


def test_pharma_ttl_bonus_configurable(tmp_path, monkeypatch):
    # No bonus configured
    zero_dir = tmp_path / "zero"
    monkeypatch.setenv("PUBMED_CACHE_DIR", str(zero_dir))
    monkeypatch.setenv("PHARMA_TTL_BONUS_HOURS", "0")
    manager_zero = NCBICacheManager.from_env()
    metadata = {
        "status": "success",
        "result_count": 1,
        "pharmaceutical_focus": True,
    }
    ttl_zero, _ = manager_zero.ttl_calculator.calculate(metadata)
    assert ttl_zero == manager_zero.ttl_calculator.default_ttl

    # Bonus enabled
    bonus_dir = tmp_path / "bonus"
    monkeypatch.setenv("PUBMED_CACHE_DIR", str(bonus_dir))
    monkeypatch.setenv("PHARMA_TTL_BONUS_HOURS", "24")
    manager_bonus = NCBICacheManager.from_env()
    ttl_bonus, _ = manager_bonus.ttl_calculator.calculate(metadata)
    assert ttl_bonus >= manager_bonus.ttl_calculator.default_ttl + timedelta(hours=24)


def test_cache_write_on_access_disabled_preserves_file_mtime(tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_write_on_access=False,  # Disable write on access
        default_ttl_hours=24,
        grace_period_hours=2,
    )

    # Set initial cache entry
    payload = [{"title": "no_write_test"}]
    cache_manager.set("test-key", payload)

    # Get the file path and initial mtime
    cache_file = cache_manager.advanced_cache_dir / "test-key.json"
    assert cache_file.exists()
    initial_mtime = cache_file.stat().st_mtime

    # Read from cache - this should not modify the file
    result = cache_manager.get("test-key")
    assert result is not None
    assert result.payload == payload

    # Verify file mtime is unchanged
    final_mtime = cache_file.stat().st_mtime
    assert final_mtime == initial_mtime, "File mtime should not change when cache_write_on_access is disabled"

    # Verify stats are still recorded despite no file write
    stats = cache_manager.get_statistics()
    assert stats["hits"] == 1


def test_cache_write_on_access_enabled_updates_file_mtime(tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_write_on_access=True,  # Enable write on access (default)
        default_ttl_hours=24,
        grace_period_hours=2,
    )

    # Set initial cache entry
    payload = [{"title": "write_test"}]
    cache_manager.set("test-key-2", payload)

    # Get the file path and initial mtime
    cache_file = cache_manager.advanced_cache_dir / "test-key-2.json"
    assert cache_file.exists()
    initial_mtime = cache_file.stat().st_mtime

    # Small delay to ensure mtime would be different if file is modified
    import time

    time.sleep(0.1)

    # Read from cache - this should modify the file
    result = cache_manager.get("test-key-2")
    assert result is not None
    assert result.payload == payload

    # Verify file mtime is changed
    final_mtime = cache_file.stat().st_mtime
    assert final_mtime > initial_mtime, "File mtime should change when cache_write_on_access is enabled"

    # Verify stats are recorded
    stats = cache_manager.get_statistics()
    assert stats["hits"] == 1


def test_eviction_prefers_least_recently_used(tmp_path):
    manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_max_entries=2,
        cache_max_size_mb=1000,
    )

    payload = [{"title": "entry"}]
    manager.set("a", payload)
    manager.set("b", payload)
    assert manager.get("a") is not None
    manager.set("c", payload)

    remaining = {path.stem for path in manager.advanced_cache_dir.glob("*.json")}
    assert "a" in remaining
    assert "c" in remaining
    assert "b" not in remaining


def test_strict_ttl_clamps_short_values(tmp_path):
    manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=24,
        grace_period_hours=2,
        empty_results_ttl_minutes=5,
        error_results_ttl_minutes=10,
        strict_ncbi_ttl=True,
    )

    empty_ttl, _ = manager.ttl_calculator.calculate({"status": "success", "result_count": 0})
    error_ttl, _ = manager.ttl_calculator.calculate({"status": "error", "result_count": 5})

    assert empty_ttl >= timedelta(hours=24)
    assert error_ttl >= timedelta(hours=24)


def test_cleanup_daemon_runs_when_enabled(tmp_path):
    manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=1,
        grace_period_hours=1,
        cleanup_interval_hours=1 / (3600 * 10),  # roughly one-tenth of a second
        cleanup_daemon_enabled=True,
    )

    manager.stop_cleanup_daemon()

    triggered = threading.Event()

    def fake_cleanup(self) -> int:
        triggered.set()
        return 0

    manager.cleanup_expired_entries = types.MethodType(fake_cleanup, manager)
    manager.start_cleanup_daemon()

    try:
        assert triggered.wait(timeout=2.0), "Cleanup daemon did not trigger within expected timeframe"
    finally:
        manager.stop_cleanup_daemon()


def test_cleanup_scheduler_run_on_start_disabled(tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cleanup_run_on_start=False,  # Disable run on start
        cleanup_interval_hours=1,
    )

    # Verify that cleanup should not run initially
    assert not cache_manager.cleanup_scheduler.should_run(), "Cleanup should not run on start when run_on_start=False"


def test_cleanup_scheduler_run_on_start_enabled(tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cleanup_run_on_start=True,  # Enable run on start
        cleanup_interval_hours=1,
    )

    # Verify that cleanup should run initially
    assert cache_manager.cleanup_scheduler.should_run(), "Cleanup should run on start when run_on_start=True"


def test_eviction_logic_max_entries_limit(tmp_path, frozen_datetime):
    # Set a low max entries limit for testing
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_max_entries=3,  # Only allow 3 entries
        cache_max_size_mb=1000,  # Large size limit to focus on entry count
        default_ttl_hours=24,
        grace_period_hours=2,
    )

    # Create entries with deterministic timestamps
    entries_data = [
        ("oldest", "2024-01-01T10:00:00Z"),
        ("middle", "2024-01-01T11:00:00Z"),
        ("newest", "2024-01-01T12:00:00Z"),
    ]

    for i, (name, timestamp) in enumerate(entries_data):
        frozen_datetime._now = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        cache_manager.set(f"key-{name}", [{"title": f"entry-{name}"}])

    # Verify all 3 entries exist
    assert len(list(cache_manager.advanced_cache_dir.glob("*.json"))) == 3

    # Add a 4th entry which should trigger eviction of the oldest
    frozen_datetime._now = datetime.fromisoformat("2024-01-01T13:00:00+00:00")
    cache_manager.set("key-fourth", [{"title": "entry-fourth"}])

    # Verify only 3 entries remain and the oldest was evicted
    remaining_files = list(cache_manager.advanced_cache_dir.glob("*.json"))
    assert len(remaining_files) == 3

    # Verify the oldest entry was evicted
    assert not (cache_manager.advanced_cache_dir / "key-oldest.json").exists()
    assert (cache_manager.advanced_cache_dir / "key-middle.json").exists()
    assert (cache_manager.advanced_cache_dir / "key-newest.json").exists()
    assert (cache_manager.advanced_cache_dir / "key-fourth.json").exists()

    # Verify eviction was recorded in analytics
    stats = cache_manager.get_statistics()
    assert stats["evictions"] >= 1


def test_eviction_logic_max_size_limit(tmp_path, frozen_datetime):
    # Set a very low size limit for testing
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_max_entries=100,  # Large entry limit to focus on size
        cache_max_size_mb=0.001,  # 1KB limit to force evictions
        default_ttl_hours=24,
        grace_period_hours=2,
    )

    # Create large payloads that will exceed size limit
    large_payload_1 = [{"title": "large1", "data": "x" * 500}]  # ~500 bytes
    large_payload_2 = [{"title": "large2", "data": "y" * 500}]  # ~500 bytes
    large_payload_3 = [{"title": "large3", "data": "z" * 500}]  # ~500 bytes

    # Create entries with deterministic timestamps
    frozen_datetime._now = datetime.fromisoformat("2024-01-01T10:00:00+00:00")
    cache_manager.set("large-1", large_payload_1)

    frozen_datetime._now = datetime.fromisoformat("2024-01-01T11:00:00+00:00")
    cache_manager.set("large-2", large_payload_2)

    # Adding a third large entry should trigger evictions to stay under size limit
    frozen_datetime._now = datetime.fromisoformat("2024-01-01T12:00:00+00:00")
    cache_manager.set("large-3", large_payload_3)

    # Check that evictions occurred to maintain size limit
    remaining_files = list(cache_manager.advanced_cache_dir.glob("*.json"))

    # Calculate total size of remaining files
    total_size = sum(f.stat().st_size for f in remaining_files)
    max_size_bytes = int(0.001 * 1024 * 1024)  # Convert MB to bytes

    # Verify size constraint is respected (allowing some overhead for JSON structure)
    assert total_size <= max_size_bytes * 2, f"Total size {total_size} exceeds reasonable limit"

    # Verify some evictions occurred
    stats = cache_manager.get_statistics()
    assert stats["evictions"] >= 1

    # Verify at least the newest entry remains
    assert (cache_manager.advanced_cache_dir / "large-3.json").exists()


def test_cache_warming_with_rate_limiter_optimal_timing(tmp_path, frozen_datetime):
    """Test that cache warming respects rate limiter optimal timing."""
    # Create a mock rate limiter that returns optimal timing
    mock_rate_limiter = types.SimpleNamespace()

    # Test case 1: Optimal timing - should be due
    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=1,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=0,
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=False,
        optimal_timing=True,
    )

    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        rate_limiter=mock_rate_limiter,
        default_ttl_hours=1 / 60,  # 1 minute TTL
        grace_period_hours=1,  # 1 hour grace
    )

    # Schedule a pharmaceutical entry (high priority)
    cache_manager.warming_scheduler.schedule(
        "pharma-key", {"pharmaceutical_focus": True, "result_count": 20, "last_accessed_at": "2024-01-01T12:00:00Z"}
    )

    # Should be marked as due due to optimal timing
    warm_entries = cache_manager.warmable_entries()
    assert "pharma-key" in warm_entries

    # Clear and test case 2: Not optimal timing, low priority - should defer
    cache_manager.warming_scheduler._pending.clear()

    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=1,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=0,
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=False,
        optimal_timing=False,
    )

    # Schedule a low priority entry
    cache_manager.warming_scheduler.schedule(
        "low-priority-key",
        {"pharmaceutical_focus": False, "result_count": 5, "last_accessed_at": "2024-01-01T10:00:00Z"},
    )

    # Should be deferred (not returned)
    warm_entries = cache_manager.warmable_entries()
    assert "low-priority-key" not in warm_entries

    # Check decision summary
    if hasattr(cache_manager.warming_scheduler, "get_decision_summary"):
        summary = cache_manager.warming_scheduler.get_decision_summary()
        assert summary.get("defer", 0) > 0


def test_cache_warming_rate_limit_defer_long_wait(tmp_path, frozen_datetime):
    """Test that cache warming defers when rate limit wait is too long."""
    # Create a mock rate limiter with long wait time
    mock_rate_limiter = types.SimpleNamespace()
    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=3,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=400,  # > 5 minutes
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=True,
        optimal_timing=False,
    )

    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        rate_limiter=mock_rate_limiter,
    )

    # Schedule high priority entry
    cache_manager.warming_scheduler.schedule("high-priority-key", {"pharmaceutical_focus": True, "result_count": 50})

    # Should be deferred due to long wait time
    warm_entries = cache_manager.warmable_entries()
    assert "high-priority-key" not in warm_entries

    # Check decisions
    decisions = cache_manager.warming_scheduler.get_warming_decisions()
    pharma_decision = next(d for d in decisions if d.cache_key == "high-priority-key")
    assert pharma_decision.decision == "defer"
    assert "Rate limited" in pharma_decision.reason
    assert pharma_decision.suggested_defer_time == 400


def test_cache_warming_priority_calculation(tmp_path):
    """Test that warming priority is calculated correctly."""
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
    )

    now = datetime.now(UTC)

    # Test different priority levels
    entries = [
        # Low priority
        (
            "low",
            {
                "pharmaceutical_focus": False,
                "result_count": 5,
                "last_accessed_at": (now - timedelta(hours=48)).isoformat(),
            },
        ),
        # Medium priority (recent access)
        (
            "medium",
            {
                "pharmaceutical_focus": False,
                "result_count": 5,
                "last_accessed_at": (now - timedelta(hours=12)).isoformat(),
            },
        ),
        # High priority (pharmaceutical)
        (
            "high",
            {
                "pharmaceutical_focus": True,
                "result_count": 5,
                "last_accessed_at": (now - timedelta(hours=48)).isoformat(),
            },
        ),
        # Highest priority (pharma + many results + recent)
        (
            "highest",
            {
                "pharmaceutical_focus": True,
                "result_count": 50,
                "last_accessed_at": (now - timedelta(hours=12)).isoformat(),
            },
        ),
    ]

    for key, metadata in entries:
        cache_manager.warming_scheduler.schedule(key, metadata)

    # Get decisions (should be sorted by priority)
    decisions = cache_manager.warming_scheduler.get_warming_decisions()

    # Verify priority order
    assert decisions[0].cache_key == "highest"
    assert decisions[0].priority >= 125  # 100 (pharma) + 25 (recent) + 50 (results, capped)
    assert decisions[1].cache_key == "high"
    assert decisions[1].priority == 105  # 100 (pharma) + 5 (results) - no optimal timing bonus
    assert decisions[2].cache_key == "medium"
    assert decisions[2].priority == 30  # 25 (recent) + 5 (results)
    assert decisions[3].cache_key == "low"
    assert decisions[3].priority == 5  # 5 (results)


def test_cache_warming_optimal_timing_bonus(tmp_path):
    """Test that optimal timing gives priority bonus."""
    # Mock rate limiter with optimal timing
    mock_rate_limiter = types.SimpleNamespace()
    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=1,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=0,
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=False,
        optimal_timing=True,
    )

    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        rate_limiter=mock_rate_limiter,
    )

    # Schedule entry with base priority 50
    cache_manager.warming_scheduler.schedule("timing-test", {"pharmaceutical_focus": False, "result_count": 50})

    decisions = cache_manager.warming_scheduler.get_warming_decisions()
    decision = decisions[0]

    # Should have optimal timing bonus (50 * 0.7 = 35, but threshold makes it due anyway)
    assert decision.decision == "due"
    assert "optimal timing" in decision.reason


def test_cache_warming_compute_optimal_timing_with_rate_limiter(tmp_path):
    """Test compute_optimal_warming_timing with rate limiter."""
    # Mock rate limiter returning available immediately
    mock_rate_limiter = types.SimpleNamespace()
    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=1,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=0,
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=False,
        optimal_timing=True,
    )

    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        rate_limiter=mock_rate_limiter,
    )

    # Test high priority entry
    metadata = {"pharmaceutical_focus": True, "result_count": 30}
    timing = cache_manager.warming_scheduler.compute_optimal_warming_timing("test-key", metadata)

    assert timing["due_now"] is True
    assert timing["defer_until"] is None
    assert timing["priority"] >= 125  # 100 (pharma) + 30 (results) - optimal timing bonus
    assert "Rate limiter available" in timing["reason"]


def test_cache_warming_compute_optimal_timing_deferred(tmp_path):
    """Test compute_optimal_warming_timing with deferred timing."""
    # Mock rate limiter with wait time
    mock_rate_limiter = types.SimpleNamespace()
    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=3,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=600,  # 10 minutes
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=True,
        optimal_timing=False,
    )

    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        rate_limiter=mock_rate_limiter,
    )

    # Test medium priority entry
    metadata = {"pharmaceutical_focus": False, "result_count": 40}
    timing = cache_manager.warming_scheduler.compute_optimal_warming_timing("test-key", metadata)

    assert timing["due_now"] is False
    assert timing["defer_until"] is not None
    assert timing["priority"] == 40  # Base priority without bonus
    assert "optimal window" in timing["reason"].lower() or "deferred" in timing["reason"].lower()


def test_cache_warming_get_warming_schedule(tmp_path):
    """Test get_warming_schedule returns complete timing info."""
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
    )

    # Schedule multiple entries
    entries = {
        "high": {"pharmaceutical_focus": True, "result_count": 50},
        "medium": {"pharmaceutical_focus": False, "result_count": 30},
        "low": {"pharmaceutical_focus": False, "result_count": 10},
    }

    for key, metadata in entries.items():
        cache_manager.warming_scheduler.schedule(key, metadata)

    schedule = cache_manager.warming_scheduler.get_warming_schedule()

    # Verify all entries are in schedule
    assert len(schedule) == 3
    assert "high" in schedule
    assert "medium" in schedule
    assert "low" in schedule

    # Verify each entry has timing info
    for key, info in schedule.items():
        assert "due_now" in info
        assert "defer_until" in info
        assert "reason" in info
        assert "priority" in info
        assert "metadata" in info

    # Without rate limiter, high priority should be due
    assert schedule["high"]["due_now"] is True
    assert schedule["high"]["priority"] >= 125


def test_cache_warming_schedule_with_defer_times(tmp_path):
    """Test that deferred entries have appropriate defer times."""
    # Mock rate limiter that always requires waiting
    mock_rate_limiter = types.SimpleNamespace()
    mock_rate_limiter.get_status = lambda: RateLimitStatus(
        requests_last_second=3,
        seconds_until_daily_reset=3600,
        seconds_until_next_request=300,  # 5 minutes
        daily_count=10,
        daily_limit=500,
        remaining_daily_requests=490,
        rate_limited=True,
        optimal_timing=False,
    )

    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        rate_limiter=mock_rate_limiter,
    )

    # Schedule entry that would be deferred
    cache_manager.warming_scheduler.schedule("defer-test", {"pharmaceutical_focus": False, "result_count": 20})

    schedule = cache_manager.warming_scheduler.get_warming_schedule()
    entry_info = schedule["defer-test"]

    assert entry_info["due_now"] is False
    assert entry_info["defer_until"] is not None
    assert isinstance(entry_info["defer_until"], datetime)


def test_cache_expiry_override_behavior(tmp_path, frozen_datetime):
    """Test that explicit expiry override works correctly."""
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=1,  # 1 hour TTL for testing
        grace_period_hours=2,
    )

    # Create entry with standard TTL
    cache_manager.set("standard", {"data": "test"})

    # Create entry with explicit expiry (3 hours from now)
    explicit_expiry = frozen_datetime._now + timedelta(hours=3)
    cache_manager.set("explicit", {"data": "test"}, explicit_expiry=explicit_expiry)

    # Create entry with preserve_expiry
    cache_manager.set("preserve", {"data": "test"})
    frozen_datetime.advance(minutes=30)
    cache_manager.set("preserve", {"data": "updated"}, preserve_expiry=True)

    # Advance time past standard TTL but within grace period and before explicit expiry
    frozen_datetime.advance(hours=1, minutes=30)

    # Standard entry should be stale but available (within grace)
    standard_result = cache_manager.get("standard")
    assert standard_result is not None and standard_result.stale is True

    # Explicit entry should still be fresh
    explicit_result = cache_manager.get("explicit")
    assert explicit_result is not None and explicit_result.stale is False

    # Preserve entry should have original expiry (30 mins used + 1.5 hours = 1h 50m < 2h grace)
    preserve_result = cache_manager.get("preserve")
    assert preserve_result is not None and preserve_result.stale is True


def test_cache_expiry_grace_period_override(tmp_path, frozen_datetime):
    """Test explicit grace period expiry override."""
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=1,
        grace_period_hours=1,  # 1 hour grace
    )

    # Create entry with explicit grace expiry (2 hours from now)
    now = frozen_datetime._now
    explicit_grace_expiry = now + timedelta(hours=2)
    cache_manager.set("grace-test", {"data": "test"}, explicit_grace_expiry=explicit_grace_expiry)

    # Advance past TTL but within explicit grace period
    frozen_datetime.advance(hours=1, minutes=30)

    # Entry should be stale but available
    result = cache_manager.get("grace-test")
    assert result is not None
    assert result.stale is True

    # Advance past explicit grace period
    frozen_datetime.advance(hours=1)

    # Entry should be evicted
    result = cache_manager.get("grace-test")
    assert result is None


def test_cache_warming_with_new_timing_methods(tmp_path):
    """Test that warmable_entries uses new timing methods when available."""
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
    )

    # Schedule entries with different priorities
    cache_manager.warming_scheduler.schedule("high-priority", {"pharmaceutical_focus": True, "result_count": 50})
    cache_manager.warming_scheduler.schedule("low-priority", {"pharmaceutical_focus": False, "result_count": 10})

    # Get warmable entries - should use new timing method
    warm_entries = cache_manager.warmable_entries()

    # High priority should be returned, low priority deferred
    assert "high-priority" in warm_entries
    assert "low-priority" not in warm_entries

    # Verify analytics recorded warming
    stats = cache_manager.get_statistics()
    assert stats["warmed"] >= 1


def test_cache_warming_fallback_without_rate_limiter(tmp_path):
    """Test that warming falls back gracefully without rate limiter."""
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_warming_enabled=True,
        # No rate_limiter provided
    )

    # Schedule entries
    cache_manager.warming_scheduler.schedule("key1", {"result_count": 50})
    cache_manager.warming_scheduler.schedule("key2", {"pharmaceutical_focus": True})

    # Should still work, using priority-based decisions
    warm_entries = cache_manager.warmable_entries()

    # Without rate limiter:
    # - key1: priority 50 (result count) -> deferred (< 75 threshold)
    # - key2: priority 100 (pharma) -> due (>= 75 threshold)
    assert len(warm_entries) == 1
    assert "key2" in warm_entries
    assert "key1" not in warm_entries


def test_import_cache_skips_invalid_entries(cache_manager, tmp_path):
    import_dir = tmp_path / "import_test"
    import_dir.mkdir()

    # Create a valid cache entry
    valid_entry = {
        "schema_version": "1",
        "created_at": "2024-01-01T10:00:00Z",
        "expires_at": "2024-01-02T10:00:00Z",
        "grace_expires_at": "2024-01-02T12:00:00Z",
        "hit_count": 0,
        "payload": "eyJ0aXRsZSI6InZhbGlkIn0=",  # base64 encoded {"title":"valid"}
        "payload_encoding": "plain",
        "metadata": {"original_query": "test"},
    }
    (import_dir / "valid.json").write_text(json.dumps(valid_entry))

    # Create an invalid JSON file (malformed)
    (import_dir / "malformed.json").write_text('{"invalid": json}')

    # Create a file missing required fields
    invalid_entry = {"schema_version": "1", "incomplete": "data"}
    (import_dir / "incomplete.json").write_text(json.dumps(invalid_entry))

    # Create a file with wrong schema version
    wrong_version_entry = dict(valid_entry)
    wrong_version_entry["schema_version"] = "999"
    (import_dir / "wrong_version.json").write_text(json.dumps(wrong_version_entry))

    # Import should only process the valid entry
    imported_count = cache_manager.import_cache(import_dir)
    assert imported_count == 1

    # Verify only the valid entry was imported
    imported_files = list(cache_manager.advanced_cache_dir.glob("*.json"))
    assert len(imported_files) == 1
    assert (cache_manager.advanced_cache_dir / "valid.json").exists()

    # Verify the valid entry can be retrieved
    result = cache_manager.get("valid")
    assert result is not None
    assert result.payload == {"title": "valid"}


def test_export_import_preserves_valid_entries(cache_manager, tmp_path):
    # Set up cache entries
    cache_manager.set("export-test-1", [{"title": "first"}])
    cache_manager.set("export-test-2", [{"title": "second"}])

    # Export to a directory
    export_dir = tmp_path / "export"
    exported_count = cache_manager.export_cache(export_dir)
    assert exported_count == 2

    # Clear the cache
    for file in cache_manager.advanced_cache_dir.glob("*.json"):
        file.unlink()

    # Import back
    imported_count = cache_manager.import_cache(export_dir)
    assert imported_count == 2

    # Verify entries are accessible
    result1 = cache_manager.get("export-test-1")
    result2 = cache_manager.get("export-test-2")
    assert result1 is not None and result1.payload == [{"title": "first"}]
    assert result2 is not None and result2.payload == [{"title": "second"}]


def test_stale_entry_disallowed_schedules_warming(tmp_path, frozen_datetime):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=1 / 60,  # 1 minute TTL
        grace_period_hours=1,  # 1 hour grace period
        cache_warming_enabled=True,  # Enable warming
    )

    # Create a cache entry with pharmaceutical focus to ensure high priority
    cache_manager.set(
        "warming-test", [{"title": "to-warm"}], metadata={"pharmaceutical_focus": True, "result_count": 50}
    )

    # Make entry stale but within grace period
    frozen_datetime.advance(minutes=2)  # Past TTL but within grace

    # Verify entry is stale
    stale_check = cache_manager.get("warming-test", allow_stale=True)
    assert stale_check is not None and stale_check.stale is True

    # Clear warming queue
    cache_manager.warmable_entries()

    # Access with allow_stale=False should schedule warming
    result = cache_manager.get("warming-test", allow_stale=False)
    assert result is None  # Should return None due to stale + disallowed

    # Verify entry was scheduled for warming
    # Since this entry has low priority (not pharmaceutical, only 1 result),
    # it may be deferred by the new rate-limiter aware scheduler.
    # Use pop_all to get all entries regardless of priority for this legacy test.
    warm_entries = cache_manager.warming_scheduler.pop_all()

    assert "warming-test" in warm_entries
    assert warm_entries["warming-test"] is not None


def test_stale_entry_disallowed_no_warming_when_disabled(tmp_path, frozen_datetime):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        default_ttl_hours=1 / 60,  # 1 minute TTL
        grace_period_hours=1,  # 1 hour grace period
        cache_warming_enabled=False,  # Disable warming
    )

    # Create a cache entry
    cache_manager.set("no-warming-test", [{"title": "no-warm"}])

    # Make entry stale but within grace period
    frozen_datetime.advance(minutes=2)

    # Access with allow_stale=False should not schedule warming
    result = cache_manager.get("no-warming-test", allow_stale=False)
    assert result is None

    # Verify no entry was scheduled for warming
    warm_entries = cache_manager.warmable_entries()
    assert len(warm_entries) == 0


def test_json_formatting_compact_by_default(tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_pretty_json=False,  # Compact formatting (default)
    )

    cache_manager.set("compact-test", [{"title": "compact"}])

    cache_file = cache_manager.advanced_cache_dir / "compact-test.json"
    content = cache_file.read_text()

    # Compact JSON should not have indentation or extra spaces
    assert "\n  " not in content, "Compact JSON should not have indented lines"
    assert ": " not in content, "Compact JSON should not have spaces after colons"
    assert ", " not in content, "Compact JSON should not have spaces after commas"


def test_json_formatting_pretty_when_enabled(tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=tmp_path,
        cache_pretty_json=True,  # Pretty formatting enabled
    )

    cache_manager.set("pretty-test", [{"title": "pretty"}])

    cache_file = cache_manager.advanced_cache_dir / "pretty-test.json"
    content = cache_file.read_text()

    # Pretty JSON should have indentation
    assert "\n  " in content, "Pretty JSON should have indented lines"
