import asyncio
import threading
from datetime import datetime
from datetime import timedelta

import pytest

import src.rate_limiting as rl


class TimeStub:
    def __init__(self):
        self.current = 0.0
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self.current

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            seconds = 0.0
        with self._lock:
            self.current += seconds


def test_sliding_window_rate_limit(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=2,
        daily_request_limit=10,
        enable_optimal_timing_detection=False,
    )

    allowed, wait = limiter.check_rate_limit()
    assert allowed
    assert wait == pytest.approx(0)

    limiter.record_request()
    limiter.record_request()

    allowed, wait = limiter.check_rate_limit()
    assert not allowed
    assert wait > 0

    time_stub.sleep(wait)
    allowed, wait = limiter.check_rate_limit()
    assert allowed
    assert wait == pytest.approx(0)


def test_daily_quota_resets(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=5,
        daily_request_limit=2,
        enable_optimal_timing_detection=False,
    )

    limiter.record_request()
    limiter.record_request()

    allowed, wait = limiter.check_rate_limit()
    assert not allowed
    assert wait > 0
    status = limiter.get_status()
    assert status is not None
    assert status.rate_limited is True
    assert status.remaining_daily_requests == 0

    limiter._daily_reset_date = limiter._daily_reset_date - timedelta(days=1)
    limiter._daily_count = limiter.daily_request_limit

    allowed, wait = limiter.check_rate_limit()
    assert allowed
    assert limiter._daily_count == 0
    status_after_reset = limiter.get_status()
    assert status_after_reset is not None
    assert status_after_reset.rate_limited is False


def test_optimal_timing_detection(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_WEEKEND_OPTIMAL", "true")
    limiter = rl.NCBIRateLimiter(enable_optimal_timing_detection=True, weekend_optimal=True)

    optimal_time = limiter.tz.localize(datetime(2024, 1, 1, 22, 0))  # Monday 10 PM
    non_optimal_time = limiter.tz.localize(datetime(2024, 1, 1, 12, 0))  # Monday noon

    weekend_time = limiter.tz.localize(datetime(2024, 1, 6, 12, 0))  # Saturday noon

    assert limiter._is_optimal_timing(optimal_time) is True
    assert limiter._is_optimal_timing(non_optimal_time) is False
    assert limiter._is_optimal_timing(weekend_time) is True


def test_optimal_timing_weekend_disabled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_WEEKEND_OPTIMAL", "false")
    limiter = rl.NCBIRateLimiter(enable_optimal_timing_detection=True, weekend_optimal=False)

    weekend_time = limiter.tz.localize(datetime(2024, 1, 6, 12, 0))  # Saturday noon

    assert limiter._is_optimal_timing(weekend_time) is False


def test_wait_until_ready_async(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=1,
        daily_request_limit=10,
        enable_optimal_timing_detection=False,
    )

    limiter.record_request()

    call_sequence = []

    async def fake_sleep(duration):
        call_sequence.append(duration)
        time_stub.sleep(duration)

    monkeypatch.setattr(rl.asyncio, "sleep", fake_sleep)

    asyncio.run(limiter.acquire_async())
    assert call_sequence
    assert call_sequence[0] >= 0


def test_seconds_until_daily_reset_handles_dst():
    limiter = rl.NCBIRateLimiter(enable_optimal_timing_detection=False, timezone="US/Eastern")
    dst_transition = limiter.tz.localize(datetime(2024, 3, 10, 1, 30))

    wait = limiter._seconds_until_daily_reset(dst_transition)

    assert wait > 0
    assert wait < 25 * 3600


def test_unlimited_daily_limit_via_env(monkeypatch):
    monkeypatch.setenv("DAILY_REQUEST_LIMIT", "0")
    monkeypatch.setenv("MAX_REQUESTS_PER_SECOND", "10")

    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter.from_env()

    assert limiter.daily_request_limit is None

    assert limiter.max_requests_per_second == 10

    for _ in range(5):
        limiter.acquire()

    status = limiter.get_status()
    assert status is not None
    assert status.daily_limit is None
    assert status.remaining_daily_requests is None


def test_acquire_raises_when_daily_limit_exceeded(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=5,
        daily_request_limit=1,
        enable_optimal_timing_detection=False,
        raise_on_daily_limit=True,
    )

    limiter.record_request()

    with pytest.raises(rl.DailyQuotaExceeded):
        limiter.acquire()


def test_acquire_respects_max_wait_seconds(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=5,
        daily_request_limit=1,
        enable_optimal_timing_detection=False,
        raise_on_daily_limit=False,
    )

    limiter.record_request()

    with pytest.raises(rl.DailyQuotaExceeded):
        limiter.acquire(max_wait_seconds=10)


def test_acquire_is_atomic_across_threads(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=2,
        daily_request_limit=None,
        enable_optimal_timing_detection=False,
    )

    results = []
    results_lock = threading.Lock()

    def worker():
        limiter.acquire()
        with results_lock:
            results.append(time_stub.time())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 6
    results.sort()
    limit = limiter.max_requests_per_second
    for idx in range(limit, len(results)):
        assert results[idx] - results[idx - limit] >= 1


def test_wait_until_ready_sync(monkeypatch):
    time_stub = TimeStub()
    monkeypatch.setattr(rl, "_now_monotonic", time_stub.time)

    limiter = rl.NCBIRateLimiter(
        max_requests_per_second=1,
        daily_request_limit=3,
        enable_optimal_timing_detection=False,
    )

    # First request should pass immediately
    limiter.wait_until_ready()
    limiter.record_request()

    # Second call should require waiting ~1s due to rate limiting
    start = time_stub.time()
    limiter.wait_until_ready()
    after_wait = time_stub.time()
    assert after_wait - start >= 1
    limiter.record_request()

    # Exhaust daily quota and ensure it raises when max_wait_seconds is small
    limiter.record_request()
    with pytest.raises(rl.DailyQuotaExceeded):
        limiter.wait_until_ready(max_wait_seconds=5)
