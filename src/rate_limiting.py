"""Rate limiting utilities for enforcing NCBI compliance."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from typing import Deque, Dict, Optional, Tuple

import pytz


logger = logging.getLogger(__name__)


_default_process_limiter: Optional["NCBIRateLimiter"] = None
_process_lock = threading.Lock()


def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


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


def _now_monotonic() -> float:
    """Module-level function for monotonic time that tests can patch."""
    return time.monotonic()


@dataclass(frozen=True)
class RateLimitStatus:
    requests_last_second: int
    seconds_until_daily_reset: float
    seconds_until_next_request: float
    daily_count: int
    daily_limit: Optional[int]
    remaining_daily_requests: Optional[int]
    rate_limited: bool
    optimal_timing: bool


class NCBIRateLimiter:
    """Thread-safe sliding window limiter with daily quota and timing guidance.

    Args:
        daily_request_limit: Maximum daily requests (None, 0, or -1 disables quota).
    """

    def __init__(
        self,
        max_requests_per_second: int = 3,
        daily_request_limit: Optional[int] = 500,
        enable_optimal_timing_detection: bool = True,
        timezone: str = "US/Eastern",
        log_level: str = "DEBUG",
        raise_on_daily_limit: bool = False,
        max_daily_wait_seconds: Optional[int] = None,
        weekend_optimal: bool = True,
    ) -> None:
        if max_requests_per_second <= 0:
            raise ValueError("max_requests_per_second must be positive")
        if daily_request_limit is not None and daily_request_limit <= 0:
            if daily_request_limit in {0, -1}:
                daily_request_limit = None
            else:
                raise ValueError("daily_request_limit must be positive, 0, or -1 to disable")

        self.max_requests_per_second = max_requests_per_second
        self.daily_request_limit = daily_request_limit
        self.enable_optimal_timing_detection = enable_optimal_timing_detection
        self.tz = pytz.timezone(timezone)
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()
        self._daily_count = 0
        self._daily_reset_date = datetime.now(self.tz).date()
        self._log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.raise_on_daily_limit = raise_on_daily_limit
        self.max_daily_wait_seconds = max_daily_wait_seconds
        self.weekend_optimal = weekend_optimal

    @classmethod
    def from_env(cls) -> "NCBIRateLimiter":
        """Create limiter from environment variables.

        By default, the limiter enforces NCBI's 3 req/sec guidance and caps waiting
        for daily quota resets to one hour (`RATE_LIMIT_MAX_DAILY_WAIT_SECONDS`).
        Override the behaviour by setting the associated environment variables.
        """
        daily_limit_env = os.getenv("DAILY_REQUEST_LIMIT")
        daily_limit: Optional[int]
        if daily_limit_env is None:
            daily_limit = 500
        else:
            try:
                parsed_daily = int(daily_limit_env)
            except ValueError as exc:
                raise ValueError("Environment variable DAILY_REQUEST_LIMIT must be an integer") from exc
            if parsed_daily in {0, -1}:
                daily_limit = None
            elif parsed_daily < -1:
                raise ValueError("Environment variable DAILY_REQUEST_LIMIT must be >= -1")
            else:
                daily_limit = parsed_daily

        max_wait_env = os.getenv("RATE_LIMIT_MAX_DAILY_WAIT_SECONDS")
        if max_wait_env is None or max_wait_env == "":
            max_wait = None
        else:
            try:
                max_wait = int(max_wait_env)
            except ValueError as exc:
                raise ValueError(
                    "Environment variable RATE_LIMIT_MAX_DAILY_WAIT_SECONDS must be an integer"
                ) from exc
            if max_wait < 0:
                raise ValueError(
                    "Environment variable RATE_LIMIT_MAX_DAILY_WAIT_SECONDS must be non-negative"
                )

        weekend_optimal = _env_flag("RATE_LIMIT_WEEKEND_OPTIMAL", "true")

        mps = _env_int("MAX_REQUESTS_PER_SECOND", 3)
        if mps < 1:
            raise ValueError("Environment variable MAX_REQUESTS_PER_SECOND must be >= 1")

        return cls(
            max_requests_per_second=mps,
            daily_request_limit=daily_limit,
            enable_optimal_timing_detection=_env_flag("ENABLE_OPTIMAL_TIMING_DETECTION", "true"),
            timezone=os.getenv("RATE_LIMIT_TIMEZONE", "US/Eastern"),
            log_level=os.getenv("RATE_LIMITING_LOG_LEVEL", "DEBUG"),
            raise_on_daily_limit=_env_flag("RATE_LIMIT_RAISE_ON_DAILY_LIMIT", "false"),
            max_daily_wait_seconds=max_wait,
            weekend_optimal=weekend_optimal,
        )

    def _append_request(self, now_monotonic: float, tz_now: datetime) -> None:
        self._timestamps.append(now_monotonic)
        self._daily_count += 1
        logger.log(
            self._log_level,
            "Recorded request #%d for %s",
            self._daily_count,
            tz_now.date(),
        )

    def _prune_old_requests(self, now_monotonic: float) -> None:
        max_age = 1.0
        while self._timestamps and now_monotonic - self._timestamps[0] >= max_age:
            self._timestamps.popleft()

    def _reset_daily_counter_if_needed(self, tz_now: datetime) -> None:
        current_date = tz_now.date()
        if current_date != self._daily_reset_date:
            self._daily_reset_date = current_date
            self._daily_count = 0

    def _seconds_until_slot(self, now_monotonic: float) -> float:
        if len(self._timestamps) < self.max_requests_per_second:
            return 0.0
        oldest = self._timestamps[0]
        return max(0.0, 1.0 - (now_monotonic - oldest))

    def _seconds_until_daily_reset(self, tz_now: datetime) -> float:
        next_midnight = self.tz.localize(datetime.combine(tz_now.date(), dtime.min)) + timedelta(days=1)
        next_midnight = self.tz.normalize(next_midnight)
        return max(0.0, (next_midnight - tz_now).total_seconds())

    def _is_optimal_timing(self, tz_now: datetime) -> bool:
        if not self.enable_optimal_timing_detection:
            return False
        if self.weekend_optimal and tz_now.weekday() >= 5:
            return True
        hour = tz_now.hour
        return hour >= 21 or hour < 5

    def check_rate_limit(self) -> Tuple[bool, float]:
        now_monotonic = _now_monotonic()
        tz_now = datetime.now(self.tz)
        with self._lock:
            self._reset_daily_counter_if_needed(tz_now)
            self._prune_old_requests(now_monotonic)

            if self.daily_request_limit is not None and self._daily_count >= self.daily_request_limit:
                logger.warning("Daily NCBI quota reached: %s", self._daily_count)
                return False, self._seconds_until_daily_reset(tz_now)

            wait_seconds = self._seconds_until_slot(now_monotonic)
            if wait_seconds > 0:
                logger.log(self._log_level, "NCBI rate limited for %.2f seconds", wait_seconds)
                return False, wait_seconds

            return True, 0.0

    def record_request(self) -> None:
        now_monotonic = _now_monotonic()
        tz_now = datetime.now(self.tz)
        with self._lock:
            self._reset_daily_counter_if_needed(tz_now)
            self._prune_old_requests(now_monotonic)
            self._append_request(now_monotonic, tz_now)

    def acquire(
        self,
        raise_on_daily_limit: Optional[bool] = None,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        if raise_on_daily_limit is None:
            raise_on_daily_limit = self.raise_on_daily_limit
        if max_wait_seconds is None:
            max_wait_seconds = self.max_daily_wait_seconds

        while True:
            now_monotonic = _now_monotonic()
            tz_now = datetime.now(self.tz)
            with self._lock:
                self._reset_daily_counter_if_needed(tz_now)
                self._prune_old_requests(now_monotonic)

                if self.daily_request_limit is not None and self._daily_count >= self.daily_request_limit:
                    wait = self._seconds_until_daily_reset(tz_now)
                    if raise_on_daily_limit or (
                        max_wait_seconds is not None and wait > max_wait_seconds
                    ):
                        raise DailyQuotaExceeded(
                            "Daily limit reached; waiting {:.0f}s exceeds policy".format(wait)
                        )
                    reason = "daily"
                else:
                    wait = self._seconds_until_slot(now_monotonic)
                    if wait <= 0:
                        self._append_request(now_monotonic, tz_now)
                        return
                    reason = "rate"

            if wait <= 0:
                continue

            if reason == "daily":
                logger.warning("Daily NCBI quota reached: %s", self._daily_count)

            time.sleep(min(wait, 60))

    async def acquire_async(
        self,
        raise_on_daily_limit: Optional[bool] = None,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        if raise_on_daily_limit is None:
            raise_on_daily_limit = self.raise_on_daily_limit
        if max_wait_seconds is None:
            max_wait_seconds = self.max_daily_wait_seconds

        while True:
            now_monotonic = _now_monotonic()
            tz_now = datetime.now(self.tz)
            with self._lock:
                self._reset_daily_counter_if_needed(tz_now)
                self._prune_old_requests(now_monotonic)

                if self.daily_request_limit is not None and self._daily_count >= self.daily_request_limit:
                    wait = self._seconds_until_daily_reset(tz_now)
                    if raise_on_daily_limit or (
                        max_wait_seconds is not None and wait > max_wait_seconds
                    ):
                        raise DailyQuotaExceeded(
                            "Daily limit reached; waiting {:.0f}s exceeds policy".format(wait)
                        )
                    reason = "daily"
                else:
                    wait = self._seconds_until_slot(now_monotonic)
                    if wait <= 0:
                        self._append_request(now_monotonic, tz_now)
                        return
                    reason = "rate"

            if wait <= 0:
                continue

            if reason == "daily":
                logger.warning("Daily NCBI quota reached: %s", self._daily_count)

            await asyncio.sleep(wait)

    def wait_until_ready(
        self,
        raise_on_daily_limit: Optional[bool] = None,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        if raise_on_daily_limit is None:
            raise_on_daily_limit = self.raise_on_daily_limit
        if max_wait_seconds is None:
            max_wait_seconds = self.max_daily_wait_seconds

        while True:
            allowed, wait_time = self.check_rate_limit()
            if allowed:
                return
            if self.daily_request_limit is not None:
                if raise_on_daily_limit or (
                    max_wait_seconds is not None and wait_time > max_wait_seconds
                ):
                    raise DailyQuotaExceeded(
                        "Daily limit reached; waiting {:.0f}s exceeds policy".format(wait_time)
                    )
                if wait_time > 60:
                    logger.error(
                        "Daily quota exceeded, waiting %.0f seconds until reset", wait_time
                    )
            time.sleep(min(wait_time, 60))

    async def wait_until_ready_async(
        self,
        raise_on_daily_limit: Optional[bool] = None,
        max_wait_seconds: Optional[int] = None,
    ) -> None:
        if raise_on_daily_limit is None:
            raise_on_daily_limit = self.raise_on_daily_limit
        if max_wait_seconds is None:
            max_wait_seconds = self.max_daily_wait_seconds

        while True:
            allowed, wait_time = self.check_rate_limit()
            if allowed:
                return
            if self.daily_request_limit is not None:
                if raise_on_daily_limit or (
                    max_wait_seconds is not None and wait_time > max_wait_seconds
                ):
                    raise DailyQuotaExceeded(
                        "Daily limit reached; waiting {:.0f}s exceeds policy".format(wait_time)
                    )
            await asyncio.sleep(min(wait_time, 60))

    def get_status(self) -> RateLimitStatus:
        now_monotonic = _now_monotonic()
        tz_now = datetime.now(self.tz)
        with self._lock:
            self._reset_daily_counter_if_needed(tz_now)
            self._prune_old_requests(now_monotonic)
            seconds_until_reset = self._seconds_until_daily_reset(tz_now)
            requests_last_second = len(self._timestamps)
            daily_exhausted = (
                self.daily_request_limit is not None and self._daily_count >= self.daily_request_limit
            )
            rate_limited = requests_last_second >= self.max_requests_per_second or daily_exhausted
            next_request_wait = (
                self._seconds_until_daily_reset(tz_now)
                if daily_exhausted
                else self._seconds_until_slot(now_monotonic)
            )
            remaining = (
                None
                if self.daily_request_limit is None
                else max(self.daily_request_limit - self._daily_count, 0)
            )
        return RateLimitStatus(
            requests_last_second=requests_last_second,
            seconds_until_daily_reset=seconds_until_reset,
            seconds_until_next_request=next_request_wait,
            daily_count=self._daily_count,
            daily_limit=self.daily_request_limit,
            remaining_daily_requests=remaining,
            rate_limited=rate_limited,
            optimal_timing=self._is_optimal_timing(tz_now),
        )

    def get_compliance_report(self) -> Dict[str, object]:
        status = self.get_status()
        return {
            "requests_last_second": status.requests_last_second,
            "daily_count": status.daily_count,
            "daily_limit": status.daily_limit,
            "remaining_daily_requests": status.remaining_daily_requests,
            "rate_limited": status.rate_limited,
            "seconds_until_next_request": status.seconds_until_next_request,
            "optimal_timing": status.optimal_timing,
        }

    def is_optimal_timing(self) -> bool:
        tz_now = datetime.now(self.tz)
        return self._is_optimal_timing(tz_now)

    def remaining_daily_requests(self) -> Optional[int]:
        with self._lock:
            if self.daily_request_limit is None:
                return None
            return max(self.daily_request_limit - self._daily_count, 0)

    def seconds_until_next_request(self) -> float:
        allowed, wait_time = self.check_rate_limit()
        return 0.0 if allowed else wait_time

    def seconds_until_daily_reset(self) -> float:
        tz_now = datetime.now(self.tz)
        return self._seconds_until_daily_reset(tz_now)

    def should_wait(self) -> float:
        """Return wait seconds from check_rate_limit() (0 when ready)."""
        allowed, wait_time = self.check_rate_limit()
        return 0.0 if allowed else wait_time


def get_process_limiter() -> NCBIRateLimiter:
    """Return a shared process-wide limiter configured from the environment."""

    global _default_process_limiter
    if _default_process_limiter is None:
        with _process_lock:
            if _default_process_limiter is None:
                _default_process_limiter = NCBIRateLimiter.from_env()
    return _default_process_limiter


class DailyQuotaExceeded(RuntimeError):
    """Raised when the daily API quota is exceeded and waiting is disallowed."""
