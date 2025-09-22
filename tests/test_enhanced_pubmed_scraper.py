import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from src.cache_management import NCBICacheManager
from src.enhanced_pubmed_scraper import EnhancedPubMedScraper
from src.pubmed_scraper import PubMedScraper
from src.rate_limiting import RateLimitStatus


class DummyLimiter:
    def __init__(self):
        self.actions: List[str] = []
        self.max_requests_per_second = 3
        self.daily_request_limit = 50

    def acquire(self, **_kwargs) -> None:
        self.actions.append("acquire")

    async def acquire_async(self, **_kwargs) -> None:
        self.actions.append("acquire_async")

    def get_status(self) -> RateLimitStatus:
        return RateLimitStatus(
            requests_last_second=0,
            seconds_until_daily_reset=0.0,
            seconds_until_next_request=0.0,
            daily_count=0,
            daily_limit=50,
            remaining_daily_requests=50,
            rate_limited=False,
            optimal_timing=True,
        )

    def get_compliance_report(self) -> dict:
        return {"rate_limited": False}

    def is_optimal_timing(self) -> bool:
        return True


class FakeActor:
    def __init__(self, client):
        self._client = client

    def call(self, run_input=None):
        self._client.calls.append(run_input)
        if self._client.responses:
            run = self._client.responses.pop(0)
        else:
            run = {"items": []}

        if "defaultDatasetId" not in run:
            run = {**run, "defaultDatasetId": "test-dataset"}
        return run


class FakeDataset:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):  # pragma: no cover - should not be called when patched
        yield from self._items


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: List[dict] = []

    def actor(self, _actor_id):
        return FakeActor(self)

    def dataset(self, _dataset_id):
        return FakeDataset([])


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("APIFY_TOKEN", "test-token")
    monkeypatch.setenv("PUBMED_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
    monkeypatch.setenv("ENABLE_ENHANCED_PUBMED_SCRAPER", "true")
    monkeypatch.setenv("ENABLE_ADVANCED_CACHING", "true")
    monkeypatch.setenv("ENABLE_EASYAPI_SCHEMA_FALLBACK", "false")
    monkeypatch.setenv("ENABLE_PHARMA_QUERY_ENHANCEMENT", "false")
    monkeypatch.setenv("ENABLE_STUDY_RANKING", "false")
    monkeypatch.setenv("ENABLE_OPTIMAL_TIMING_DETECTION", "true")
    monkeypatch.setenv("MIRROR_TO_LEGACY_ROOT", "false")
    monkeypatch.setenv("PRUNE_LEGACY_AFTER_MIGRATION", "false")
    return cache_dir


@pytest.fixture
def fake_client(monkeypatch):
    responses = [{"items": [{"title": "primary"}], "defaultDatasetId": "dataset-1"}]
    created_clients: List[FakeClient] = []

    def factory(_token):
        client = FakeClient(responses)
        created_clients.append(client)
        return client

    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", factory)
    return created_clients


@pytest.fixture(autouse=True)
def patch_process_results(monkeypatch):
    def fake_process(self, run, apply_ranking):
        return run.get("items", [])

    monkeypatch.setattr(PubMedScraper, "_process_run_results", fake_process)


def test_cache_hit_skips_rate_limiter(fake_env, fake_client, tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=fake_env,
        default_ttl_hours=24,
        grace_period_hours=2,
        cache_warming_enabled=True,
    )

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    results = scraper.search_pubmed("oncology", max_items=5, rank=False)
    assert results == [{"title": "primary"}]
    assert limiter.actions == ["acquire"]

    limiter.actions.clear()
    fake_client[0].calls.clear()

    cached = scraper.search_pubmed("oncology", max_items=5, rank=False)
    assert cached == results
    assert limiter.actions == []
    assert fake_client[0].calls == []
    assert cache_manager.get_statistics()["hits"] >= 1


def test_stale_hit_schedules_warming(fake_env, fake_client, tmp_path):
    cache_manager = NCBICacheManager(
        cache_dir=fake_env,
        default_ttl_hours=1 / 60,
        grace_period_hours=1,
        cache_warming_enabled=True,
    )

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)
    scraper.search_pubmed("pharmacology", max_items=5, rank=False)

    cache_files = list(cache_manager.advanced_cache_dir.glob("*.json"))
    assert cache_files, "Expected cached entry to exist"
    cache_entry_path = cache_files[0]

    entry = json.loads(cache_entry_path.read_text())
    entry["expires_at"] = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    entry["grace_expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    cache_entry_path.write_text(json.dumps(entry))

    cache_key = cache_entry_path.stem
    stale_probe = cache_manager.get(cache_key)
    assert stale_probe is not None
    assert stale_probe.stale is True

    fake_client[0].calls.clear()
    limiter.actions.clear()

    cached = scraper.search_pubmed("pharmacology", max_items=5, rank=False)
    assert cached == [{"title": "primary"}]
    assert limiter.actions == []
    assert fake_client[0].calls == []

    warming = scraper.warm_cache_entries()
    assert warming


def test_combined_status_report(fake_env, fake_client):
    cache_manager = NCBICacheManager(cache_dir=fake_env)
    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)
    scraper.search_pubmed("diabetes", max_items=3, rank=False)

    report = scraper.combined_status_report()
    assert report["cache"]["enabled"] is True
    assert "statistics" in report["cache"]
    assert report["rate_limit"]["daily_limit"] == 50


def test_advanced_caching_can_be_disabled(fake_env, fake_client, monkeypatch):
    monkeypatch.setenv("ENABLE_ADVANCED_CACHING", "false")
    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(rate_limiter=limiter, enable_rate_limiting=True)

    cache_manager = scraper.get_cache_manager()
    assert cache_manager is None


def test_normalized_cache_keys_migrate_and_duplicate(fake_env, fake_client, monkeypatch):
    monkeypatch.setenv("USE_NORMALIZED_CACHE_KEYS", "true")
    cache_manager = NCBICacheManager(cache_dir=fake_env)
    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    scraper.search_pubmed("cardiology", max_items=5, rank=False)

    cache_files = {path.name for path in cache_manager.advanced_cache_dir.glob("*.json")}
    assert cache_files
    assert any(name.startswith("v1-") for name in cache_files), "expected normalized cache key"
    assert any(len(name) == 37 for name in cache_files), "expected legacy cache key copy"

    # Clearing limiter actions to observe cache hit
    limiter.actions.clear()
    fake_client[0].calls.clear()

    cached = scraper.search_pubmed("cardiology", max_items=5, rank=False)
    assert cached == [{"title": "primary"}]
    assert limiter.actions == []
    assert fake_client[0].calls == []

    fake_client[0].responses.append({"items": [{"title": "primary"}], "defaultDatasetId": "dataset-2"})
    result = scraper.search_pubmed("neurology", max_items=5, rank=False)
    assert result == [{"title": "primary"}]

    cache_dir = Path(os.getenv("PUBMED_CACHE_DIR"))
    assert not list(cache_dir.glob("*.json")), "legacy cache directory should stay clean when advanced cache is active"


def test_prunes_legacy_after_normalized_migration(fake_env, fake_client, monkeypatch):
    monkeypatch.setenv("PRUNE_LEGACY_AFTER_MIGRATION", "true")
    monkeypatch.setenv("USE_NORMALIZED_CACHE_KEYS", "true")

    cache_manager = NCBICacheManager(
        cache_dir=fake_env,
        default_ttl_hours=24,
        grace_period_hours=2,
    )

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(
        cache_manager=cache_manager,
        rate_limiter=limiter,
        enable_rate_limiting=True,
        use_normalized_cache_keys=True,
    )

    enhanced_query = scraper._enhance_pharmaceutical_query("prune", False)
    cache_key = scraper._get_cache_key(
        enhanced_query,
        max_items=5,
        apply_ranking=False,
        pharma_enhance_enabled=False,
        include_tags_effective=False,
        include_abstract_effective=False,
        preserve_order=False,
    )
    context = scraper._get_cache_context()
    legacy_key = context["legacy_cache_key"]
    legacy_path = fake_env / f"{legacy_key}.json"
    legacy_payload = [{"title": "legacy"}]
    legacy_path.write_text(json.dumps(legacy_payload))

    advanced_path = cache_manager.advanced_cache_dir / f"{cache_key}.json"
    if advanced_path.exists():
        advanced_path.unlink()

    results = scraper._get_cached_results(cache_key, apply_ranking=False)
    assert results == legacy_payload
    assert advanced_path.exists()
    assert not legacy_path.exists(), "Legacy cache file should be pruned after migration"


def test_cache_allow_stale_within_grace_flag_disabled(fake_env, fake_client, monkeypatch):
    monkeypatch.setenv("CACHE_ALLOW_STALE_WITHIN_GRACE", "false")
    cache_manager = NCBICacheManager(
        cache_dir=fake_env,
        default_ttl_hours=1 / 60,  # 1 minute to make entries stale quickly
        grace_period_hours=1,
        cache_warming_enabled=False,
    )

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    # First request should cache results
    scraper.search_pubmed("stale_test", max_items=5, rank=False)

    # Make the cache entry stale but within grace period
    cache_files = list(cache_manager.advanced_cache_dir.glob("*.json"))
    assert cache_files, "Expected cached entry to exist"
    cache_entry_path = cache_files[0]

    entry = json.loads(cache_entry_path.read_text())
    entry["expires_at"] = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    entry["grace_expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    cache_entry_path.write_text(json.dumps(entry))

    # Clear previous calls and add new response for the fresh request
    fake_client[0].calls.clear()
    fake_client[0].responses.append({"items": [{"title": "fresh"}], "defaultDatasetId": "dataset-fresh"})
    limiter.actions.clear()

    # Second request should not use stale cache and make fresh API call
    results = scraper.search_pubmed("stale_test", max_items=5, rank=False)
    assert results == [{"title": "fresh"}]
    assert len(fake_client[0].calls) == 1, "Expected API call when stale not allowed"
    assert limiter.actions == ["acquire"], "Expected rate limiter to be used"


def test_cache_allow_stale_within_grace_flag_enabled(fake_env, fake_client):
    cache_manager = NCBICacheManager(
        cache_dir=fake_env,
        default_ttl_hours=1 / 60,  # 1 minute to make entries stale quickly
        grace_period_hours=1,
        cache_warming_enabled=False,
        cache_allow_stale_within_grace=True,  # Explicitly enabled
    )

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    # First request should cache results
    scraper.search_pubmed("stale_test_enabled", max_items=5, rank=False)

    # Make the cache entry stale but within grace period
    cache_files = list(cache_manager.advanced_cache_dir.glob("*.json"))
    assert cache_files, "Expected cached entry to exist"
    cache_entry_path = cache_files[0]

    entry = json.loads(cache_entry_path.read_text())
    entry["expires_at"] = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    entry["grace_expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    cache_entry_path.write_text(json.dumps(entry))

    # Clear previous calls
    fake_client[0].calls.clear()
    limiter.actions.clear()

    # Second request should use stale cache without making API call
    results = scraper.search_pubmed("stale_test_enabled", max_items=5, rank=False)
    assert results == [{"title": "primary"}]
    assert len(fake_client[0].calls) == 0, "Expected no API call when stale allowed"
    assert limiter.actions == [], "Expected no rate limiter usage when serving stale"


def test_stale_hits_count_as_skipped_rate_limit(fake_env, fake_client):
    cache_manager = NCBICacheManager(
        cache_dir=fake_env,
        default_ttl_hours=1 / 60,  # 1 minute to make entries stale quickly
        grace_period_hours=1,
        cache_warming_enabled=False,
        rate_limit_integration=True,
    )

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    # First request should cache results
    scraper.search_pubmed("rate_limit_test", max_items=5, rank=False)
    initial_stats = cache_manager.get_statistics()

    # Make the cache entry stale but within grace period
    cache_files = list(cache_manager.advanced_cache_dir.glob("*.json"))
    assert cache_files, "Expected cached entry to exist"
    cache_entry_path = cache_files[0]

    entry = json.loads(cache_entry_path.read_text())
    entry["expires_at"] = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    entry["grace_expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    cache_entry_path.write_text(json.dumps(entry))

    # Clear previous calls
    fake_client[0].calls.clear()
    limiter.actions.clear()

    # Second request should use stale cache and record skipped rate limit
    results = scraper.search_pubmed("rate_limit_test", max_items=5, rank=False)
    assert results == [{"title": "primary"}]
    assert len(fake_client[0].calls) == 0, "Expected no API call when using stale cache"

    final_stats = cache_manager.get_statistics()
    # Verify that skipped_rate_limit count increased
    assert final_stats["skipped_rate_limit"] > initial_stats["skipped_rate_limit"], \
        "Expected stale hit to increase skipped_rate_limit count"
    assert final_stats["stale_hits"] > initial_stats["stale_hits"], \
        "Expected stale hit to be recorded"


def test_search_pubmed_batch_reuses_results(fake_env, fake_client):
    cache_manager = NCBICacheManager(cache_dir=fake_env)
    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    fake_client[0].responses = [
        {"items": [{"title": "oncology result"}], "defaultDatasetId": "dataset-oncology"},
        {"items": [{"title": "cardiology result"}], "defaultDatasetId": "dataset-cardiology"},
    ]
    fake_client[0].calls.clear()
    limiter.actions.clear()

    batch_requests = [
        {"query": "oncology", "max_items": 5, "rank": False},
        {"query": "oncology", "max_items": 5, "rank": False},
        {"query": "cardiology", "max_items": 5, "rank": False},
    ]

    batch_results = scraper.search_pubmed_batch(batch_requests)
    assert len(batch_results) == 3
    assert batch_results[0]["query"] == "oncology"
    assert batch_results[0]["reused"] is False
    assert batch_results[1]["reused"] is True
    assert batch_results[1]["results"] == batch_results[0]["results"]
    assert batch_results[2]["query"] == "cardiology"
    assert limiter.actions.count("acquire") == 2
    assert len(fake_client[0].calls) == 2


def test_preload_cache_reports_status(fake_env, fake_client):
    cache_manager = NCBICacheManager(cache_dir=fake_env)
    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    fake_client[0].responses = [
        {"items": [{"title": "oncology result"}], "defaultDatasetId": "dataset-oncology"},
    ]
    fake_client[0].calls.clear()
    limiter.actions.clear()

    first_summary = scraper.preload_cache(["oncology"], max_items=5)
    assert len(first_summary) == 1
    assert first_summary[0]["status"] in {"stored", "executed"}
    assert limiter.actions == ["acquire"]
    assert len(fake_client[0].calls) == 1

    limiter.actions.clear()
    fake_client[0].calls.clear()

    second_summary = scraper.preload_cache(["oncology"], max_items=5)
    assert len(second_summary) == 1
    assert second_summary[0]["status"] == "hit"
    assert limiter.actions == []
    assert fake_client[0].calls == []


def test_get_performance_metrics_reports_cache_and_rate_limit(fake_env, fake_client):
    cache_manager = NCBICacheManager(cache_dir=fake_env)
    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(cache_manager=cache_manager, rate_limiter=limiter, enable_rate_limiting=True)

    metrics_initial = scraper.get_performance_metrics()
    assert metrics_initial["cache"] == {
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
    assert metrics_initial["cache_hit_rate"] is None
    assert metrics_initial["rate_limit"]["daily_limit"] == 50
    assert metrics_initial["rate_limit"]["skipped_due_to_cache"] == 0

    fake_client[0].responses = [
        {"items": [{"title": "oncology result"}], "defaultDatasetId": "dataset-oncology"},
    ]
    scraper.search_pubmed("oncology", max_items=5, rank=False)

    metrics_after = scraper.get_performance_metrics()
    assert metrics_after["cache"]["misses"] >= 1
    assert metrics_after["cache_hit_rate"] in {0.0, None}
    assert metrics_after["cache_rate_limit_skips"] == 0
    assert metrics_after["rate_limit"]["daily_limit"] == 50
    assert metrics_after["rate_limit"]["skipped_due_to_cache"] == 0
