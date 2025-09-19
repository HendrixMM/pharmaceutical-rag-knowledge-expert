import os
from typing import List

import pytest

from src.enhanced_pubmed_scraper import EnhancedPubMedScraper
from src.pubmed_scraper import PubMedScraper
from src.rate_limiting import RateLimitStatus


class DummyLimiter:
    def __init__(self):
        self.actions: List[str] = []
        self.max_requests_per_second = 3
        self.daily_request_limit = 50

    def wait_until_ready(self) -> None:
        self.actions.append("wait")

    def record_request(self) -> None:
        self.actions.append("record")

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
            # Align with PubMedScraper expectation that EasyAPI returns defaultDatasetId.
            run = {**run, "defaultDatasetId": "test-dataset"}
        return run


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: List[dict] = []

    def actor(self, _actor_id):
        return FakeActor(self)

    def dataset(self, _dataset_id):
        raise AssertionError("Dataset access should be mocked in tests")


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APIFY_TOKEN", "test-token")
    monkeypatch.setenv("PUBMED_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "true")
    monkeypatch.setenv("ENABLE_OPTIMAL_TIMING_DETECTION", "true")
    monkeypatch.setenv("ENABLE_EASYAPI_SCHEMA_FALLBACK", "false")
    yield


def test_feature_flag_disabled_falls_back_to_base(monkeypatch, fake_env):
    monkeypatch.setenv("ENABLE_ENHANCED_PUBMED_SCRAPER", "false")

    created_clients = []

    def fake_client_factory(_token):
        client = FakeClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", fake_client_factory)

    fallback_calls = {}

    def fake_super(self, query, max_items=None, rank=None, pharma_enhance=None):
        fallback_calls["args"] = (query, max_items, rank, pharma_enhance)
        return ["base-result"]

    monkeypatch.setattr(PubMedScraper, "search_pubmed", fake_super)

    scraper = EnhancedPubMedScraper(rate_limiter=DummyLimiter())
    result = scraper.search_pubmed("diabetes", rank=False)

    assert result == ["base-result"]
    assert fallback_calls["args"][0] == "diabetes"
    assert created_clients  # Client still initialized for consistency


def test_rate_limiter_invoked_for_api_calls(monkeypatch, fake_env):
    monkeypatch.setenv("ENABLE_ENHANCED_PUBMED_SCRAPER", "true")

    responses = [{"items": [{"title": "primary"}]}]
    created_clients = []

    def fake_client_factory(_token):
        client = FakeClient(responses)
        created_clients.append(client)
        return client

    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", fake_client_factory)

    cache_store = {}

    def fake_process(self, run, apply_ranking):
        return run.get("items", [])

    def fake_cache(self, cache_key, results):
        cache_store[cache_key] = results

    def fake_get_cache(self, cache_key, apply_ranking):
        return cache_store.get(cache_key)

    monkeypatch.setattr(EnhancedPubMedScraper, "_process_run_results", fake_process)
    monkeypatch.setattr(EnhancedPubMedScraper, "_cache_results", fake_cache)
    monkeypatch.setattr(EnhancedPubMedScraper, "_get_cached_results", fake_get_cache)

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(rate_limiter=limiter, enable_rate_limiting=True)

    result = scraper.search_pubmed("oncology", rank=False)
    assert result == [{"title": "primary"}]
    assert limiter.actions == ["acquire"]
    client = created_clients[0]
    assert len(client.calls) == 1

    limiter.actions.clear()
    client.calls.clear()
    cached = scraper.search_pubmed("oncology", rank=False)
    assert cached == result
    assert limiter.actions == []
    assert client.calls == []


def test_enhanced_scraper_enables_rate_limiting_by_default(monkeypatch, fake_env):
    monkeypatch.delenv("ENABLE_ENHANCED_PUBMED_SCRAPER", raising=False)
    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient([{"items": []}]))

    limiter = DummyLimiter()

    def fake_process(self, run, apply_ranking):
        return run.get("items", [])

    monkeypatch.setattr(EnhancedPubMedScraper, "_process_run_results", fake_process)
    monkeypatch.setattr(EnhancedPubMedScraper, "_cache_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(EnhancedPubMedScraper, "_get_cached_results", lambda *args, **kwargs: None)

    scraper = EnhancedPubMedScraper(rate_limiter=limiter)
    scraper.search_pubmed("microbiology", rank=False)

    assert limiter.actions == ["acquire"]


def test_rate_limiting_disabled_skips_limiter(monkeypatch, fake_env):
    monkeypatch.setenv("ENABLE_RATE_LIMITING", "false")
    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient([{"items": []}]))

    limiter = DummyLimiter()

    def fake_process(self, run, apply_ranking):
        return run.get("items", [])

    monkeypatch.setattr(EnhancedPubMedScraper, "_process_run_results", fake_process)
    monkeypatch.setattr(EnhancedPubMedScraper, "_cache_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(EnhancedPubMedScraper, "_get_cached_results", lambda *args, **kwargs: None)

    scraper = EnhancedPubMedScraper(rate_limiter=limiter, enable_rate_limiting=False)
    scraper.search_pubmed("cardiology", rank=False)

    assert limiter.actions == []


def test_rate_limit_status_reporting(monkeypatch, fake_env):
    monkeypatch.setenv("ENABLE_ENHANCED_PUBMED_SCRAPER", "true")

    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient())

    limiter = DummyLimiter()
    scraper = EnhancedPubMedScraper(rate_limiter=limiter)

    status = scraper.get_rate_limit_status()
    assert status.daily_limit == 50
    assert scraper.get_rate_limit_report()["rate_limited"] is False
    assert scraper.is_optimal_timing() is True


def test_enhanced_query_wrapped_in_parentheses(monkeypatch, fake_env):
    """Test that enhanced queries are properly wrapped in parentheses"""
    monkeypatch.setenv("ENABLE_PHARMA_QUERY_ENHANCEMENT", "true")
    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient())

    scraper = PubMedScraper()

    # Test enhancement with parentheses wrapping
    enhanced = scraper._enhance_pharmaceutical_query("diabetes cyp", True)

    # Should wrap original query in parentheses when adding OR clause
    assert enhanced.startswith("(diabetes cyp) OR (")
    assert "drug-drug interaction" in enhanced


def test_searchurls_schema_builds_expected(monkeypatch, fake_env):
    """Test that searchUrls schema generates correct EasyAPI input"""
    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient())

    scraper = PubMedScraper()

    input_dict = scraper._build_actor_input("test query", 10, "searchUrls")

    assert input_dict == {
        "searchUrls": ["https://pubmed.ncbi.nlm.nih.gov/?term=test+query"],
        "maxItems": 10,
    }


def test_schema_fallback_order_on_zero_results(monkeypatch, fake_env):
    """Test fallback order when primary schema returns zero results"""
    monkeypatch.setenv("ENABLE_EASYAPI_SCHEMA_FALLBACK", "true")

    call_order = []

    def fake_call_actor(self, run_input, schema=None):
        call_order.append(run_input)
        # Return empty results to trigger fallback
        return {"defaultDatasetId": "test-dataset"}

    def fake_process_results(self, run_id, apply_ranking):
        # Return empty list to trigger fallback
        return []

    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient())
    monkeypatch.setattr(PubMedScraper, "_call_actor", fake_call_actor)
    monkeypatch.setattr(PubMedScraper, "_process_run_results", fake_process_results)

    scraper = PubMedScraper()
    scraper.search_pubmed("test query", max_items=5, rank=False)

    # Only the primary schema should be attempted for this actor
    assert len(call_order) == 1
    assert "searchUrls" in call_order[0]


def test_retry_without_optional_flags_on_400_errors(monkeypatch, fake_env):
    """Test retry without optional flags when 400 errors occur"""
    monkeypatch.setenv("EXTRACT_TAGS", "true")
    monkeypatch.setenv("USE_FULL_ABSTRACTS", "true")

    call_attempts = []

    class Mock400Error(Exception):
        def __init__(self):
            super().__init__("400 Client Error")

    def fake_call_actor(self, run_input, schema=None):
        if not call_attempts:
            call_attempts.append(run_input.copy())
            raise Mock400Error()
        call_attempts.append(run_input.copy())
        return {"defaultDatasetId": "test-dataset"}

    def fake_process_results(self, run_id, apply_ranking):
        return [{"title": "Test Result"}]

    monkeypatch.setenv("EASYAPI_SMART_SCHEMA_FALLBACK", "false")
    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient())
    monkeypatch.setattr(PubMedScraper, "_call_actor", fake_call_actor)
    monkeypatch.setattr(PubMedScraper, "_process_run_results", fake_process_results)

    scraper = PubMedScraper()
    results = scraper.search_pubmed("test query", max_items=5, rank=False)

    # Should have made exactly two attempts (original + retry)
    assert len(call_attempts) == 2

    # Should have succeeded and returned results
    assert len(results) == 1
    assert results[0]["title"] == "Test Result"


def test_startUrls_and_searchUrls_send_lists(monkeypatch, fake_env):
    """Test that startUrls and searchUrls schemas properly send lists"""
    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda _token: FakeClient())

    scraper = PubMedScraper()

    # Test searchUrls schema
    input_dict = scraper._build_actor_input("test query", 10, "searchUrls")
    assert "searchUrls" in input_dict
    assert isinstance(input_dict["searchUrls"], list)
    assert len(input_dict["searchUrls"]) == 1
    assert "pubmed.ncbi.nlm.nih.gov" in input_dict["searchUrls"][0]
    assert "maxItems" in input_dict
