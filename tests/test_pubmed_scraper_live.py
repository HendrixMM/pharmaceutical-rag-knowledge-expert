import os
from datetime import datetime
from pathlib import Path

import pytest

from src.pubmed_scraper import PubMedScraper


REQUIRED_ENV = "APIFY_TOKEN"
LIVE_FLAG_ENV = "ENABLE_PUBMED_LIVE_TESTS"
TRUTHY_VALUES = {"1", "true", "yes", "on"}
RUN_LIVE_TEST = os.getenv(LIVE_FLAG_ENV, "").strip().lower() in TRUTHY_VALUES

pytestmark = pytest.mark.skipif(
    not RUN_LIVE_TEST,
    reason="PubMed live tests require ENABLE_PUBMED_LIVE_TESTS=true and a valid APIFY_TOKEN",
)

TEST_QUERIES = ("cancer treatment", "diabetes management")


def _ensure_apify_token(monkeypatch) -> str:
    token = os.getenv(REQUIRED_ENV)
    if token:
        return token

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == REQUIRED_ENV:
                token_value = value.split("#", 1)[0].strip()
                if token_value:
                    monkeypatch.setenv(REQUIRED_ENV, token_value)
                    return token_value

    pytest.skip("APIFY_TOKEN required for live PubMed scraper integration test")


def test_pubmed_scraper_live_queries(monkeypatch, tmp_path):
    if not RUN_LIVE_TEST:
        pytest.skip(
            "ENABLE_PUBMED_LIVE_TESTS flag not enabled; skipping live PubMed scraper integration test",
        )
    _ensure_apify_token(monkeypatch)
    monkeypatch.setenv("PUBMED_CACHE_DIR", str(tmp_path / "pubmed_live_cache"))

    scraper = PubMedScraper()

    upper_year_bound = datetime.now().year + 1

    for query in TEST_QUERIES:
        results = scraper.search_pubmed(query, max_items=5)
        assert results, f"Expected results for query '{query}'"

        journals = [entry.get("journal") for entry in results if entry.get("journal")]
        assert journals, f"Expected at least one structured journal for query '{query}'"

        years = []
        for entry in results:
            year_value = entry.get("year")
            if not year_value:
                continue
            try:
                year_int = int(str(year_value))
            except ValueError:
                continue
            assert 1900 <= year_int <= upper_year_bound
            years.append(year_int)

        assert years, f"Expected at least one valid year extracted for query '{query}'"
