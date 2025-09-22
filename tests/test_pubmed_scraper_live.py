import os
from datetime import datetime
from pathlib import Path

import pytest

from src.pubmed_scraper import PubMedScraper


pytestmark = pytest.mark.skip(
    reason="Apify-based live tests deprecated; E-utilities is the primary source."
)

TEST_QUERIES = ("cancer treatment", "diabetes management")


def _ensure_apify_token(monkeypatch) -> str:
    pytest.skip("Apify removed")


def test_pubmed_scraper_live_queries(monkeypatch, tmp_path):
    pytest.skip("Apify removed")
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
