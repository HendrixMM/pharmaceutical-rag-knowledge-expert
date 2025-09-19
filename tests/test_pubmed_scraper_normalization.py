import os

import pytest

from src.pubmed_scraper import PubMedScraper


class _DummyApifyClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def actor(self, *_args, **_kwargs):
        raise RuntimeError("actor should not be invoked in normalization tests")

    def dataset(self, *_args, **_kwargs):
        raise RuntimeError("dataset should not be invoked in normalization tests")


@pytest.fixture
def scraper(monkeypatch, tmp_path):
    monkeypatch.setenv("APIFY_TOKEN", "test-token")
    monkeypatch.setenv("PUBMED_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EASYAPI_FALLBACK_SCHEMAS", "searchQuery,startUrls")
    monkeypatch.setenv("PHARMA_MAX_TERMS", "8")

    monkeypatch.setattr("src.pubmed_scraper.ApifyClient", lambda *_args, **_kwargs: _DummyApifyClient())

    return PubMedScraper()


def test_normalize_authors_list(scraper):
    authors = [
        {"fullName": "Alice Doe"},
        {"name": "Bob Roe"},
        "Charlie Foo",
    ]

    normalized = scraper._normalize_authors(authors)
    assert normalized == "Alice Doe, Bob Roe, Charlie Foo"


def test_normalize_item_authors_dict_with_list(scraper):
    item = {
        "title": "Sample Study",
        "authors": {
            "list": [
                {"fullName": "Alice Doe"},
                {"short": "B. Roe"},
                "Charlie Foo",
            ]
        },
    }

    normalized = scraper._normalize_item(item)

    assert normalized["authors"] == "Alice Doe, B. Roe, Charlie Foo"
    assert normalized["authors_list"] == ["Alice Doe", "B. Roe", "Charlie Foo"]


def test_normalize_item_authors_string(scraper):
    item = {
        "title": "Another Study",
        "authors": "Alice Doe; Bob Roe; Carol Foo",
    }

    normalized = scraper._normalize_item(item)

    assert normalized["authors"] == "Alice Doe; Bob Roe; Carol Foo"
    assert "authors_list" not in normalized
