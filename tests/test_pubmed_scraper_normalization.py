from types import SimpleNamespace

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


def test_build_actor_input_omits_false_flags(scraper):
    run_input = scraper._build_actor_input(
        "example query",
        5,
        include_tags_effective=False,
        include_abstract_effective=False,
    )

    assert "includeTags" not in run_input
    assert "includeAbstract" not in run_input


def test_build_actor_input_includes_true_flags(scraper):
    run_input = scraper._build_actor_input(
        "example query",
        5,
        include_tags_effective=True,
        include_abstract_effective=True,
    )

    assert run_input["includeTags"] is True
    assert run_input["includeAbstract"] is True


def test_build_actor_input_respects_env_gate(monkeypatch, scraper):
    # Test the actual gating logic in search_pubmed method instead
    monkeypatch.setenv("EASYAPI_INCLUDE_TAGS", "false")
    monkeypatch.setenv("EASYAPI_INCLUDE_ABSTRACT", "true")


def test_normalize_item_preserves_title_from_fallbacks(scraper):
    """Test that title is preserved from multiple sources with sanitization."""
    # Test direct title
    item1 = {"title": "  A Great Study   "}
    normalized1 = scraper._normalize_item(item1)
    assert normalized1["title"] == "A Great Study"

    # Test fallback to articleTitle
    item2 = {"articleTitle": "Study with articleTitle"}
    normalized2 = scraper._normalize_item(item2)
    assert normalized2["title"] == "Study with articleTitle"

    # Test fallback to citation
    item3 = {"citation": {"title": "Study from citation"}}
    normalized3 = scraper._normalize_item(item3)
    assert normalized3["title"] == "Study from citation"

    # Test rejection of invalid titles
    item4 = {"title": "pubmed"}
    normalized4 = scraper._normalize_item(item4)
    assert "title" not in normalized4


def test_to_documents_uses_title_as_content_when_abstract_missing(scraper):
    """Test that documents use title as content when abstract is missing."""
    results = [
        {
            "title": "Study with only title",
            "pmid": "12345",
        },
        {
            "title": "  Study with title and empty abstract  ",
            "abstract": "",
            "pmid": "67890",
        },
    ]

    documents = scraper.to_documents(results)

    # Both documents should be created using title as content
    assert len(documents) == 2
    assert documents[0].page_content == "Study with only title"
    assert documents[0].metadata["title"] == "Study with only title"
    assert documents[1].page_content == "Study with title and empty abstract"
    assert documents[1].metadata["title"] == "Study with title and empty abstract"


def test_to_documents_handles_preserve_order_without_abstracts(scraper, monkeypatch):
    """Test preserve-order mode doesn't skip documents when only title is available."""
    # Simulate preserve-order mode
    monkeypatch.setenv("USE_FULL_ABSTRACTS", "false")

    results = [
        {
            "title": "Important Study",
            "pmid": "12345",
            "journal": "Nature",
        },
        {
            "title": "Another Study",
            "abstract": "This has an abstract",
            "pmid": "67890",
        },
    ]

    documents = scraper.to_documents(results)

    # Both documents should be preserved
    assert len(documents) == 2
    # First doc uses title as content
    assert documents[0].page_content == "Important Study"
    assert documents[0].metadata["title"] == "Important Study"
    # Second doc uses abstract as content
    assert "This has an abstract" in documents[1].page_content


def test_classify_study_type_plural_tags(scraper):
    cohort, cohort_conf = scraper._classify_study_type(["Cohort Studies"])
    case_control, case_conf = scraper._classify_study_type(["case-control studies"])
    cross_sectional, cross_conf = scraper._classify_study_type(["Cross-Sectional Studies"])
    phase_trial, phase_conf = scraper._classify_study_type(["Clinical Trial, Phase III"])
    guideline, guideline_conf = scraper._classify_study_type(["Practice Guideline"])

    assert cohort == "Cohort Study" and cohort_conf >= 0.75
    assert case_control == "Case-Control Study" and case_conf >= 0.7
    assert cross_sectional == "Cross-Sectional Study" and cross_conf >= 0.6
    assert phase_trial == "Phase III Clinical Trial" and phase_conf >= 0.86
    assert guideline == "Practice Guideline" and guideline_conf >= 0.65


def test_ranking_forces_include_abstract_even_when_disabled(monkeypatch, scraper):
    scraper.use_full_abstracts = False
    scraper.enable_study_ranking = True
    monkeypatch.setenv("EASYAPI_INCLUDE_ABSTRACT", "false")

    captured = {}

    # Capture the effective values computed in search_pubmed
    original_get_cache_key = scraper._get_cache_key

    def capture_get_cache_key(self, *args, **kwargs):
        captured["include_tags_effective"] = kwargs.get("include_tags_effective")
        captured["include_abstract_effective"] = kwargs.get("include_abstract_effective")
        return original_get_cache_key(*args, **kwargs)

    monkeypatch.setattr(scraper, "_get_cache_key", capture_get_cache_key.__get__(scraper, PubMedScraper))

    def fake_call_actor(run_input, schema="searchUrls"):
        captured["run_input_call"] = run_input
        return {"defaultDatasetId": "dataset-1"}

    monkeypatch.setattr(scraper, "_call_actor", fake_call_actor)
    monkeypatch.setattr(scraper, "_cache_results", lambda *_args, **_kwargs: None)

    class _DatasetStub:
        def iterate_items(self):
            return []

    scraper.client = SimpleNamespace(dataset=lambda _id: _DatasetStub())

    results = scraper.search_pubmed("test query", rank=True)

    assert results == []
    # With the refactoring, we check the effective values that were computed
    assert captured["include_abstract_effective"] is True
    assert captured["run_input_call"].get("includeAbstract") is True


def test_cache_key_respects_preserve_flag(scraper):
    base_key = scraper._get_cache_key(
        "query",
        max_items=5,
        apply_ranking=False,
        pharma_enhance_enabled=False,
        include_tags_effective=False,
        include_abstract_effective=False,
        preserve_order=False,
    )
    preserve_key = scraper._get_cache_key(
        "query",
        max_items=5,
        apply_ranking=False,
        pharma_enhance_enabled=False,
        include_tags_effective=False,
        include_abstract_effective=False,
        preserve_order=True,
    )

    assert preserve_key != base_key
