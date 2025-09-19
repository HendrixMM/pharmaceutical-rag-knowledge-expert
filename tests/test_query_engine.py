from datetime import datetime

from src.query_engine import EnhancedQueryEngine
from src.ranking_filter import StudyRankingFilter


class StubScraper:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def search_pubmed(self, query, max_items=30):
        self.calls += 1
        return list(self._results)


def test_species_filter_accepts_iterable_preferences(tmp_path):
    scraper = StubScraper([])
    engine = EnhancedQueryEngine(scraper, cache_dir=str(tmp_path / "cache"))

    results = [
        {"species": ["Human", "Mouse"], "publication_year": datetime.utcnow().year},
        {"species": "rat", "publication_year": datetime.utcnow().year},
    ]
    filtered, applied = engine._apply_filters(results, {"species_preference": ["human"]})

    assert len(filtered) == 1
    assert applied["species_preference"] == ["human"]


def test_raw_cache_reuse_applies_new_filters(tmp_path):
    current_year = datetime.utcnow().year
    scraper = StubScraper(
        [
            {
                "title": "Study A",
                "abstract": "n=40 humans",
                "publication_year": current_year,
                "species": ["Human"],
            },
            {
                "title": "Study B",
                "abstract": "mouse study",
                "publication_year": current_year - 1,
                "species": ["Mouse"],
            },
        ]
    )
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache2"),
        ranking_filter=StudyRankingFilter(),
    )

    response_human = engine.process_pharmaceutical_query(
        "drug exposure",
        filters={"species_preference": "human"},
    )
    assert scraper.calls == 1
    assert response_human["cache_hit"] is False
    assert response_human["results_count"] == 1

    response_mouse = engine.process_pharmaceutical_query(
        "drug exposure",
        filters={"species_preference": ["mouse"]},
    )
    assert scraper.calls == 1  # raw cache reused
    assert response_mouse["cache_hit"] is True
    assert response_mouse["results_count"] == 1


def test_drug_filter_expands_aliases(tmp_path):
    from src.pharmaceutical_processor import PharmaceuticalProcessor

    generic_path = tmp_path / "generic.txt"
    brand_path = tmp_path / "brand.txt"
    generic_path.write_text("ketoconazole\n")
    brand_path.write_text("Nizoral\n")

    processor = PharmaceuticalProcessor(
        generic_lexicon_path=str(generic_path),
        brand_lexicon_path=str(brand_path),
    )

    scraper = StubScraper([
        {
            "title": "Azole inhibitor study",
            "abstract": "CYP3A4 inhibition observed with Nizoral (ketoconazole).",
            "drug_annotations": [{"name": "ketoconazole", "type": "generic"}],
        },
        {
            "title": "Control study",
            "abstract": "Placebo group without azole treatment.",
        },
    ])

    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_drug"),
        pharma_processor=processor,
    )

    response = engine.process_pharmaceutical_query(
        "azole therapy",
        filters={"drug_names": ["Nizoral"]},
    )

    assert response["results_count"] == 1
    assert "ketoconazole" in response["results"][0]["drug_names"]
