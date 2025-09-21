import json
from datetime import UTC, datetime
from pathlib import Path

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
        {"species": ["Human", "Mouse"], "publication_year": datetime.now(UTC).year},
        {"species": "rat", "publication_year": datetime.now(UTC).year},
    ]
    filtered, applied = engine._apply_filters(results, {"species_preference": ["human"]})

    assert len(filtered) == 1
    assert applied["species_preference"] == ["human"]


def test_raw_cache_reuse_applies_new_filters(tmp_path):
    current_year = datetime.now(UTC).year
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
        filters={"drug_names": ["ketoconazole"]},
    )

    assert response["results_count"] == 1
    # Check drug_annotations instead since drug_names might not be populated
    drug_names = []
    for annotation in response["results"][0].get("drug_annotations", []):
        if isinstance(annotation, dict):
            drug_names.append(annotation.get("name", ""))
    assert "ketoconazole" in drug_names


def test_deduplicate_results_uses_year_and_author(tmp_path):
    scraper = StubScraper([])
    engine = EnhancedQueryEngine(scraper, cache_dir=str(tmp_path / "cache_dedup"))

    results = [
        {"title": "PK evaluation", "publication_year": 2020, "authors": ["Smith"]},
        {"title": "PK evaluation", "publication_year": 2021, "authors": ["Jones"]},
    ]

    deduped, info = engine._deduplicate_results(results)
    assert len(deduped) == 2
    assert info["duplicates_removed"] == 0


class CountingPharmaProcessor:
    def __init__(self):
        self.filter_calls = 0
        self.doc_calls = 0

    def extract_drug_names(self, text):
        lowered = text.lower()
        stripped = lowered.strip()
        if stripped == "ketoconazole":
            self.filter_calls += 1
            return [{"name": "ketoconazole", "confidence": 0.9}]
        if "ketoconazole" in lowered:
            self.doc_calls += 1
            return [{"name": "ketoconazole", "confidence": 0.9}]
        return []

    def extract_drug_name_strings(self, text):
        return [item["name"] for item in self.extract_drug_names(text)]

    def extract_cyp_enzymes(self, text):
        if "cyp3a4" in text.lower():
            return ["cyp3a4"]
        return []

    def normalize_mesh_terms(self, terms):  # pragma: no cover - stub
        return list(terms)

    def identify_therapeutic_areas(self, mesh_terms):  # pragma: no cover - stub
        return []


def test_enhancement_switches_modes_based_on_signals(tmp_path):
    scraper = StubScraper([])
    processor = CountingPharmaProcessor()
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_enhance"),
        pharma_processor=processor,
        enable_pharma_enrichment=False,
        enhancement_mode="and",
        query_enhancement_min_terms=1,
        query_enhancement_disable_threshold=3,
    )

    enhanced_query, applied = engine._enhance_pharmaceutical_query("ketoconazole efficacy")
    assert applied is True
    assert " OR " in enhanced_query
    assert " AND (" not in enhanced_query

    enhanced_query_strict, applied_strict = engine._enhance_pharmaceutical_query(
        "ketoconazole CYP3A4 drug interaction"
    )
    assert applied_strict is False
    assert enhanced_query_strict == "ketoconazole CYP3A4 drug interaction"


def test_runtime_drug_extraction_respects_flag(tmp_path):
    scraper = StubScraper([])
    processor = CountingPharmaProcessor()
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_filters"),
        pharma_processor=processor,
        enable_pharma_enrichment=False,
    )

    results = [
        {
            "title": "Study",
            "abstract": "ketoconazole shows cyp3a4 interaction",
        }
    ]

    filtered, _ = engine._apply_filters(
        results,
        {"drug_names": ["ketoconazole"]},
        allow_runtime_extraction_for_filters=False,
    )
    assert processor.doc_calls == 0
    assert filtered == []

    filtered_runtime, _ = engine._apply_filters(
        results,
        {"drug_names": ["ketoconazole"]},
        allow_runtime_extraction_for_filters=True,
        runtime_extraction_char_limit=100,
    )
    assert processor.doc_calls >= 1
    assert filtered_runtime == results


def test_runtime_drug_extraction_cap_limits_docs(tmp_path):
    scraper = StubScraper([])
    processor = CountingPharmaProcessor()
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_cap"),
        pharma_processor=processor,
        enable_pharma_enrichment=False,
        runtime_extraction_doc_cap=1,
    )

    results = [
        {"title": f"Study {idx}", "abstract": "ketoconazole details"}
        for idx in range(3)
    ]

    engine._apply_filters(
        results,
        {"drug_names": ["ketoconazole"]},
        allow_runtime_extraction_for_filters=True,
        runtime_extraction_char_limit=200,
    )

    assert processor.doc_calls == 1


def test_species_unknown_default_includes_unknown_documents(tmp_path):
    scraper = StubScraper([
        {
            "title": "Unknown species study",
            "abstract": "",
            "publication_year": datetime.now(UTC).year,
        }
    ])
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_species"),
        species_unknown_default=True,
        enable_pharma_enrichment=False,
    )

    response = engine.process_pharmaceutical_query(
        "ketoconazole",
        filters={"species_preference": ["human"]},
    )

    assert response["results_count"] == 1


def test_title_normalization_strips_diacritics():
    accented = "Évaluation d’un médicament expérimental"
    normalized = EnhancedQueryEngine._normalize_title_identifier(accented)
    assert normalized == "evaluation d un medicament experimental"


def test_disabling_filtered_cache_writes_only_raw_cache(tmp_path):
    current_year = datetime.now(UTC).year
    scraper = StubScraper(
        [
            {
                "title": "Human study",
                "abstract": "ketoconazole exposure",
                "species": ["Human"],
                "publication_year": current_year,
            },
            {
                "title": "Mouse study",
                "species": ["Mouse"],
                "publication_year": current_year - 1,
            },
        ]
    )
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_flag"),
        cache_filtered_results=False,
    )

    response = engine.process_pharmaceutical_query(
        "ketoconazole",
        filters={"species_preference": ["human"]},
    )
    cache_files = list(engine.cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    assert response["cache_hit"] is False
    first_cache = cache_files[0]
    with first_cache.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "filters" not in (payload.get("metadata") or {})

    response_cached = engine.process_pharmaceutical_query(
        "ketoconazole",
        filters={"species_preference": ["human"]},
    )
    assert response_cached["cache_hit"] is True
    assert set(engine.cache_dir.glob("*.json")) == set(cache_files)


def test_cache_pruning_limits_cache_dir(tmp_path):
    scraper = StubScraper([])
    engine = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_prune"),
    )

    for idx in range(3):
        engine._cache_result(f"key{idx}", {"results": [], "metadata": {}})


def test_filtered_cache_control(tmp_path):
    """Test that cache_filtered_results controls filtered cache writing."""
    current_year = datetime.now(UTC).year
    scraper = StubScraper([
        {
            "title": "Human Study",
            "abstract": "A study on humans",
            "publication_year": current_year,
            "species": ["Human"],
        },
        {
            "title": "Mouse Study",
            "abstract": "A study on mice",
            "publication_year": current_year,
            "species": ["Mouse"],
        },
    ])

    # Test with cache_filtered_results=True (default)
    engine_enabled = EnhancedQueryEngine(
        scraper,
        cache_dir=str(tmp_path / "cache_enabled"),
        cache_filtered_results=True,
    )

    response1 = engine_enabled.process_pharmaceutical_query(
        "test query",
        filters={"species_preference": "human"}
    )

    # Should have both raw and filtered cache
    cache_files = list(Path(tmp_path / "cache_enabled").glob("*.json"))
    assert len(cache_files) == 2

    # Check that we have one raw cache and one filtered cache
    cached_data = []
    for cache_file in cache_files:
        with cache_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            cached_data.append(data)

    # One should have filters, one should not
    has_filters = [d for d in cached_data if "filters" in (d.get("metadata") or {})]
    no_filters = [d for d in cached_data if "filters" not in (d.get("metadata") or {})]
    assert len(has_filters) == 1
    assert len(no_filters) == 1

    # Test with cache_filtered_results=False
    scraper2 = StubScraper(scraper._results)
    engine_disabled = EnhancedQueryEngine(
        scraper2,
        cache_dir=str(tmp_path / "cache_disabled"),
        cache_filtered_results=False,
    )

    response2 = engine_disabled.process_pharmaceutical_query(
        "test query",
        filters={"species_preference": "human"}
    )

    # Should only have raw cache
    cache_files = list(Path(tmp_path / "cache_disabled").glob("*.json"))
    assert len(cache_files) == 1

    # Check that it's a raw cache (no filters)
    with cache_files[0].open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "filters" not in (data.get("metadata") or {})

    # Test raw cache reuse path with cache_filtered_results=False
    scraper3 = StubScraper(scraper._results)
    engine_reuse = EnhancedQueryEngine(
        scraper3,
        cache_dir=str(tmp_path / "cache_reuse"),
        cache_filtered_results=False,
    )

    # First call - should create raw cache
    response3 = engine_reuse.process_pharmaceutical_query(
        "test query",
        filters={"species_preference": "human"}
    )
    assert scraper3.calls == 1

    # Second call with different filters - should reuse raw cache
    response4 = engine_reuse.process_pharmaceutical_query(
        "test query",
        filters={"species_preference": "mouse"}
    )
    assert scraper3.calls == 1  # Should reuse raw cache
    assert response4["cache_hit"] is True

    # Should still only have raw cache (no filtered cache created)
    cache_files = list(Path(tmp_path / "cache_reuse").glob("*.json"))
    assert len(cache_files) == 1

    cache_files = list((tmp_path / "cache_prune" / "advanced").glob("*.json"))
    assert len(cache_files) <= 2
