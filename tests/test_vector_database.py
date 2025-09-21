from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.vector_database import VectorDatabase


class DummyEmbeddings:
    """Minimal embeddings stub for testing."""

    def embed_documents(self, texts):
        return []

    def embed_query(self, text):
        return []


def test_add_documents_enriches_when_pharma_enabled(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db"))
    doc = Document(page_content="Test", metadata={})
    db.vectorstore = MagicMock()
    db.vectorstore.add_texts.return_value = ["doc-1"]
    db._pharma_metadata_enabled = True

    db._prepare_pharmaceutical_document = MagicMock(side_effect=lambda value: value)
    db._ensure_document_metadata = MagicMock(side_effect=lambda value: value)
    db._detect_pharma_metadata = MagicMock(return_value=True)
    db._record_doc_ids = MagicMock()

    result = db.add_documents([doc])

    assert result is True
    db._prepare_pharmaceutical_document.assert_called_once_with(doc)
    db._ensure_document_metadata.assert_not_called()
    db.vectorstore.add_texts.assert_called_once()


def test_add_documents_respects_auto_enrich_flag(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db2"))
    doc = Document(page_content="Test", metadata={})
    db.vectorstore = MagicMock()
    db.vectorstore.add_texts.return_value = ["doc-2"]
    db._pharma_metadata_enabled = True

    db._prepare_pharmaceutical_document = MagicMock(side_effect=lambda value: value)
    db._ensure_document_metadata = MagicMock(side_effect=lambda value: value)
    db._detect_pharma_metadata = MagicMock(return_value=True)
    db._record_doc_ids = MagicMock()

    result = db.add_documents([doc], auto_enrich_on_add=False)

    assert result is True
    db._prepare_pharmaceutical_document.assert_not_called()
    db._ensure_document_metadata.assert_called_once_with(doc)


def test_iter_documents_uses_docstore_search_when_dict_missing(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db3"))
    document = Document(page_content="PK details", metadata={})

    class StubDocstore:
        def search(self, key):
            if key in ("", "doc-1"):
                return document
            return None

    db.vectorstore = SimpleNamespace(
        docstore=StubDocstore(),
        index_to_docstore_id={"0": "doc-1"},
    )
    db._docstore_ids = []

    docs = db._iter_documents()

    assert docs == [document]


def test_get_pharmaceutical_stats_includes_ratios(tmp_path, monkeypatch):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db_stats"))
    db.vectorstore = object()

    doc1 = Document(
        page_content="Study about ketoconazole",
        metadata={
            "drug_names": ["Ketoconazole"],
            "therapeutic_areas": ["hepatology"],
            "species": ["Human"],
            "cyp_enzymes": ["CYP3A4"],
            "ranking_score": 0.75,
            "pharmaceutical_enriched": True,
        },
    )
    doc2 = Document(
        page_content="Unannotated study",
        metadata={},
    )

    monkeypatch.setattr(db, "_iter_documents", lambda: [doc1, doc2])

    stats = db.get_pharmaceutical_stats()
    assert stats["documents_indexed"] == 2
    assert stats["drug_annotation_ratio"] == 0.5
    assert stats["pharma_metadata_ratio"] == 0.5
    ranking_stats = stats["ranking_score"]
    assert ranking_stats["average"] == 0.75
    assert ranking_stats["median"] == 0.75
    assert ranking_stats["count"] == 1


def test_pharma_filters_support_species_alias_and_enriched(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db_filters"))
    doc = Document(
        page_content="",
        metadata={
            "species": ["Human"],
            "pharmaceutical_enriched": True,
            "cyp_enzymes": ["CYP3A4"],
        },
    )

    matched = db._apply_pharmaceutical_filters(
        [doc],
        {
            "species": ["human"],
            "pharmaceutical_enriched": True,
            "cyp_enzymes": ["cyp3a4"],
        },
    )

    assert matched == [doc]


def test_species_filter_excludes_unknown_by_default(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db_species"))
    db._resolve_species = lambda _metadata, _mesh: None

    metadata = {"species": []}
    filters = {"species_preference": ["human"]}

    assert db._document_matches_pharma_filters(metadata, filters) is False

    filters_allow_unknown = {"species_preference": ["human"], "include_unknown_species": True}
    assert db._document_matches_pharma_filters(metadata, filters_allow_unknown) is True


def test_species_default_configuration_includes_unknown(tmp_path):
    db = VectorDatabase(
        DummyEmbeddings(),
        db_path=str(tmp_path / "db_species_config"),
        species_unknown_default=True,
    )
    metadata = {"species": []}
    filters = {"species_preference": ["human"]}

    assert db._document_matches_pharma_filters(metadata, filters) is True


def test_search_by_drug_name_uses_generous_oversample(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db_search"))
    calls = {}

    def fake_search(query, *, k, filters, oversample):
        calls["params"] = {
            "query": query,
            "k": k,
            "filters": filters,
            "oversample": oversample,
        }
        return []

    db.similarity_search_with_pharmaceutical_filters = fake_search  # type: ignore[assignment]

    db.search_by_drug_name("ketoconazole", k=3)
    assert calls["params"]["filters"] == {"drug_names": ["ketoconazole"]}
    assert calls["params"]["oversample"] >= 5


def test_drug_filtering_uses_normalized_storage(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db_drug_filter"))
    doc = Document(page_content="", metadata={"drug_names": "Ketoconazole"})
    ensured = db._ensure_document_metadata(doc)
    metadata = ensured.metadata
    filters = {"drug_names": ["ketoconazole"]}

    assert db._document_matches_pharma_filters(metadata, filters) is True
