"""Tests for VectorDB filter visibility enhancement.

Verifies that the search_with_info method and internal _similarity_search_with_pharmaceutical_filters_with_info
method properly track and report which filters are ignored and why.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from src.vector_database import VectorDatabase
from src.pharma_utils import _PK_FILTERING_ENABLED
from src.nvidia_embeddings import NVIDIAEmbeddings


class TestVectorDBFilterVisibility:
    """Test suite for VectorDB filter visibility enhancements."""

    def test_search_with_info_no_filters(self):
        """Test search_with_info with no filters returns basic info."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()
        db.vectorstore.similarity_search.return_value = [
            Mock(page_content="doc1", metadata={}),
            Mock(page_content="doc2", metadata={})
        ]
        db._pharma_metadata_enabled = True

        # Execute
        results, info = db.search_with_info("test query", k=2)

        # Verify
        assert len(results) == 2
        assert info["pharma_metadata_enabled"] is True
        assert info["ignored_filters"] == []
        assert info["filter_results"] is None

    def test_search_with_info_pharma_metadata_disabled(self):
        """Test search_with_info when pharma metadata is disabled."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()
        db.vectorstore.similarity_search.return_value = [
            Mock(page_content="doc1", metadata={})
        ]
        db._pharma_metadata_enabled = False

        filters = {
            "drug_names": ["aspirin"],
            "species_preference": "human"
        }

        # Execute
        results, info = db.search_with_info("test query", k=2, filters=filters)

        # Verify
        assert len(results) == 1
        assert info["pharma_metadata_enabled"] is False
        assert len(info["ignored_filters"]) == 1
        assert info["ignored_filters"][0]["filter"] == "drug_names"
        assert "Pharmaceutical metadata not enabled" in info["ignored_filters"][0]["reason"]
        assert "warning" in info
        assert info["warning"] == "Pharmaceutical filters provided but metadata extraction is disabled. Filters will be ignored."

    def test_search_with_info_pk_filtering_disabled(self):
        """Test search_with_info when PK filtering is disabled."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock the similarity search to return empty list first, then with results
        db.vectorstore.similarity_search.return_value = [
            Mock(page_content="doc1", metadata={"drug_names": ["aspirin"]})
        ]
        db._pharma_metadata_enabled = True

        filters = {
            "drug_names": ["aspirin"],
            "pharmacokinetics": True
        }

        # Execute with PK filtering disabled
        with patch('src.pharma_utils._PK_FILTERING_ENABLED', False):
            results, info = db.search_with_info("test query", k=2, filters=filters)

        # Verify
        assert len(results) == 1
        assert info["pharma_metadata_enabled"] is True
        # Check that PK filter is tracked as ignored
        pk_ignored = any(
            f["filter"] == "pharmacokinetics" and "PK filtering disabled" in f["reason"]
            for f in info["ignored_filters"]
        )
        assert pk_ignored, "PK filtering should be tracked as ignored when disabled"

    def test_search_with_info_with_valid_filters(self):
        """Test search_with_info with valid pharmaceutical filters."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock that the vectorstore has documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "doc1"
        mock_doc1.metadata = {"drug_names": ["aspirin"], "species": "human"}

        mock_doc2 = Mock()
        mock_doc2.page_content = "doc2"
        mock_doc2.metadata = {"drug_names": ["ibuprofen"], "species": "rat"}

        db.vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]
        db._pharma_metadata_enabled = True

        filters = {
            "drug_names": ["aspirin"],
            "species_preference": "human"
        }

        # Execute
        results, info = db.search_with_info("test query", k=2, filters=filters)

        # Verify
        assert info["pharma_metadata_enabled"] is True
        assert info["filter_results"] is not None
        assert "Applied 2 of 2 filter categories" in info["filter_results"]
        assert "filter_stats" in info

    def test_similarity_search_with_pharmaceutical_filters_with_info_ignored_filters(self):
        """Test internal method tracks ignored filters correctly."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()
        db.vectorstore.similarity_search.return_value = []
        db._pharma_metadata_enabled = False

        filters = {
            "drug_names": ["aspirin"],
            "therapeutic_areas": ["pain"]
        }

        # Execute
        results, info = db._similarity_search_with_pharmaceutical_filters_with_info(
            "test query", k=2, filters=filters
        )

        # Verify
        assert results == []
        assert info["pharma_metadata_enabled"] is False
        assert info["original_filters_count"] == 2
        assert info["effective_filters_count"] == 0
        assert len(info["ignored_filters"]) == 2
        # Check both filters are marked as ignored
        ignored_filter_names = {f["filter"] for f in info["ignored_filters"]}
        assert "drug_names" in ignored_filter_names
        assert "therapeutic_areas" in ignored_filter_names

    def test_similarity_search_with_pharmaceutical_filters_with_info_filter_stats(self):
        """Test internal method provides filter statistics."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock documents that would be filtered
        mock_doc1 = Mock()
        mock_doc1.page_content = "doc1"
        mock_doc1.metadata = {"drug_names": ["aspirin"]}

        mock_doc2 = Mock()
        mock_doc2.page_content = "doc2"
        mock_doc2.metadata = {"drug_names": ["ibuprofen"]}

        db.vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]
        db._pharma_metadata_enabled = True

        filters = {
            "drug_names": ["aspirin"]
        }

        # Execute
        results, info = db._similarity_search_with_pharmaceutical_filters_with_info(
            "test query", k=2, filters=filters
        )

        # Verify filter stats
        assert "filter_stats" in info
        stats = info["filter_stats"]
        assert "documents_before_filter" in stats
        assert "documents_after_filter" in stats
        assert "filter_effectiveness" in stats
        assert "oversample_iterations" in stats
        assert stats["documents_before_filter"] >= stats["documents_after_filter"]

    def test_apply_pharmaceutical_filters_with_info_tracking(self):
        """Test _apply_pharmaceutical_filters_with_info tracks ignored filters."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")

        # Mock documents
        mock_doc = Mock()
        mock_doc.page_content = "doc1"
        mock_doc.metadata = {"species": "human"}

        documents = [mock_doc]
        filters = {"species_preference": "human"}
        ignored_filters = set()

        def track_ignored_filter(key: str, reason: str) -> None:
            ignored_filters.add((key, reason))

        # Execute
        with patch.object(db, '_document_matches_pharma_filters_with_info', return_value=True):
            results = db._apply_pharmaceutical_filters_with_info(
                documents, filters, ignored_filters, track_ignored_filter
            )

        # Verify
        assert len(results) == 1
        assert results[0] == mock_doc

    def test_document_matches_pharma_filters_with_info_pk_tracking(self):
        """Test _document_matches_pharma_filters_with_info tracks PK filters."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        metadata = {"pharmacokinetics": {"half_life": "2 hours"}}
        filters = {"pharmacokinetics": True}
        ignored_filters = set()

        def track_ignored_filter(key: str, reason: str) -> None:
            ignored_filters.add((key, reason))

        # Execute with PK filtering disabled
        with patch('src.pharma_utils._PK_FILTERING_ENABLED', False):
            result = db._document_matches_pharma_filters_with_info(
                metadata, filters, ignored_filters=ignored_filters,
                track_ignored_filter=track_ignored_filter
            )

        # Verify PK filter is tracked as ignored
        assert ("pharmacokinetics", "PK filtering disabled (set ENABLE_PK_FILTERING=true)") in ignored_filters
        # Result should be based on actual filter application
        assert result is True  # Since metadata has PK data