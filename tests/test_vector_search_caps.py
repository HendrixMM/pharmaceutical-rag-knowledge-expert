"""Tests for vector search oversampling caps and guards.

Verifies that the new oversampling configuration parameters properly
guard against excessive memory usage and infinite loops.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.vector_database import VectorDatabase
from src.nvidia_embeddings import NVIDIAEmbeddings


class TestVectorSearchCaps:
    """Test suite for vector search oversampling caps and guards."""

    def test_oversample_caps_enforced(self):
        """Test that oversample multiplier is capped at configured maximum."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock documents
        mock_docs = []
        for i in range(100):
            doc = Mock()
            doc.page_content = f"Document {i}"
            doc.metadata = {"drug_names": [f"drug_{i % 10}"]}
            mock_docs.append(doc)

        # Configure mock to return decreasing number of results
        db.vectorstore.similarity_search.side_effect = [
            mock_docs[:50],  # First call
            mock_docs[:20],  # Second call
            mock_docs[:5],   # Third call
        ]

        db._pharma_metadata_enabled = True

        # Test with excessive oversample value
        results = db.similarity_search_with_pharmaceutical_filters(
            "test query",
            k=5,
            filters={"drug_names": ["drug_1"]},
            oversample=100  # Way above default cap
        )

        # Should still return results
        assert len(results) <= 5
        # Should have capped the oversampling
        assert db.vectorstore.similarity_search.call_count <= 3

    def test_max_fetch_size_cap(self):
        """Test that maximum fetch size is respected."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock with many documents
        mock_docs = []
        for i in range(50000):
            doc = Mock()
            doc.page_content = f"Document {i}"
            doc.metadata = {"species": "human"}
            mock_docs.append(doc)

        # Always return all docs (simulating large index)
        db.vectorstore.similarity_search.return_value = mock_docs[:10000]
        db._pharma_metadata_enabled = True

        # Test with filters that would match few documents
        results = db.similarity_search_with_pharmaceutical_filters(
            "test query",
            k=10,
            filters={"species_preference": "mouse"},  # Few matches expected
            oversample=5
        )

        # Should not attempt to fetch more than max size
        for call in db.vectorstore.similarity_search.call_args_list:
            requested_k = call[1]['k']
            assert requested_k <= 10000, f"Requested {requested_k} exceeds max fetch size"

    def test_max_iterations_cap(self):
        """Test that maximum iterations cap is respected."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock documents that don't match filters
        mock_docs = []
        for i in range(10):
            doc = Mock()
            doc.page_content = f"Document {i}"
            doc.metadata = {"drug_names": ["other_drug"]}
            mock_docs.append(doc)

        # Always return the same non-matching docs
        db.vectorstore.similarity_search.return_value = mock_docs
        db._pharma_metadata_enabled = True

        # Patch to reduce max iterations for faster test
        with patch.object(db, 'VECTOR_DB_MAX_OVERSAMPLE_ITERATIONS', 2):
            results = db.similarity_search_with_pharmaceutical_filters(
                "test query",
                k=5,
                filters={"drug_names": ["missing_drug"]},
                oversample=5
            )

            # Should stop after max iterations
            assert db.vectorstore.similarity_search.call_count <= 2

    def test_progressive_oversampling_with_caps(self):
        """Test that progressive oversampling respects caps."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Track call arguments
        call_args_list = []

        def mock_similarity_search(query, k):
            call_args_list.append(k)
            # Return fewer results each time to trigger oversampling
            return [Mock(page_content=f"Doc {i}", metadata={"drug_names": ["aspirin"]}) for i in range(min(k, 10))]

        db.vectorstore.similarity_search.side_effect = mock_similarity_search
        db._pharma_metadata_enabled = True

        results = db.similarity_search_with_pharmaceutical_filters(
            "test query",
            k=20,  # Request more than we get back
            filters={"drug_names": ["aspirin"]},
            oversample=3
        )

        # Verify fetch sizes increase but are capped
        assert len(call_args_list) > 1
        for i, k in enumerate(call_args_list[1:], 1):
            # Each fetch should be larger but capped
            assert k <= 10000  # Max fetch size

    def test_oversample_none_uses_default(self):
        """Test that None oversample uses configured default."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        db.vectorstore.similarity_search.return_value = [
            Mock(page_content="Doc", metadata={"drug_names": ["aspirin"]})
        ]
        db._pharma_metadata_enabled = True

        # Test with None oversample
        results = db.similarity_search_with_pharmaceutical_filters(
            "test query",
            k=5,
            filters={"drug_names": ["aspirin"]},
            oversample=None
        )

        # Should use default and work normally
        assert len(results) == 1

    def test_search_with_info_includes_caps_info(self):
        """Test that search_with_info includes cap information in stats."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        db.vectorstore.similarity_search.return_value = [
            Mock(page_content="Doc", metadata={"drug_names": ["aspirin"]})
        ]
        db._pharma_metadata_enabled = True

        # Test with search_with_info
        results, info = db.search_with_info(
            "test query",
            k=5,
            filters={"drug_names": ["aspirin"]}
        )

        # Should have filter stats
        assert "filter_stats" in info
        stats = info["filter_stats"]
        assert "max_oversample_reached" in stats
        assert "iterations_exhausted" in stats

    def test_warning_logged_for_excessive_growth(self):
        """Test that warning is logged when oversampling grows too large."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")
        db.vectorstore = Mock()

        # Mock to trigger the growth warning
        db.vectorstore.similarity_search.side_effect = [
            [Mock(page_content="Doc", metadata={}) for _ in range(1000)],
            [Mock(page_content="Doc", metadata={}) for _ in range(500)],
            [Mock(page_content="Doc", metadata={}) for _ in range(200)],
        ]
        db._pharma_metadata_enabled = True

        with patch('src.vector_database.logger') as mock_logger:
            results = db.similarity_search_with_pharmaceutical_filters(
                "test query",
                k=5,
                filters={},  # No filters to trigger oversampling
                oversample=10  # High oversample
            )

            # Should log warning about excessive growth
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if "growing too large" in str(call)]
            assert len(warning_calls) > 0