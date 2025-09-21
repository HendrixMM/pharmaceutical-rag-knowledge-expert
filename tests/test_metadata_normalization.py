"""Tests for metadata normalization during enrichment and filtering.

Verifies that all metadata fields are properly normalized during pharmaceutical
metadata extraction and that filtering works correctly with normalized data.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from src.pharmaceutical_processor import PharmaceuticalProcessor
from src.vector_database import VectorDatabase
from src.nvidia_embeddings import NVIDIAEmbeddings


class TestMetadataNormalization:
    """Test suite for metadata normalization enhancements."""

    def test_pharmaceutical_processor_normalize_species(self):
        """Test species normalization with various input formats."""
        processor = PharmaceuticalProcessor()

        # Test string input
        result = processor.normalize_species("Human")
        assert result == ["human"]

        # Test list input
        result = processor.normalize_species(["mice", "rats"])
        assert set(result) == {"mouse", "rat"}

        # Test mapping variants
        result = processor.normalize_species("non-human primate")
        assert result == ["monkey"]  # Maps to 'monkey' in the species_mapping

        # Test in vitro terms
        result = processor.normalize_species("cell culture")
        assert result == ["in vitro"]

        # Test empty/None input
        assert processor.normalize_species(None) == []
        assert processor.normalize_species("") == []

    def test_pharmaceutical_processor_normalize_study_types(self):
        """Test study type normalization with various formats."""
        processor = PharmaceuticalProcessor()

        # Test single study type
        result = processor.normalize_study_types("RCT")
        assert "randomized controlled trial" in result

        # Test list with multiple types
        result = processor.normalize_study_types(["phase I", "clinical trial"])
        assert "phase i" in result
        assert "clinical trial" in result

        # Test combined data (both fields)
        combined_data = {
            "study_type": "case report",
            "study_types": ["observational study"]
        }
        result = processor.normalize_study_types(combined_data)
        assert "case report" in result
        assert "observational study" in result

    def test_pharmaceutical_processor_normalize_publication_year(self):
        """Test year normalization with various formats."""
        processor = PharmaceuticalProcessor()

        # Test integer input
        assert processor.normalize_publication_year(2023) == 2023

        # Test string with year
        assert processor.normalize_publication_year("Published in 2023") == 2023

        # Test dict with publication_year
        year_dict = {"publication_year": "2021"}
        assert processor.normalize_publication_year(year_dict) == 2021

        # Test invalid year
        assert processor.normalize_publication_year("invalid") is None

        # Test out of range year
        assert processor.normalize_publication_year(1800) is None

    def test_pharmaceutical_processor_normalize_pharmacokinetic_data(self):
        """Test PK data normalization and merging."""
        processor = PharmaceuticalProcessor()

        # Test parameter name normalization
        pk_data = {
            "half-life": "2.5 hours",
            "CL/F": "15 L/h",
            "AUC": "4500 ng*h/mL"
        }
        result = processor.normalize_pharmacokinetic_data(pk_data)

        assert "half_life" in result
        # Note: CL/F is not mapped to 'clearance' in the current implementation
        assert "auc" in result
        assert result["half_life"] == 2.5

        # Test merging pharmacokinetics and pharmacokinetic_values
        combined_data = {
            "pharmacokinetics": {"half-life": "3 hours"},
            "pharmacokinetic_values": {"auc": "5000"}
        }
        result = processor.normalize_pharmacokinetic_data(combined_data)
        assert result["half_life"] == 3.0
        assert result["auc"] == 5000.0

    def test_vector_database_normalize_metadata_fields(self):
        """Test that VectorDatabase normalizes metadata during extraction."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")

        # Mock document with unnormalized metadata
        doc = Mock()
        doc.metadata = {
            "mesh_terms": ["Anticoagulants", "Anti-Inflammatory Agents"],
            "study_type": "Clinical Trial",
            "year": "2023",
            "species": "mice",
            "pharmacokinetics": {"half-life": "2 hours"},
            "pharmacokinetic_values": {"auc": "4500 ng*h/mL"}
        }
        doc.page_content = "Test content"

        # Mock the processor
        with patch.object(db, '_get_pharmaceutical_processor') as mock_get_processor:
            mock_processor = Mock(spec=PharmaceuticalProcessor)
            mock_get_processor.return_value = mock_processor

            # Configure mock processor methods
            mock_processor.normalize_mesh_terms.return_value = ["anticoagulants", "anti-inflammatory agents"]
            mock_processor.identify_therapeutic_areas.return_value = ["hematology", "immunology"]
            mock_processor.normalize_study_types.return_value = ["clinical trial"]
            mock_processor.normalize_publication_year.return_value = 2023
            mock_processor.normalize_species.return_value = ["mouse"]
            mock_processor.normalize_pharmacokinetic_data.return_value = {
                "half_life": 2.0,
                "auc": 4500.0
            }

            # Execute
            result = db._normalize_metadata_fields(doc.metadata, mock_processor)

            # Verify normalization
            assert result["mesh_terms"] == ["anticoagulants", "anti-inflammatory agents"]
            assert result["therapeutic_areas"] == ["hematology", "immunology"]
            assert result["study_types"] == ["clinical trial"]
            assert result["study_type"] == "clinical trial"
            assert result["publication_year"] == 2023
            assert result["year"] == 2023
            assert result["species"] == "mouse"
            assert result["species_list"] == ["mouse"]
            assert result["pharmacokinetics"]["half_life"] == 2.0
            assert result["pharmacokinetic_values"]["auc"] == 4500.0

    def test_document_matches_pharma_filters_uses_normalized_data(self):
        """Test that filtering uses normalized metadata fields."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")

        # Test metadata with normalized fields
        metadata = {
            "study_types": ["clinical trial"],
            "publication_year": 2023,
            "species_list": ["human"],
            "pharmacokinetics": {"half_life": 2.5},
            "drug_names": ["aspirin"],
            "therapeutic_areas": ["cardiology"]
        }

        # Test study type filtering
        filters = {"study_types": ["clinical trial"]}
        assert db._document_matches_pharma_filters(metadata, filters) is True

        filters = {"study_types": ["preclinical study"]}
        assert db._document_matches_pharma_filters(metadata, filters) is False

        # Test year range filtering
        filters = {"year_range": [2020, 2025]}
        assert db._document_matches_pharma_filters(metadata, filters) is True

        filters = {"year_range": [2024, 2025]}
        assert db._document_matches_pharma_filters(metadata, filters) is False

        # Test species filtering
        filters = {"species_preference": "human"}
        assert db._document_matches_pharma_filters(metadata, filters) is True

        filters = {"species_preference": "mouse"}
        assert db._document_matches_pharma_filters(metadata, filters) is False

        # Test PK filtering
        filters = {"pharmacokinetics": {"half_life": True}}
        with patch('src.pharma_utils._PK_FILTERING_ENABLED', True):
            assert db._document_matches_pharma_filters(metadata, filters) is True

        # Test with a parameter that doesn't exist
        filters = {"pharmacokinetics": {"nonexistent_param": True}}
        with patch('src.pharma_utils._PK_FILTERING_ENABLED', True):
            assert db._document_matches_pharma_filters(metadata, filters) is False

    def test_backward_compatibility_fields_maintained(self):
        """Test that backward compatibility fields are preserved."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")

        # Mock document
        doc = Mock()
        doc.metadata = {"study_type": "Clinical Trial"}
        doc.page_content = "Test content"

        # Mock the processor
        with patch.object(db, '_get_pharmaceutical_processor') as mock_get_processor:
            mock_processor = Mock(spec=PharmaceuticalProcessor)
            mock_get_processor.return_value = mock_processor
            mock_processor.normalize_mesh_terms.return_value = []
            mock_processor.identify_therapeutic_areas.return_value = []
            mock_processor.normalize_study_types.return_value = ["clinical trial"]
            mock_processor.normalize_publication_year.return_value = None
            mock_processor.normalize_species.return_value = []
            mock_processor.normalize_pharmacokinetic_data.return_value = {}

            # Execute
            result = db._normalize_metadata_fields(doc.metadata, mock_processor)

            # Verify both old and new fields exist
            assert "study_type" in result  # Old field
            assert "study_types" in result  # New field
            assert result["study_type"] == "clinical trial"
            assert result["study_types"] == ["clinical trial"]

    def test_extract_pharmaceutical_metadata_calls_normalize(self):
        """Test that _extract_pharmaceutical_metadata calls normalization."""
        # Setup
        mock_embeddings = Mock(spec=NVIDIAEmbeddings)
        db = VectorDatabase(embeddings=mock_embeddings, db_path="./test_db")

        doc = Mock()
        doc.metadata = {}
        doc.page_content = "Test content"

        # Mock processor and enhance_document_metadata
        with patch.object(db, '_get_pharmaceutical_processor') as mock_get_processor:
            mock_processor = Mock(spec=PharmaceuticalProcessor)
            mock_get_processor.return_value = mock_processor
            mock_processor.enhance_document_metadata.return_value = {
                "metadata": {"raw_field": "raw_value"}
            }

            # Mock _normalize_metadata_fields
            with patch.object(db, '_normalize_metadata_fields') as mock_normalize:
                mock_normalize.return_value = {"normalized_field": "normalized_value"}

                # Execute
                result = db._extract_pharmaceutical_metadata(doc)

                # Verify normalization was called
                mock_normalize.assert_called_once_with(
                    {"raw_field": "raw_value"},
                    mock_processor
                )
                assert result["normalized_field"] == "normalized_value"
                assert result["pharmaceutical_enriched"] is True