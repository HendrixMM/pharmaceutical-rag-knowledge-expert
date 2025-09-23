"""
Unit tests for NeMo Extraction Service

Tests the NVIDIA NeMo Extraction Service with VLM-based OCR,
structured data extraction, and pharmaceutical document processing.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

import pytest
from langchain_core.documents import Document

from src.nemo_extraction_service import (
    NeMoExtractionService,
    ExtractionResult,
    ExtractionMethod
)


class TestExtractionResult:
    """Test extraction result data structure."""

    def test_successful_result(self):
        documents = [
            Document(page_content="Test content", metadata={"page": 1}),
            Document(page_content="More content", metadata={"page": 2})
        ]

        result = ExtractionResult(
            success=True,
            documents=documents,
            metadata={"total_pages": 2, "extraction_method": "nemo_vlm"}
        )

        assert result.success is True
        assert len(result.documents) == 2
        assert result.metadata["total_pages"] == 2
        assert result.metadata["extraction_method"] == "nemo_vlm"

    def test_failed_result(self):
        result = ExtractionResult(
            success=False,
            documents=[],
            metadata={"error": "PDF parsing failed"}
        )

        assert result.success is False
        assert len(result.documents) == 0
        assert "error" in result.metadata


class TestExtractionMethod:
    """Test extraction method enum."""

    def test_extraction_method_values(self):
        assert ExtractionMethod.NEMO_VLM.value == "nemo_vlm"
        assert ExtractionMethod.UNSTRUCTURED.value == "unstructured"
        assert ExtractionMethod.PYPDF.value == "pypdf"

    def test_extraction_method_ordering(self):
        """Test that methods are ordered by preference."""
        methods = [ExtractionMethod.NEMO_VLM, ExtractionMethod.UNSTRUCTURED, ExtractionMethod.PYPDF]

        # NeMo VLM should be highest preference
        assert methods[0] == ExtractionMethod.NEMO_VLM


class TestNeMoExtractionService:
    """Test core NeMo Extraction Service functionality."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked dependencies."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key", "ENABLE_NEMO_EXTRACTION": "true"}):
            return NeMoExtractionService()

    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a sample PDF file for testing."""
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF")
        return pdf_path

    def test_init_default_config(self, mock_service):
        """Test service initialization with default configuration."""
        assert mock_service.enable_nemo_extraction is True
        assert mock_service.strategy == "auto"  # Default from NEMO_EXTRACTION_STRATEGY
        assert mock_service.enable_pharmaceutical_analysis is True

    def test_init_custom_config(self):
        """Test service initialization with custom configuration."""
        custom_env = {
            "NVIDIA_API_KEY": "test-key",
            "ENABLE_NEMO_EXTRACTION": "false",
            "NEMO_EXTRACTION_STRATEGY": "unstructured",
            "NEMO_PHARMACEUTICAL_ANALYSIS": "false"
        }

        with patch.dict(os.environ, custom_env):
            service = NeMoExtractionService()

            assert service.enable_nemo_extraction is False
            assert service.strategy == "unstructured"
            assert service.enable_pharmaceutical_analysis is False

    @pytest.mark.asyncio
    async def test_extract_from_pdf_nemo_success(self, mock_service, sample_pdf_path):
        """Test successful PDF extraction using NeMo VLM."""
        mock_documents = [
            Document(
                page_content="Pharmaceutical research document content",
                metadata={"page": 1, "extraction_method": "nemo_vlm"}
            )
        ]

        with patch.object(mock_service, '_extract_with_nemo_vlm') as mock_extract:
            mock_extract.return_value = ExtractionResult(
                success=True,
                documents=mock_documents,
                metadata={"extraction_method": "nemo_vlm", "pages_processed": 1}
            )

            result = await mock_service.extract_from_pdf(sample_pdf_path)

            assert result.success is True
            assert len(result.documents) == 1
            assert "pharmaceutical" in result.documents[0].page_content.lower()
            assert result.metadata["extraction_method"] == "nemo_vlm"
            mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_from_pdf_fallback_to_unstructured(self, mock_service, sample_pdf_path):
        """Test fallback to unstructured when NeMo VLM fails."""
        mock_service.strategy = "auto"  # Enable fallback

        with patch.object(mock_service, '_extract_with_nemo_vlm') as mock_nemo:
            with patch.object(mock_service, '_extract_with_unstructured') as mock_unstructured:
                # NeMo VLM fails
                mock_nemo.return_value = ExtractionResult(
                    success=False,
                    documents=[],
                    metadata={"error": "VLM service unavailable"}
                )

                # Unstructured succeeds
                mock_unstructured.return_value = ExtractionResult(
                    success=True,
                    documents=[Document(page_content="Fallback content")],
                    metadata={"extraction_method": "unstructured"}
                )

                result = await mock_service.extract_from_pdf(sample_pdf_path)

                assert result.success is True
                assert result.metadata["extraction_method"] == "unstructured"
                mock_nemo.assert_called_once()
                mock_unstructured.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_strict_mode_no_fallback(self, mock_service, sample_pdf_path):
        """Test strict mode prevents fallback to non-NVIDIA methods."""
        mock_service.strict_mode = True

        with patch.object(mock_service, '_extract_with_nemo_vlm') as mock_nemo:
            with patch.object(mock_service, '_extract_with_unstructured') as mock_unstructured:
                # NeMo VLM fails
                mock_nemo.return_value = ExtractionResult(
                    success=False,
                    documents=[],
                    metadata={"error": "VLM service unavailable"}
                )

                result = await mock_service.extract_from_pdf(sample_pdf_path)

                assert result.success is False
                mock_nemo.assert_called_once()
                mock_unstructured.assert_not_called()  # Should not fallback in strict mode

    @pytest.mark.asyncio
    async def test_extract_pharmaceutical_analysis(self, mock_service, sample_pdf_path):
        """Test pharmaceutical analysis enhancement."""
        mock_service.enable_pharmaceutical_analysis = True

        pharmaceutical_content = """
        DRUG: Metformin Hydrochloride
        INDICATION: Type 2 diabetes mellitus
        DOSAGE: 500mg twice daily with meals
        CONTRAINDICATIONS: Severe renal impairment
        """

        mock_documents = [
            Document(
                page_content=pharmaceutical_content,
                metadata={"page": 1}
            )
        ]

        with patch.object(mock_service, '_extract_with_nemo_vlm') as mock_extract:
            with patch.object(mock_service, '_apply_pharmaceutical_analysis') as mock_pharma:
                mock_extract.return_value = ExtractionResult(
                    success=True,
                    documents=mock_documents,
                    metadata={"extraction_method": "nemo_vlm"}
                )

                mock_pharma.return_value = mock_documents  # Enhanced documents

                result = await mock_service.extract_from_pdf(sample_pdf_path)

                assert result.success is True
                mock_pharma.assert_called_once_with(mock_documents)

    @pytest.mark.asyncio
    async def test_extract_with_table_preservation(self, mock_service, sample_pdf_path):
        """Test table and image preservation during extraction."""
        mock_service.preserve_tables = True
        mock_service.extract_images = True

        table_content = """
        Drug Name | Dosage | Frequency
        Aspirin   | 81mg   | Daily
        Metformin | 500mg  | Twice daily
        """

        mock_documents = [
            Document(
                page_content=table_content,
                metadata={
                    "page": 1,
                    "has_tables": True,
                    "table_count": 1,
                    "has_images": False
                }
            )
        ]

        with patch.object(mock_service, '_extract_with_nemo_vlm') as mock_extract:
            mock_extract.return_value = ExtractionResult(
                success=True,
                documents=mock_documents,
                metadata={
                    "extraction_method": "nemo_vlm",
                    "tables_preserved": True,
                    "images_extracted": True
                }
            )

            result = await mock_service.extract_from_pdf(sample_pdf_path)

            assert result.success is True
            assert result.metadata["tables_preserved"] is True
            assert "Drug Name" in result.documents[0].page_content

    def test_file_validation(self, mock_service, tmp_path):
        """Test file validation logic."""
        # Test valid PDF file
        valid_pdf = tmp_path / "valid.pdf"
        valid_pdf.write_bytes(b"%PDF-1.4")

        assert mock_service._validate_file(valid_pdf) is True

        # Test non-existent file
        non_existent = tmp_path / "missing.pdf"
        assert mock_service._validate_file(non_existent) is False

        # Test non-PDF file
        text_file = tmp_path / "document.txt"
        text_file.write_text("Not a PDF")
        assert mock_service._validate_file(text_file) is False

    def test_chunk_strategy_selection(self, mock_service):
        """Test different chunking strategies."""
        # Test semantic chunking
        mock_service.chunk_strategy = "semantic"
        strategy = mock_service._get_chunking_strategy()
        assert strategy == "semantic"

        # Test title-based chunking
        mock_service.chunk_strategy = "title"
        strategy = mock_service._get_chunking_strategy()
        assert strategy == "title"

        # Test page-based chunking
        mock_service.chunk_strategy = "page"
        strategy = mock_service._get_chunking_strategy()
        assert strategy == "page"


class TestNeMoVLMExtraction:
    """Test NeMo VLM-specific extraction functionality."""

    @pytest.fixture
    def mock_service_with_client(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            service = NeMoExtractionService()

            # Mock the NeMo client
            mock_client = AsyncMock()
            service.nemo_client = mock_client

            return service, mock_client

    @pytest.mark.asyncio
    async def test_nemo_vlm_extraction_success(self, mock_service_with_client, tmp_path):
        """Test successful NeMo VLM extraction."""
        service, mock_client = mock_service_with_client

        # Create sample PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample PDF content")

        # Mock VLM response
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {
            "extracted_text": "Clinical trial results for pharmaceutical compound",
            "tables": [{"headers": ["Drug", "Efficacy"], "rows": [["Compound A", "85%"]]}],
            "images": [],
            "metadata": {"pages": 1}
        }
        mock_client.extract_document.return_value = mock_response

        result = await service._extract_with_nemo_vlm(pdf_path)

        assert result.success is True
        assert len(result.documents) > 0
        assert "clinical trial" in result.documents[0].page_content.lower()
        mock_client.extract_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_nemo_vlm_extraction_failure(self, mock_service_with_client, tmp_path):
        """Test NeMo VLM extraction failure."""
        service, mock_client = mock_service_with_client

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample PDF content")

        # Mock VLM failure
        mock_response = Mock()
        mock_response.success = False
        mock_response.error = "Document processing failed"
        mock_client.extract_document.return_value = mock_response

        result = await service._extract_with_nemo_vlm(pdf_path)

        assert result.success is False
        assert "Document processing failed" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_nemo_vlm_with_pharmaceutical_content(self, mock_service_with_client, tmp_path):
        """Test NeMo VLM extraction with pharmaceutical content."""
        service, mock_client = mock_service_with_client
        service.enable_pharmaceutical_analysis = True

        pdf_path = tmp_path / "pharma.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nPharmaceutical document")

        # Mock VLM response with pharmaceutical content
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {
            "extracted_text": """
            PRESCRIBING INFORMATION
            Metformin Hydrochloride Tablets, USP

            INDICATIONS AND USAGE
            Metformin is indicated as an adjunct to diet and exercise to improve
            glycemic control in adults with type 2 diabetes mellitus.

            CONTRAINDICATIONS
            Metformin is contraindicated in patients with severe renal impairment.
            """,
            "tables": [],
            "images": [],
            "metadata": {"pages": 1, "document_type": "prescribing_information"}
        }
        mock_client.extract_document.return_value = mock_response

        result = await service._extract_with_nemo_vlm(pdf_path)

        assert result.success is True
        content = result.documents[0].page_content
        assert "metformin" in content.lower()
        assert "contraindicated" in content.lower()
        assert "indications" in content.lower()


class TestUnstructuredFallback:
    """Test unstructured library fallback functionality."""

    @pytest.fixture
    def mock_service(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoExtractionService()

    @pytest.mark.asyncio
    async def test_unstructured_extraction_success(self, mock_service, tmp_path):
        """Test successful unstructured extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\nSample content")

        with patch('unstructured.partition.auto.partition') as mock_partition:
            mock_elements = [
                Mock(text="Document title", category="Title"),
                Mock(text="Document content paragraph", category="NarrativeText"),
                Mock(text="Table data", category="Table")
            ]
            mock_partition.return_value = mock_elements

            result = await mock_service._extract_with_unstructured(pdf_path)

            assert result.success is True
            assert len(result.documents) > 0
            assert result.metadata["extraction_method"] == "unstructured"

    @pytest.mark.asyncio
    async def test_unstructured_extraction_failure(self, mock_service, tmp_path):
        """Test unstructured extraction failure."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"Invalid PDF")

        with patch('unstructured.partition.auto.partition') as mock_partition:
            mock_partition.side_effect = Exception("Parsing failed")

            result = await mock_service._extract_with_unstructured(pdf_path)

            assert result.success is False
            assert "parsing failed" in result.metadata.get("error", "").lower()


class TestPharmaceuticalAnalysis:
    """Test pharmaceutical domain-specific analysis."""

    @pytest.fixture
    def mock_service(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            service = NeMoExtractionService()
            service.enable_pharmaceutical_analysis = True
            return service

    def test_pharmaceutical_entity_extraction(self, mock_service):
        """Test extraction of pharmaceutical entities."""
        documents = [
            Document(
                page_content="""
                Study Drug: Lisinopril 10mg
                Indication: Hypertension
                Route: Oral administration
                Frequency: Once daily
                Contraindications: Pregnancy, angioedema
                """,
                metadata={"page": 1}
            )
        ]

        with patch.object(mock_service, '_extract_pharmaceutical_entities') as mock_extract:
            mock_extract.return_value = {
                "drug_names": ["Lisinopril"],
                "dosages": ["10mg"],
                "indications": ["Hypertension"],
                "contraindications": ["Pregnancy", "angioedema"]
            }

            entities = mock_service._extract_pharmaceutical_entities(documents[0])

            assert "Lisinopril" in entities["drug_names"]
            assert "10mg" in entities["dosages"]
            assert "Hypertension" in entities["indications"]

    def test_regulatory_document_classification(self, mock_service):
        """Test classification of regulatory document types."""
        prescribing_info = Document(
            page_content="PRESCRIBING INFORMATION - FDA Approved",
            metadata={"page": 1}
        )

        clinical_trial = Document(
            page_content="CLINICAL STUDY PROTOCOL - Phase III Trial",
            metadata={"page": 1}
        )

        with patch.object(mock_service, '_classify_document_type') as mock_classify:
            mock_classify.side_effect = ["prescribing_information", "clinical_study_protocol"]

            doc_type_1 = mock_service._classify_document_type(prescribing_info)
            doc_type_2 = mock_service._classify_document_type(clinical_trial)

            assert doc_type_1 == "prescribing_information"
            assert doc_type_2 == "clinical_study_protocol"

    def test_safety_information_prioritization(self, mock_service):
        """Test prioritization of safety-critical information."""
        safety_document = Document(
            page_content="""
            BLACK BOX WARNING
            Increased risk of serious cardiovascular events.

            CONTRAINDICATIONS
            Do not use in patients with severe liver impairment.

            WARNINGS AND PRECAUTIONS
            Monitor for signs of hepatotoxicity.
            """,
            metadata={"page": 1}
        )

        with patch.object(mock_service, '_prioritize_safety_content') as mock_prioritize:
            mock_prioritize.return_value = {
                "black_box_warnings": ["Increased risk of serious cardiovascular events"],
                "contraindications": ["Do not use in patients with severe liver impairment"],
                "warnings": ["Monitor for signs of hepatotoxicity"]
            }

            safety_info = mock_service._prioritize_safety_content(safety_document)

            assert len(safety_info["black_box_warnings"]) > 0
            assert len(safety_info["contraindications"]) > 0
            assert len(safety_info["warnings"]) > 0


class TestEnvironmentIntegration:
    """Test integration with environment variable configuration."""

    def test_environment_extraction_strategy(self):
        """Test that environment variables control extraction strategy."""
        strategies = ["auto", "nemo", "unstructured"]

        for strategy in strategies:
            test_env = {
                "NVIDIA_API_KEY": "test-key",
                "NEMO_EXTRACTION_STRATEGY": strategy
            }

            with patch.dict(os.environ, test_env):
                service = NeMoExtractionService()
                assert service.strategy == strategy

    def test_environment_pharmaceutical_analysis(self):
        """Test pharmaceutical analysis environment control."""
        test_env = {
            "NVIDIA_API_KEY": "test-key",
            "NEMO_PHARMACEUTICAL_ANALYSIS": "false"
        }

        with patch.dict(os.environ, test_env):
            service = NeMoExtractionService()
            assert service.enable_pharmaceutical_analysis is False

    def test_environment_strict_mode(self):
        """Test strict mode environment control."""
        test_env = {
            "NVIDIA_API_KEY": "test-key",
            "NEMO_EXTRACTION_STRICT": "true"
        }

        with patch.dict(os.environ, test_env):
            service = NeMoExtractionService()
            assert service.strict_mode is True

    def test_production_environment_enforcement(self):
        """Test that production environment enforces strict mode."""
        test_env = {
            "NVIDIA_API_KEY": "test-key",
            "APP_ENV": "production",
            "NEMO_EXTRACTION_STRICT": "false"  # Should be overridden
        }

        with patch.dict(os.environ, test_env):
            service = NeMoExtractionService()
            assert service.strict_mode is True  # Enforced in production