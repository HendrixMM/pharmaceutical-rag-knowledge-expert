"""
NVIDIA NeMo Extraction Service

Advanced document processing using NVIDIA NeMo Retriever Extraction (NV-Ingest).
Replaces PyPDF2 with VLM-based OCR and structured data extraction optimized
for pharmaceutical and medical documents.

Key Features:
- VLM-based OCR with semantic understanding
- Table and chart extraction with structure preservation
- Chemical formula and molecular structure recognition
- Multi-page document processing with context preservation
- Pharmaceutical regulatory document parsing
- Metadata enrichment for medical literature

<<use_mcp microsoft-learn>>

Based on latest NVIDIA NeMo Retriever documentation patterns.
"""

import asyncio
import base64
import io
import logging
import mimetypes
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import aiofiles
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document

from .nemo_retriever_client import NeMoRetrieverClient, NeMoAPIResponse

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Result from document extraction processing."""
    success: bool
    documents: List[Document] = None
    metadata: Dict[str, Any] = None
    tables: List[Dict[str, Any]] = None
    charts: List[Dict[str, Any]] = None
    chemical_structures: List[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    extraction_method: str = "unknown"

@dataclass
class PharmaceuticalMetadata:
    """Pharmaceutical-specific metadata extracted from documents."""
    drug_names: List[str] = None
    dosages: List[str] = None
    indications: List[str] = None
    contraindications: List[str] = None
    side_effects: List[str] = None
    clinical_trial_ids: List[str] = None
    regulatory_agencies: List[str] = None
    approval_dates: List[str] = None
    chemical_formulas: List[str] = None
    molecular_weights: List[str] = None

class NeMoExtractionService:
    """
    Document extraction service using NVIDIA NeMo Retriever capabilities.

    Provides advanced document processing specifically optimized for:
    - Pharmaceutical research papers
    - Regulatory documents (FDA, EMA submissions)
    - Clinical trial reports
    - Patent documents
    - Chemical literature
    - Medical device documentation
    """

    # Supported file types for NeMo extraction
    SUPPORTED_FORMATS = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".txt": "text/plain",
        ".html": "text/html",
        ".rtf": "application/rtf"
    }

    # Pharmaceutical document patterns for enhanced extraction
    PHARMA_PATTERNS = {
        "drug_names": [
            r"\b[A-Z][a-z]+(?:mab|nib|zumab|cept|tinib|prazole|statin|cillin|mycin|oxin)\b",
            r"\b(?:compound|drug|medication|pharmaceutical)\s+[A-Z0-9-]+\b"
        ],
        "dosages": [
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|μg|ml|L|units?|IU|mmol|mEq)(?:/\w+)?\b",
            r"\b\d+(?:\.\d+)?\s*(?:milligrams?|grams?|micrograms?|milliliters?|liters?)\b"
        ],
        "clinical_trials": [
            r"\b(?:NCT|ISRCTN|EudraCT)\d+\b",
            r"\b(?:Phase|Study)\s+(?:I{1,3}|[1-4]|[IV]{1,4})\b"
        ],
        "regulatory": [
            r"\b(?:FDA|EMA|PMDA|TGA|Health Canada|MHRA)\b",
            r"\b(?:NDA|BLA|ANDA|510\(k\)|PMA|IDE)\s*\#?\s*\d+\b"
        ],
        "chemical_formulas": [
            r"\b[A-Z][a-z]?(?:\d+[A-Z][a-z]?\d*)*\b",  # Basic chemical formula
            r"\b(?:C|H|O|N|S|P|Cl|Br|F|I)\d*(?:[A-Z][a-z]?\d*)*\b"  # More specific
        ]
    }

    def __init__(self, nemo_client: Optional[NeMoRetrieverClient] = None):
        """
        Initialize NeMo Extraction Service.

        Args:
            nemo_client: Initialized NeMo client (creates new if None)
        """
        self.nemo_client = nemo_client
        self.extraction_stats = {
            "total_documents": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_processing_time_ms": 0.0
        }

    async def _ensure_nemo_client(self) -> NeMoRetrieverClient:
        """Ensure NeMo client is available."""
        if not self.nemo_client:
            from .nemo_retriever_client import create_nemo_client
            self.nemo_client = await create_nemo_client()
        return self.nemo_client

    async def extract_document(self,
                             file_path: Union[str, Path],
                             extraction_strategy: str = "auto",
                             enable_pharmaceutical_analysis: bool = True,
                             chunk_strategy: str = "semantic",
                             preserve_tables: bool = True,
                             extract_images: bool = True) -> ExtractionResult:
        """
        Extract content from a document using NeMo Retriever capabilities.

        Args:
            file_path: Path to document file
            extraction_strategy: Strategy to use ("nemo", "unstructured", "auto")
            enable_pharmaceutical_analysis: Extract pharmaceutical-specific metadata
            chunk_strategy: How to chunk the document ("semantic", "title", "page")
            preserve_tables: Whether to preserve table structures
            extract_images: Whether to process images and charts

        Returns:
            ExtractionResult with processed document content
        """
        start_time = time.time()
        file_path = Path(file_path)

        try:
            # Validate file
            if not file_path.exists():
                return ExtractionResult(
                    success=False,
                    error=f"File not found: {file_path}",
                    extraction_method="validation_failed"
                )

            file_ext = file_path.suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                return ExtractionResult(
                    success=False,
                    error=f"Unsupported file format: {file_ext}",
                    extraction_method="unsupported_format"
                )

            # Choose extraction method based on strategy
            if extraction_strategy == "auto":
                extraction_strategy = self._choose_extraction_strategy(file_path)

            # Perform extraction
            if extraction_strategy == "nemo":
                result = await self._extract_with_nemo(file_path, preserve_tables, extract_images)
            else:
                result = await self._extract_with_unstructured(file_path, preserve_tables)

            # Post-process with pharmaceutical analysis
            if result.success and enable_pharmaceutical_analysis:
                result = await self._enhance_with_pharmaceutical_analysis(result)

            # Apply chunking strategy
            if result.success and result.documents:
                result.documents = await self._apply_chunking_strategy(
                    result.documents, chunk_strategy
                )

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time

            self.extraction_stats["total_documents"] += 1
            self.extraction_stats["total_processing_time_ms"] += processing_time

            if result.success:
                self.extraction_stats["successful_extractions"] += 1
            else:
                self.extraction_stats["failed_extractions"] += 1

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Document extraction failed for {file_path}: {e}")

            self.extraction_stats["total_documents"] += 1
            self.extraction_stats["failed_extractions"] += 1
            self.extraction_stats["total_processing_time_ms"] += processing_time

            return ExtractionResult(
                success=False,
                error=str(e),
                processing_time_ms=processing_time,
                extraction_method="error"
            )

    def _choose_extraction_strategy(self, file_path: Path) -> str:
        """Choose optimal extraction strategy based on file characteristics."""
        file_ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size

        # Use NeMo for complex documents that benefit from VLM processing
        if file_ext == ".pdf" and file_size > 1024 * 1024:  # > 1MB PDFs
            return "nemo"
        elif file_ext in [".docx", ".pptx"]:  # Office documents with potential images/tables
            return "nemo"
        else:
            return "unstructured"  # Faster for simple documents

    async def _extract_with_nemo(self,
                                file_path: Path,
                                preserve_tables: bool,
                                extract_images: bool) -> ExtractionResult:
        """Extract document using NeMo Retriever Extraction NIM."""
        try:
            client = await self._ensure_nemo_client()

            # Read and encode file
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()

            file_b64 = base64.b64encode(file_content).decode('utf-8')
            mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

            # Prepare NeMo extraction payload
            extraction_payload = {
                "file": {
                    "content": file_b64,
                    "mime_type": mime_type,
                    "filename": file_path.name
                },
                "options": {
                    "preserve_tables": preserve_tables,
                    "extract_images": extract_images,
                    "ocr_method": "vlm",  # Use VLM-based OCR
                    "structure_analysis": True,
                    "semantic_chunking": True,
                    "language": "en"  # Can be enhanced to detect language
                }
            }

            # Call NeMo extraction service
            # Note: This is a placeholder for the actual NeMo API call
            # The exact API structure may vary based on final NeMo implementation
            nemo_response = await self._call_nemo_extraction_api(extraction_payload)

            if nemo_response.success:
                return await self._process_nemo_extraction_response(nemo_response.data, file_path)
            else:
                # Fallback to unstructured if NeMo fails
                logger.warning(f"NeMo extraction failed for {file_path}, falling back to unstructured")
                return await self._extract_with_unstructured(file_path, preserve_tables)

        except Exception as e:
            logger.error(f"NeMo extraction error for {file_path}: {e}")
            # Fallback to unstructured
            return await self._extract_with_unstructured(file_path, preserve_tables)

    async def _call_nemo_extraction_api(self, payload: Dict[str, Any]) -> NeMoAPIResponse:
        """
        Call NeMo Extraction API.

        Note: This is a placeholder implementation. The actual API structure
        will depend on the final NeMo Extraction NIM specification.
        """
        try:
            client = await self._ensure_nemo_client()
            config = client.services["extraction"]

            import aiohttp
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)
            ) as session:
                async with session.post(
                    config.endpoint,
                    headers=config.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return NeMoAPIResponse(
                            success=True,
                            data=result,
                            service="extraction"
                        )
                    else:
                        error_text = await response.text()
                        return NeMoAPIResponse(
                            success=False,
                            error=f"API error {response.status}: {error_text}",
                            service="extraction"
                        )

        except Exception as e:
            return NeMoAPIResponse(
                success=False,
                error=str(e),
                service="extraction"
            )

    async def _process_nemo_extraction_response(self,
                                              nemo_data: Dict[str, Any],
                                              file_path: Path) -> ExtractionResult:
        """Process response from NeMo Extraction API."""
        try:
            documents = []
            tables = []
            charts = []

            # Process extracted content based on NeMo response structure
            # This structure is based on expected NeMo Extraction API response
            if "content" in nemo_data:
                content_blocks = nemo_data["content"]

                for block in content_blocks:
                    if block.get("type") == "text":
                        doc = Document(
                            page_content=block["text"],
                            metadata={
                                "source": str(file_path),
                                "page": block.get("page", 1),
                                "block_type": "text",
                                "confidence": block.get("confidence", 1.0),
                                "extraction_method": "nemo_vlm"
                            }
                        )
                        documents.append(doc)

                    elif block.get("type") == "table":
                        # Preserve table structure
                        table_data = {
                            "content": block.get("table_data", {}),
                            "markdown": block.get("markdown", ""),
                            "page": block.get("page", 1),
                            "confidence": block.get("confidence", 1.0)
                        }
                        tables.append(table_data)

                        # Also create a document for the table content
                        table_text = block.get("markdown", str(block.get("table_data", "")))
                        doc = Document(
                            page_content=table_text,
                            metadata={
                                "source": str(file_path),
                                "page": block.get("page", 1),
                                "block_type": "table",
                                "confidence": block.get("confidence", 1.0),
                                "extraction_method": "nemo_vlm"
                            }
                        )
                        documents.append(doc)

                    elif block.get("type") == "image" or block.get("type") == "chart":
                        chart_data = {
                            "description": block.get("description", ""),
                            "ocr_text": block.get("ocr_text", ""),
                            "page": block.get("page", 1),
                            "confidence": block.get("confidence", 1.0),
                            "image_type": block.get("image_type", "unknown")
                        }
                        charts.append(chart_data)

                        # Create document for image/chart text
                        if chart_data["description"] or chart_data["ocr_text"]:
                            content = f"{chart_data['description']} {chart_data['ocr_text']}".strip()
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": str(file_path),
                                    "page": block.get("page", 1),
                                    "block_type": "image",
                                    "confidence": block.get("confidence", 1.0),
                                    "extraction_method": "nemo_vlm"
                                }
                            )
                            documents.append(doc)

            # Extract metadata
            metadata = {
                "source": str(file_path),
                "extraction_method": "nemo_vlm",
                "total_pages": nemo_data.get("metadata", {}).get("pages", 1),
                "language": nemo_data.get("metadata", {}).get("language", "unknown"),
                "confidence_avg": nemo_data.get("metadata", {}).get("avg_confidence", 1.0)
            }

            return ExtractionResult(
                success=True,
                documents=documents,
                metadata=metadata,
                tables=tables,
                charts=charts,
                extraction_method="nemo_vlm"
            )

        except Exception as e:
            logger.error(f"Failed to process NeMo extraction response: {e}")
            return ExtractionResult(
                success=False,
                error=str(e),
                extraction_method="nemo_processing_failed"
            )

    async def _extract_with_unstructured(self,
                                       file_path: Path,
                                       preserve_tables: bool) -> ExtractionResult:
        """Extract document using unstructured library as fallback."""
        try:
            # Use unstructured for document parsing
            elements = partition(
                filename=str(file_path),
                strategy="hi_res" if preserve_tables else "fast",
                include_page_breaks=True
            )

            documents = []
            tables = []

            current_page = 1
            for element in elements:
                # Determine page number
                if hasattr(element, 'metadata') and element.metadata.page_number:
                    current_page = element.metadata.page_number

                # Create document
                doc = Document(
                    page_content=str(element),
                    metadata={
                        "source": str(file_path),
                        "page": current_page,
                        "element_type": element.category if hasattr(element, 'category') else "unknown",
                        "extraction_method": "unstructured"
                    }
                )
                documents.append(doc)

                # Handle tables specially
                if hasattr(element, 'category') and element.category == "Table":
                    table_data = {
                        "content": str(element),
                        "page": current_page,
                        "confidence": 1.0  # Unstructured doesn't provide confidence
                    }
                    tables.append(table_data)

            metadata = {
                "source": str(file_path),
                "extraction_method": "unstructured",
                "total_pages": current_page,
                "total_elements": len(elements)
            }

            return ExtractionResult(
                success=True,
                documents=documents,
                metadata=metadata,
                tables=tables,
                extraction_method="unstructured"
            )

        except Exception as e:
            logger.error(f"Unstructured extraction failed for {file_path}: {e}")
            return ExtractionResult(
                success=False,
                error=str(e),
                extraction_method="unstructured_failed"
            )

    async def _enhance_with_pharmaceutical_analysis(self, result: ExtractionResult) -> ExtractionResult:
        """Enhance extraction result with pharmaceutical-specific analysis."""
        if not result.documents:
            return result

        try:
            # Combine all text for analysis
            all_text = " ".join([doc.page_content for doc in result.documents])

            # Extract pharmaceutical metadata
            pharma_metadata = self._extract_pharmaceutical_metadata(all_text)

            # Add pharmaceutical metadata to result
            if not result.metadata:
                result.metadata = {}

            result.metadata["pharmaceutical"] = pharma_metadata.__dict__

            # Enhance individual documents with relevant pharmaceutical tags
            for doc in result.documents:
                doc.metadata["pharmaceutical_tags"] = self._tag_pharmaceutical_content(doc.page_content)

            return result

        except Exception as e:
            logger.warning(f"Pharmaceutical analysis failed: {e}")
            return result

    def _extract_pharmaceutical_metadata(self, text: str) -> PharmaceuticalMetadata:
        """Extract pharmaceutical-specific metadata from text."""
        import re

        metadata = PharmaceuticalMetadata()

        # Extract drug names
        drug_names = set()
        for pattern in self.PHARMA_PATTERNS["drug_names"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            drug_names.update(matches)
        metadata.drug_names = list(drug_names)

        # Extract dosages
        dosages = set()
        for pattern in self.PHARMA_PATTERNS["dosages"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dosages.update(matches)
        metadata.dosages = list(dosages)

        # Extract clinical trial IDs
        trial_ids = set()
        for pattern in self.PHARMA_PATTERNS["clinical_trials"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            trial_ids.update(matches)
        metadata.clinical_trial_ids = list(trial_ids)

        # Extract regulatory information
        regulatory = set()
        for pattern in self.PHARMA_PATTERNS["regulatory"]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            regulatory.update(matches)
        metadata.regulatory_agencies = list(regulatory)

        # Extract chemical formulas
        formulas = set()
        for pattern in self.PHARMA_PATTERNS["chemical_formulas"]:
            matches = re.findall(pattern, text)
            # Filter out common non-chemical matches
            formulas.update([m for m in matches if len(m) > 2 and not m.isdigit()])
        metadata.chemical_formulas = list(formulas)

        return metadata

    def _tag_pharmaceutical_content(self, text: str) -> List[str]:
        """Tag content with pharmaceutical relevance indicators."""
        tags = []

        text_lower = text.lower()

        # Content type tags
        if any(word in text_lower for word in ["clinical trial", "study", "phase"]):
            tags.append("clinical_trial")

        if any(word in text_lower for word in ["side effect", "adverse", "contraindication"]):
            tags.append("safety_data")

        if any(word in text_lower for word in ["dosage", "dose", "administration"]):
            tags.append("dosing_information")

        if any(word in text_lower for word in ["mechanism", "action", "pharmacology"]):
            tags.append("mechanism_of_action")

        if any(word in text_lower for word in ["chemical", "formula", "structure", "molecular"]):
            tags.append("chemical_information")

        if any(word in text_lower for word in ["fda", "approval", "regulatory", "submission"]):
            tags.append("regulatory_information")

        return tags

    async def _apply_chunking_strategy(self,
                                     documents: List[Document],
                                     strategy: str) -> List[Document]:
        """Apply chunking strategy to documents."""
        if strategy == "semantic":
            return await self._semantic_chunking(documents)
        elif strategy == "title":
            return await self._title_based_chunking(documents)
        elif strategy == "page":
            return documents  # Already chunked by page
        else:
            return documents

    async def _semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """Apply semantic chunking using content understanding."""
        # Placeholder for semantic chunking implementation
        # In production, this could use NeMo embedding models to identify
        # semantically coherent chunks

        chunked_docs = []
        for doc in documents:
            # Simple implementation: split long documents into smaller chunks
            if len(doc.page_content) > 2000:
                # Split by sentences while preserving metadata
                sentences = doc.page_content.split('. ')
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + sentence) < 1500:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunk_doc = Document(
                                page_content=current_chunk.strip(),
                                metadata={**doc.metadata, "chunk_type": "semantic"}
                            )
                            chunked_docs.append(chunk_doc)
                        current_chunk = sentence + ". "

                # Add final chunk
                if current_chunk:
                    chunk_doc = Document(
                        page_content=current_chunk.strip(),
                        metadata={**doc.metadata, "chunk_type": "semantic"}
                    )
                    chunked_docs.append(chunk_doc)
            else:
                chunked_docs.append(doc)

        return chunked_docs

    async def _title_based_chunking(self, documents: List[Document]) -> List[Document]:
        """Apply title-based chunking using unstructured."""
        try:
            # Convert documents back to unstructured elements for chunking
            # This is a simplified approach
            chunked_docs = []

            for doc in documents:
                # Use unstructured chunking if available
                chunks = chunk_by_title([doc.page_content])

                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=str(chunk),
                        metadata={
                            **doc.metadata,
                            "chunk_type": "title_based",
                            "chunk_index": i
                        }
                    )
                    chunked_docs.append(chunk_doc)

            return chunked_docs

        except Exception as e:
            logger.warning(f"Title-based chunking failed: {e}")
            return documents

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        stats = self.extraction_stats.copy()

        if stats["total_documents"] > 0:
            stats["success_rate"] = stats["successful_extractions"] / stats["total_documents"]
            stats["avg_processing_time_ms"] = (
                stats["total_processing_time_ms"] / stats["total_documents"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["avg_processing_time_ms"] = 0.0

        return stats

    async def batch_extract_documents(self,
                                    file_paths: List[Union[str, Path]],
                                    max_concurrent: int = 3,
                                    **extraction_kwargs) -> List[ExtractionResult]:
        """
        Extract multiple documents concurrently.

        Args:
            file_paths: List of document paths to extract
            max_concurrent: Maximum number of concurrent extractions
            **extraction_kwargs: Arguments passed to extract_document

        Returns:
            List of ExtractionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_single(file_path):
            async with semaphore:
                return await self.extract_document(file_path, **extraction_kwargs)

        tasks = [extract_single(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExtractionResult(
                    success=False,
                    error=str(result),
                    extraction_method="batch_error"
                ))
            else:
                processed_results.append(result)

        return processed_results


# Convenience functions
async def extract_pharmaceutical_document(file_path: Union[str, Path],
                                        nemo_client: Optional[NeMoRetrieverClient] = None,
                                        **kwargs) -> ExtractionResult:
    """
    Quick function to extract a pharmaceutical document with optimized settings.

    Args:
        file_path: Path to document
        nemo_client: Optional NeMo client
        **kwargs: Additional extraction parameters

    Returns:
        ExtractionResult with pharmaceutical analysis
    """
    service = NeMoExtractionService(nemo_client)

    # Set pharmaceutical-optimized defaults
    extraction_kwargs = {
        "extraction_strategy": "auto",
        "enable_pharmaceutical_analysis": True,
        "chunk_strategy": "semantic",
        "preserve_tables": True,
        "extract_images": True,
        **kwargs
    }

    return await service.extract_document(file_path, **extraction_kwargs)


# Example usage
if __name__ == "__main__":
    import time

    async def test_extraction():
        """Test document extraction functionality."""
        service = NeMoExtractionService()

        # Test with a sample PDF (create if needed for testing)
        test_file = "test_document.pdf"

        if Path(test_file).exists():
            result = await service.extract_document(
                test_file,
                enable_pharmaceutical_analysis=True
            )

            if result.success:
                print(f"Extraction successful: {len(result.documents)} documents")
                print(f"Processing time: {result.processing_time_ms:.2f}ms")
                print(f"Tables found: {len(result.tables or [])}")
                print(f"Charts found: {len(result.charts or [])}")

                if result.metadata and "pharmaceutical" in result.metadata:
                    pharma = result.metadata["pharmaceutical"]
                    print(f"Drug names found: {pharma.get('drug_names', [])}")
                    print(f"Clinical trials found: {pharma.get('clinical_trial_ids', [])}")
            else:
                print(f"Extraction failed: {result.error}")
        else:
            print(f"Test file {test_file} not found")

    # Run test
    asyncio.run(test_extraction())