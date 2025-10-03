"""
NVIDIA RAG Agent Package
"""

__version__ = "1.0.0"
__author__ = "RAG Agent Developer"
__description__ = "NVIDIA LLaMA 3.2 NemoRetriever RAG Agent"

from .ddi_pk_processor import DDIPKProcessor, DrugInteraction, PKParameter
from .document_loader import PDFDocumentLoader
from .enhanced_rag_agent import EnhancedRAGAgent
from .medical_guardrails import MedicalGuardrails
from .nvidia_embeddings import NVIDIAEmbeddings
from .pharmaceutical_processor import PharmaceuticalProcessor
from .pharmaceutical_query_adapter import build_pharmaceutical_query_engine
from .query_engine import EnhancedQueryEngine
from .rag_agent import RAGAgent, RAGResponse
from .ranking_filter import StudyRankingFilter
from .synthesis_engine import KeyFinding, SynthesisEngine
from .vector_database import VectorDatabase

# Paper schema utilities with import safety
try:
    from .paper_schema import (
        DOI_PATTERN,
        PMID_PATTERN,
        PMID_PATTERN_EXTRACT,
        Paper,
        clean_identifier,
        coerce_paper,
        coerce_papers,
        extract_doi,
        extract_pmid,
        normalize_doi,
        normalize_identifier,
        normalize_pmid,
        validate_doi,
        validate_pmid,
    )
except ImportError:  # pragma: no cover - support direct module execution
    from paper_schema import (  # type: ignore
        DOI_PATTERN,
        PMID_PATTERN,
        PMID_PATTERN_EXTRACT,
        Paper,
        clean_identifier,
        coerce_paper,
        coerce_papers,
        extract_doi,
        extract_pmid,
        normalize_doi,
        normalize_identifier,
        normalize_pmid,
        validate_doi,
        validate_pmid,
    )

__all__ = [
    "NVIDIAEmbeddings",
    "PDFDocumentLoader",
    "VectorDatabase",
    "RAGAgent",
    "RAGResponse",
    "EnhancedRAGAgent",
    "EnhancedQueryEngine",
    "StudyRankingFilter",
    "PharmaceuticalProcessor",
    "build_pharmaceutical_query_engine",
    "SynthesisEngine",
    "KeyFinding",
    "DDIPKProcessor",
    "PKParameter",
    "DrugInteraction",
    "MedicalGuardrails",
    # Paper schema utilities
    "Paper",
    "coerce_paper",
    "coerce_papers",
    "DOI_PATTERN",
    "PMID_PATTERN",
    "PMID_PATTERN_EXTRACT",
    "clean_identifier",
    "normalize_doi",
    "normalize_pmid",
    "extract_doi",
    "extract_pmid",
    "normalize_identifier",
    "validate_doi",
    "validate_pmid",
]
