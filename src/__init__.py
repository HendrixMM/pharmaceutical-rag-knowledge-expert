"""
NVIDIA RAG Agent Package
"""

__version__ = "1.0.0"
__author__ = "RAG Agent Developer"
__description__ = "NVIDIA LLaMA 3.2 NemoRetriever RAG Agent"

from .nvidia_embeddings import NVIDIAEmbeddings
from .document_loader import PDFDocumentLoader
from .vector_database import VectorDatabase
from .rag_agent import RAGAgent, RAGResponse
from .enhanced_rag_agent import EnhancedRAGAgent
from .query_engine import EnhancedQueryEngine
from .ranking_filter import StudyRankingFilter
from .pharmaceutical_processor import PharmaceuticalProcessor
from .pharmaceutical_query_adapter import build_pharmaceutical_query_engine
from .synthesis_engine import SynthesisEngine, KeyFinding
from .ddi_pk_processor import DDIPKProcessor, PKParameter, DrugInteraction
from .medical_guardrails import MedicalGuardrails

# Paper schema utilities with import safety
try:
    from .paper_schema import Paper, coerce_paper, coerce_papers
    from .paper_schema import DOI_PATTERN, PMID_PATTERN, PMID_PATTERN_EXTRACT
    from .paper_schema import clean_identifier, normalize_doi, normalize_pmid
    from .paper_schema import extract_doi, extract_pmid, normalize_identifier
    from .paper_schema import validate_doi, validate_pmid
except ImportError:  # pragma: no cover - support direct module execution
    from paper_schema import Paper, coerce_paper, coerce_papers  # type: ignore
    from paper_schema import DOI_PATTERN, PMID_PATTERN, PMID_PATTERN_EXTRACT  # type: ignore
    from paper_schema import clean_identifier, normalize_doi, normalize_pmid  # type: ignore
    from paper_schema import extract_doi, extract_pmid, normalize_identifier  # type: ignore
    from paper_schema import validate_doi, validate_pmid  # type: ignore

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
