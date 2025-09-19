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
from .query_engine import EnhancedQueryEngine
from .ranking_filter import StudyRankingFilter
from .pharmaceutical_processor import PharmaceuticalProcessor
from .pharmaceutical_query_adapter import build_pharmaceutical_query_engine
from .synthesis_engine import SynthesisEngine, KeyFinding
from .ddi_pk_processor import DDIPKProcessor, PKParameter, DrugInteraction
from .medical_guardrails import MedicalGuardrails

__all__ = [
    "NVIDIAEmbeddings",
    "PDFDocumentLoader", 
    "VectorDatabase",
    "RAGAgent",
    "RAGResponse",
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
]
