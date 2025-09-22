import os
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _require(module_name: str):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) else None


@pytest.mark.integration
def test_real_nemo_extraction_end_to_end(tmp_path):
    """
    Real API test for NeMo extraction. Skips unless required env is present.

    Required env vars:
    - NVIDIA_API_KEY: valid API key
    - TEST_REAL_PDF_PATH: path to a real PDF for extraction

    Optional env vars:
    - NEMO_EXTRACTION_STRATEGY (default: nemo)
    """
    if not _require("aiohttp") or not _require("aiofiles"):
        pytest.skip("aiohttp/aiofiles not installed; install deps to run integration test")

    api_key = _env("NVIDIA_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not set; skipping real API test")

    pdf_path = _env("TEST_REAL_PDF_PATH")
    if not pdf_path or not Path(pdf_path).exists():
        pytest.skip("TEST_REAL_PDF_PATH not set or file missing; skipping real API test")

    # Prepare docs directory with the provided PDF
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    src_pdf = Path(pdf_path)
    dst_pdf = docs_dir / src_pdf.name
    shutil.copyfile(src_pdf, dst_pdf)

    # Import loader via package path
# Provide minimal stubs for langchain modules if not installed.
if not _require("langchain_core") or not _require("langchain_community") or not _require("langchain"):
    import sys, types
    # langchain_core
    if not _require("langchain_core"):
        lc_core = types.ModuleType("langchain_core"); lc_core.__path__ = []
        sys.modules["langchain_core"] = lc_core
        lc_core_documents = types.ModuleType("langchain_core.documents")
        class Document:
            def __init__(self, page_content: str, metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = dict(metadata or {})
        lc_core_documents.Document = Document
        sys.modules["langchain_core.documents"] = lc_core_documents
        lc_core_documents_base = types.ModuleType("langchain_core.documents.base")
        class Blob:
            @classmethod
            def from_data(cls, data: bytes, path: str | None = None):
                return cls()
        lc_core_documents_base.Blob = Blob
        sys.modules["langchain_core.documents.base"] = lc_core_documents_base
    # langchain
    if not _require("langchain"):
        lc_pkg = types.ModuleType("langchain"); lc_pkg.__path__ = []
        sys.modules["langchain"] = lc_pkg
        ts = types.ModuleType("langchain.text_splitter")
        class RecursiveCharacterTextSplitter:
            def __init__(self, *args, **kwargs): pass
            def split_documents(self, documents): return list(documents)
        ts.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        sys.modules["langchain.text_splitter"] = ts
    # langchain_community
    if not _require("langchain_community"):
        comm = types.ModuleType("langchain_community"); comm.__path__ = []
        sys.modules["langchain_community"] = comm
        dls = types.ModuleType("langchain_community.document_loaders")
        class PyPDFLoader:
            def __init__(self, path: str): self.path = path
            def load(self):
                # Only used on fallback; returning minimal structure is fine
                Document = sys.modules["langchain_core.documents"].Document
                return [Document("page", {"page": 1})]
        dls.PyPDFLoader = PyPDFLoader
        sys.modules["langchain_community.document_loaders"] = dls

from src.document_loader import PDFDocumentLoader

    # Force NeMo strategy by constructor (overrides env)
    loader = PDFDocumentLoader(
        str(docs_dir),
        enable_nemo_extraction=True,
        nemo_extraction_config={
            "extraction_strategy": os.getenv("NEMO_EXTRACTION_STRATEGY", "nemo"),
            "chunk_strategy": os.getenv("NEMO_CHUNK_STRATEGY", "page"),
            "preserve_tables": True,
            "extract_images": True,
        },
    )

    docs = loader.load_documents()

    # Basic assertions: documents extracted and have extraction metadata
    assert isinstance(docs, list)
    assert len(docs) > 0, "No documents extracted; verify API key and network access"
    assert any("extraction_method" in (d.metadata or {}) for d in docs)

    # Provide context in logs for manual review
    metrics = loader.get_nemo_metrics()
    print("NeMo extraction metrics:", metrics)
    print("First doc metadata:", docs[0].metadata)
