import os
import sys
import types
from pathlib import Path

import pytest


def _install_stub_langchain_modules():
    """Install lightweight stubs for langchain modules used by the loader.

    This avoids installing heavy dependencies while enabling import of the loader.
    """
    # langchain_core package and submodules
    lc_core = types.ModuleType("langchain_core")
    lc_core.__path__ = []  # mark as package
    sys.modules.setdefault("langchain_core", lc_core)

    lc_core_documents = types.ModuleType("langchain_core.documents")
    lc_core_documents.__package__ = "langchain_core"

    class Document:  # minimal stub
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = dict(metadata or {})

    lc_core_documents.Document = Document
    sys.modules.setdefault("langchain_core.documents", lc_core_documents)

    lc_core_documents_base = types.ModuleType("langchain_core.documents.base")

    class Blob:  # minimal stub
        @classmethod
        def from_data(cls, data: bytes, path: str | None = None):
            return cls()

    lc_core_documents_base.Blob = Blob
    sys.modules.setdefault("langchain_core.documents.base", lc_core_documents_base)

    # langchain package and text_splitter
    lc_pkg = types.ModuleType("langchain")
    lc_pkg.__path__ = []
    sys.modules.setdefault("langchain", lc_pkg)

    text_splitter = types.ModuleType("langchain.text_splitter")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_documents(self, documents):
            # Return documents unchanged for testing
            return list(documents)

    text_splitter.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules.setdefault("langchain.text_splitter", text_splitter)

    # langchain_community.document_loaders
    lc_community = types.ModuleType("langchain_community")
    lc_community.__path__ = []
    sys.modules.setdefault("langchain_community", lc_community)

    doc_loaders = types.ModuleType("langchain_community.document_loaders")

    class PyPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            Document = sys.modules["langchain_core.documents"].Document
            return [
                Document("stub page 1", {"page": 1}),
                Document("stub page 2", {"page": 2}),
            ]

    doc_loaders.PyPDFLoader = PyPDFLoader
    sys.modules.setdefault("langchain_community.document_loaders", doc_loaders)

    return Document, Blob


@pytest.fixture(autouse=True)
def stub_langchain_modules(monkeypatch):
    # install stubs before each test
    Document, Blob = _install_stub_langchain_modules()
    yield Document, Blob
    # No teardown necessary; tests are isolated in process


def _make_dummy_pdf(tmp_path: Path, name: str = "sample.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\nstartxref\n0\n%%EOF")
    return p


def test_nemo_enabled_uses_nemo_and_merges_pharma(monkeypatch, tmp_path):
    # Arrange: create dummy PDF and enable NeMo extraction via constructor
    os.environ.pop("ENABLE_NEMO_EXTRACTION", None)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _make_dummy_pdf(docs_dir)

    sys.path.insert(0, str(Path("src").resolve()))
    import document_loader as dl

    loader = dl.PDFDocumentLoader(str(docs_dir), enable_nemo_extraction=True)

    # Patch NeMo service to avoid importing real dependencies
    class StubResult:
        def __init__(self, docs):
            self.success = True
            self.documents = docs
            self.metadata = {"pharmaceutical": {"drug_names": ["aspirin"]}}
            self.error = None
            self.extraction_method = "nemo_vlm"

    class StubService:
        async def extract_document(self, *args, **kwargs):
            Document = sys.modules["langchain_core.documents"].Document
            docs = [Document("nemo page text", {"page": 1, "chunk_type": "semantic"})]
            return StubResult(docs)

    loader._ensure_nemo_service = lambda: StubService()

    # Act
    docs = loader.load_and_split()

    # Assert
    assert docs, "Expected documents from NeMo path"
    for d in docs:
        assert "extraction_method" in d.metadata
        assert "nemo" in d.metadata["extraction_method"].lower()
        # Pharmaceutical metadata merged at top-level
        assert "pharmaceutical" in d.metadata
        assert "aspirin" in d.metadata["pharmaceutical"].get("drug_names", [])

    metrics = loader.get_nemo_metrics()
    assert metrics.get("enabled") is True
    assert metrics.get("used_count", 0) >= 1
    assert metrics.get("success_count", 0) >= 1
    assert metrics.get("fallback_count", 0) == 0


def test_nemo_failure_falls_back_to_pypdf(monkeypatch, tmp_path):
    # Arrange: create dummy PDF and enable NeMo
    os.environ.pop("ENABLE_NEMO_EXTRACTION", None)
    # Allow fallback only in tests
    monkeypatch.setenv("NEMO_EXTRACTION_STRICT", "false")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _make_dummy_pdf(docs_dir)

    sys.path.insert(0, str(Path("src").resolve()))
    import document_loader as dl

    loader = dl.PDFDocumentLoader(str(docs_dir), enable_nemo_extraction=True)

    # Ensure fallback loader is bound at module-level even if import order differs
    class LocalPyPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            Document = sys.modules["langchain_core.documents"].Document
            return [Document("legacy page 1", {"page": 1}), Document("legacy page 2", {"page": 2})]

    monkeypatch.setattr(dl, "PyPDFLoader", LocalPyPDFLoader, raising=False)

    class StubFailResult:
        def __init__(self):
            self.success = False
            self.documents = []
            self.metadata = {}
            self.error = "simulated failure"
            self.extraction_method = "nemo_vlm"

    class FailingService:
        async def extract_document(self, *args, **kwargs):
            return StubFailResult()

    loader._ensure_nemo_service = lambda: FailingService()

    # Act
    docs = loader.load_documents()

    # Assert: fallback to PyPDF path yielded docs and extraction_method reflects pypdf
    assert docs, "Expected documents from fallback PyPDF path"
    assert any(d.metadata.get("extraction_method") == "pypdf" for d in docs)
    metrics = loader.get_nemo_metrics()
    assert metrics.get("fallback_count", 0) >= 1


def test_flag_disabled_uses_legacy(monkeypatch, tmp_path):
    # Arrange: disable via constructor and ensure legacy path is used
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _make_dummy_pdf(docs_dir)

    sys.path.insert(0, str(Path("src").resolve()))
    import document_loader as dl

    loader = dl.PDFDocumentLoader(str(docs_dir), enable_nemo_extraction=False)

    # Ensure legacy loader is available on the module
    class LocalPyPDFLoader2:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            Document = sys.modules["langchain_core.documents"].Document
            return [Document("legacy page 1", {"page": 1}), Document("legacy page 2", {"page": 2})]

    monkeypatch.setattr(dl, "PyPDFLoader", LocalPyPDFLoader2, raising=False)
    docs = loader.load_documents()
    assert docs, "Expected documents from legacy path"
    assert all(d.metadata.get("extraction_method") == "pypdf" for d in docs)
    assert loader.get_nemo_metrics().get("used_count", 0) == 0
