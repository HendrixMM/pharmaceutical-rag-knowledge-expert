from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain.schema import Document

from src.vector_database import VectorDatabase


class DummyEmbeddings:
    """Minimal embeddings stub for testing."""

    def embed_documents(self, texts):
        return []

    def embed_query(self, text):
        return []


def test_add_documents_enriches_when_pharma_enabled(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db"))
    doc = Document(page_content="Test", metadata={})
    db.vectorstore = MagicMock()
    db.vectorstore.add_texts.return_value = ["doc-1"]
    db._pharma_metadata_enabled = True

    db._prepare_pharmaceutical_document = MagicMock(side_effect=lambda value: value)
    db._ensure_document_metadata = MagicMock(side_effect=lambda value: value)
    db._detect_pharma_metadata = MagicMock(return_value=True)
    db._record_doc_ids = MagicMock()

    result = db.add_documents([doc])

    assert result is True
    db._prepare_pharmaceutical_document.assert_called_once_with(doc)
    db._ensure_document_metadata.assert_not_called()
    db.vectorstore.add_texts.assert_called_once()


def test_add_documents_respects_auto_enrich_flag(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db2"))
    doc = Document(page_content="Test", metadata={})
    db.vectorstore = MagicMock()
    db.vectorstore.add_texts.return_value = ["doc-2"]
    db._pharma_metadata_enabled = True

    db._prepare_pharmaceutical_document = MagicMock(side_effect=lambda value: value)
    db._ensure_document_metadata = MagicMock(side_effect=lambda value: value)
    db._detect_pharma_metadata = MagicMock(return_value=True)
    db._record_doc_ids = MagicMock()

    result = db.add_documents([doc], auto_enrich_on_add=False)

    assert result is True
    db._prepare_pharmaceutical_document.assert_not_called()
    db._ensure_document_metadata.assert_called_once_with(doc)


def test_iter_documents_uses_docstore_search_when_dict_missing(tmp_path):
    db = VectorDatabase(DummyEmbeddings(), db_path=str(tmp_path / "db3"))
    document = Document(page_content="PK details", metadata={})

    class StubDocstore:
        def search(self, key):
            if key in ("", "doc-1"):
                return document
            return None

    db.vectorstore = SimpleNamespace(
        docstore=StubDocstore(),
        index_to_docstore_id={"0": "doc-1"},
    )
    db._docstore_ids = []

    docs = db._iter_documents()

    assert docs == [document]
