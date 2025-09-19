from src.document_loader import PDFDocumentLoader


def test_extract_pubmed_metadata_from_text_normalizes_mesh_terms():
    loader = PDFDocumentLoader.__new__(PDFDocumentLoader)

    sample_text = """MeSH terms: Heart Failure; heart failure; Cardiomyopathy; cardiomyopathy"""

    metadata = loader._extract_pubmed_metadata_from_text(sample_text)

    assert metadata["mesh_terms"] == ["Heart Failure", "Cardiomyopathy"]
