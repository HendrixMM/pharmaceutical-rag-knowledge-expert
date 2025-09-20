import json
import re
from typing import Optional

import pytest

from src import document_loader


class _FakeRegexModule:
    IGNORECASE = re.IGNORECASE
    MULTILINE = re.MULTILINE
    DOTALL = re.DOTALL
    UNICODE = re.UNICODE

    @staticmethod
    def compile(pattern: str, flags: int = 0):
        replacements = {
            r"\p{Lu}": r"\w",
            r"\p{L}": r"\w",
            r"\p{M}": "",
            r"\p{Zs}": r"\s",
            r"\p{Pd}": r"\-",
        }
        translated = pattern
        for search, replacement in replacements.items():
            translated = translated.replace(search, replacement)
        return re.compile(translated, flags)


@pytest.fixture
def configure_regex():
    original_module = document_loader._REGEX_MODULE
    document_loader._setup_regex_support(original_module)

    def _configure(module: Optional[object]) -> None:
        document_loader._setup_regex_support(module)

    yield _configure

    document_loader._setup_regex_support(original_module)


def test_extract_pubmed_metadata_from_text_normalizes_mesh_terms(configure_regex):
    configure_regex(document_loader._REGEX_MODULE)
    loader = document_loader.PDFDocumentLoader.__new__(document_loader.PDFDocumentLoader)

    sample_text = "MeSH terms: Heart Failure; heart failure; Cardiomyopathy; cardiomyopathy"

    metadata = loader._extract_pubmed_metadata_from_text(sample_text)

    assert metadata["mesh_terms"] == ["Heart Failure", "Cardiomyopathy"]


@pytest.mark.parametrize(
    "regex_module, expected_has_regex",
    [(_FakeRegexModule(), True), (None, False)],
    ids=["has_regex", "no_regex"],
)
def test_extract_pubmed_metadata_from_text_handles_authors_and_journal(
    configure_regex, regex_module, expected_has_regex
):
    configure_regex(regex_module)

    loader = document_loader.PDFDocumentLoader.__new__(document_loader.PDFDocumentLoader)

    sample_text = (
        "张伟, 李娜, 王芳\n"
        "Department of Biology, City Hospital\n"
        "Nature Medicine 2023;120(5):100-120\n"
        "Abstract\n"
        "The study investigates advanced unicode patterns."
    )

    metadata = loader._extract_pubmed_metadata_from_text(sample_text)

    assert metadata.get("authors") == "张伟, 李娜, 王芳"
    assert metadata.get("journal") == "Nature Medicine 2023;120(5):100-120"
    assert document_loader.HAS_REGEX is expected_has_regex


def test_extract_pubmed_metadata_normalizes_identifiers(tmp_path, configure_regex):
    configure_regex(document_loader._REGEX_MODULE)
    loader = document_loader.PDFDocumentLoader.__new__(document_loader.PDFDocumentLoader)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"")
    sidecar_data = {
        "doi": "https://doi.org/10.1234/ABC.DEF",
        "pmid": "PMID 12345",
        "authors": [
            {"last": "Doe", "first": "Jane"},
            {"last": "Smith", "first": "John"},
        ],
        "abstract": "Important findings.",
        "journal": "Nature Medicine",
        "mesh_terms": ["Heart Failure", "heart failure"],
    }
    sidecar_path = pdf_path.with_suffix(".pubmed.json")
    sidecar_path.write_text(json.dumps(sidecar_data), encoding="utf-8")

    metadata = loader._extract_pubmed_metadata(pdf_path)

    assert metadata["doi"] == "10.1234/abc.def"
    assert metadata["pmid"] == "12345"
    assert metadata["authors"] == "Doe, Jane, Smith, John"
    assert metadata["mesh_terms"] == ["Heart Failure"]


def test_truncate_metadata_fields_respects_environment(monkeypatch, configure_regex):
    configure_regex(document_loader._REGEX_MODULE)
    loader = document_loader.PDFDocumentLoader.__new__(document_loader.PDFDocumentLoader)

    monkeypatch.setenv("DOC_METADATA_AUTHORS_MAX_LEN", "10")
    monkeypatch.setenv("DOC_METADATA_ABSTRACT_MAX_LEN", "20")

    metadata = {
        "authors": "A" * 25,
        "abstract": "B" * 50,
    }

    loader._truncate_metadata_fields(metadata, "sample.pdf")

    assert len(metadata["authors"]) == 10
    assert len(metadata["abstract"]) == 20
