import os
import sys
from types import SimpleNamespace

import pytest


def mock_requests_get_eutils(payloads_by_endpoint):
    class Resp:
        def __init__(self, status=200, text="", json_data=None):
            self.status_code = status
            self._text = text
            self._json = json_data

        def raise_for_status(self):
            if not (200 <= self.status_code < 300):
                raise RuntimeError(f"HTTP {self.status_code}")

        @property
        def text(self):
            return self._text

        def json(self):
            return self._json

    def _get(url, params=None, timeout=None):
        if "esearch.fcgi" in url:
            data = payloads_by_endpoint.get("esearch")
            return Resp(json_data=data)
        if "efetch.fcgi" in url:
            data = payloads_by_endpoint.get("efetch_xml", "")
            return Resp(text=data)
        if "elink.fcgi" in url:
            data = payloads_by_endpoint.get("elink", {"linksets": []})
            return Resp(json_data=data)
        return Resp(status=404)

    return _get


def test_eutils_mapping(monkeypatch):
    # Arrange minimal E-utilities responses
    esearch = {"esearchresult": {"idlist": ["12345678"]}}
    # XML with basic fields
    efetch_xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345678</PMID>
          <Article>
            <ArticleTitle>Sample Title</ArticleTitle>
            <Abstract>
              <AbstractText Label="Background">This is background.</AbstractText>
              <AbstractText>And abstract body.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
              <Author><CollectiveName>Consortium X</CollectiveName></Author>
            </AuthorList>
            <Journal>
              <JournalIssue>
                <PubDate>
                  <Year>2024</Year><Month>May</Month><Day>01</Day>
                </PubDate>
              </JournalIssue>
              <Title>Journal of Tests</Title>
            </Journal>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="doi">10.1000/test.doi</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """.strip()
    elink = {"linksets": [{"linksetdbs": [{"dbto": "pmc", "links": [{"id": "PMC12345"}]}]}]}

    # Mock requests.get in eutils client
    sys.path.insert(0, str((__import__("pathlib").Path("src")).resolve()))
    import pubmed_eutils_client as eutils
    monkeypatch.setattr(eutils.requests, "get", mock_requests_get_eutils({
        "esearch": esearch,
        "efetch_xml": efetch_xml,
        "elink": elink,
    }))

    client = eutils.PubMedEutilsClient(email="test@example.com")
    results = client.search_and_fetch("glioblastoma", max_items=1)
    assert results and results[0]["pmid"] == "12345678"
    assert results[0]["title"] == "Sample Title"
    assert "Background:" in results[0]["abstract"] and "abstract body" in results[0]["abstract"]
    assert results[0]["authors"].startswith("Smith, John") and "Consortium X" in results[0]["authors"]
    assert results[0]["doi"] == "10.1000/test.doi"
    assert results[0]["journal"] == "Journal of Tests"
    assert results[0]["full_text_url"].startswith("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/")
    assert results[0]["provider"] == "pubmed_eutils"


def test_eutils_failure_fallback_to_openalex(monkeypatch):
    # E-utilities fails (HTTP error), scraper should fallback to OpenAlex path
    def failing_get(url, params=None, timeout=None):
        class Resp:
            def raise_for_status(self):
                raise RuntimeError("Failing E-utilities")
        return Resp()

    # Stub OpenAlex fallback
    sys.path.insert(0, str((__import__("pathlib").Path("src")).resolve()))
    import openalex_client as oa_mod
    # Ensure the package-relative import resolves to our stubbed module
    sys.modules["src.openalex_client"] = oa_mod
    class StubOAResp:
        def __init__(self):
            self.data = {"results": [{
                "id": "https://openalex.org/W1",
                "display_name": "Fallback Title",
                "abstract_inverted_index": {"Fallback": [0], "abstract": [1]},
                "authorships": [{"author": {"display_name": "Alice"}}],
                "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/99999999/"},
                "doi": "10.1000/fallback.doi",
                "host_venue": {"display_name": "Fallback Journal"},
                "primary_location": {"url": "https://example.org"}
            }]} 
    def oa_get(url, params=None, timeout=None):
        class Resp:
            def raise_for_status(self):
                pass
            def json(self):
                return StubOAResp().data
        return Resp()
    monkeypatch.setattr(oa_mod.requests, "get", oa_get)

    # Provide test-friendly stubs for langchain modules loaded by loader/vector_db
    import types
    lc_core = types.ModuleType("langchain_core"); lc_core.__path__ = []
    sys.modules["langchain_core"] = lc_core
    docs_mod = types.ModuleType("langchain_core.documents")
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = dict(metadata or {})
    docs_mod.Document = Document
    sys.modules["langchain_core.documents"] = docs_mod
    base_mod = types.ModuleType("langchain_core.documents.base")
    class Blob:
        @classmethod
        def from_data(cls, data: bytes, path: str | None = None):
            return cls()
    base_mod.Blob = Blob
    sys.modules["langchain_core.documents.base"] = base_mod
    lc_pkg = types.ModuleType("langchain"); lc_pkg.__path__ = []
    sys.modules["langchain"] = lc_pkg
    ts_mod = types.ModuleType("langchain.text_splitter")
    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs): pass
        def split_documents(self, documents): return list(documents)
    ts_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"] = ts_mod
    comm = types.ModuleType("langchain_community"); comm.__path__ = []
    sys.modules["langchain_community"] = comm
    loaders_mod = types.ModuleType("langchain_community.document_loaders")
    class PyPDFLoader:
        def __init__(self, path: str): self.path = path
        def load(self): return [Document("p1", {"page": 1})]
    loaders_mod.PyPDFLoader = PyPDFLoader
    sys.modules["langchain_community.document_loaders"] = loaders_mod
    vecstores_mod = types.ModuleType("langchain_community.vectorstores")
    class FAISS: pass
    vecstores_mod.FAISS = FAISS
    sys.modules["langchain_community.vectorstores"] = vecstores_mod

    # Patch requests.get used by eutils client (force failure)
    sys.path.insert(0, str((__import__("pathlib").Path("src")).resolve()))
    import pubmed_eutils_client as eutils
    monkeypatch.setattr(eutils.requests, "get", failing_get)

    # Load scraper and verify fallback returns apify-tagged result
    import importlib.util, importlib.machinery
    from pathlib import Path
    pkg = types.ModuleType("src"); pkg.__path__ = [str(Path("src").resolve())]
    sys.modules["src"] = pkg
    def _load_module(mod_name: str, file_path: str):
        loader = importlib.machinery.SourceFileLoader(mod_name, file_path)
        spec = importlib.util.spec_from_loader(mod_name, loader)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        loader.exec_module(mod)
        return mod

    _load_module("src.paper_schema", str(Path("src/paper_schema.py").resolve()))
    pubmed_mod = _load_module("src.pubmed_scraper", str(Path("src/pubmed_scraper.py").resolve()))
    # Override OpenAlexClient inside pubmed_scraper to a stub
    class FakeOA:
        def search_works(self, q, max_items=30):
            return [{
                "id": "https://openalex.org/W1",
                "display_name": "Fallback Title",
                "abstract_inverted_index": {"Fallback": [0], "abstract": [1]},
                "authorships": [{"author": {"display_name": "Alice"}}],
                "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/99999999/"},
                "doi": "10.1000/fallback.doi",
                "host_venue": {"display_name": "Fallback Journal"},
                "primary_location": {"url": "https://example.org"}
            }]
        def normalize_work(self, w):
            return {
                "pmid": "99999999",
                "doi": "10.1000/fallback.doi",
                "title": "Fallback Title",
                "abstract": "Fallback abstract",
                "authors": "Alice",
                "publication_date": "",
                "journal": "Fallback Journal",
                "full_text_url": "https://example.org",
                "provider": "openalex",
            }
    pubmed_mod.OpenAlexClient = FakeOA
    PubMedScraper = getattr(pubmed_mod, "PubMedScraper")

    scraper = PubMedScraper()
    results = scraper.search_pubmed("query", max_items=1)
    assert results and results[0]["provider"] == "openalex"
