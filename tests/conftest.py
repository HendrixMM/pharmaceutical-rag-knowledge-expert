import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "src" not in sys.modules:
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(ROOT / "src")]
    sys.modules["src"] = src_package

# Provide lightweight stubs for optional LangChain dependencies when unavailable.
if "langchain" not in sys.modules:
    langchain_module = types.ModuleType("langchain")
    schema_module = types.ModuleType("langchain.schema")
    embeddings_module = types.ModuleType("langchain.embeddings")
    embeddings_base_module = types.ModuleType("langchain.embeddings.base")
    text_splitter_module = types.ModuleType("langchain.text_splitter")
    chains_module = types.ModuleType("langchain.chains")
    prompts_module = types.ModuleType("langchain.prompts")
    llms_module = types.ModuleType("langchain.llms")
    llms_base_module = types.ModuleType("langchain.llms.base")

    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class Embeddings:
        pass

    class PromptTemplate:
        def __init__(self, template: str = "", input_variables=None, **_kwargs):
            self.template = template
            self.input_variables = input_variables or []

        def format(self, **kwargs):  # pragma: no cover - trivial formatting helper
            return self.template.format(**kwargs)

    schema_module.Document = Document
    prompts_module.PromptTemplate = PromptTemplate

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def split_documents(self, documents):  # pragma: no cover - simple passthrough
            return documents

    text_splitter_module.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

    class LLM:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def generate(self, *_args, **_kwargs):  # pragma: no cover - placeholder
            return {}

    class RetrievalQA:
        def __init__(self, retriever=None, llm=None, return_source_documents=False, chain_type: str | None = None):
            self.retriever = retriever
            self.llm = llm
            self.return_source_documents = return_source_documents
            self.chain_type = chain_type

        @classmethod
        def from_chain_type(
            cls,
            chain_type: str = "stuff",
            retriever=None,
            llm=None,
            return_source_documents: bool = False,
            **_kwargs,
        ):
            return cls(retriever=retriever, llm=llm, return_source_documents=return_source_documents, chain_type=chain_type)

        def __call__(self, *_args, **_kwargs):  # pragma: no cover - placeholder invocation
            return {"result": "", "source_documents": []}

    embeddings_base_module.Embeddings = Embeddings
    embeddings_module.base = embeddings_base_module
    chains_module.RetrievalQA = RetrievalQA
    prompts_module.PromptTemplate = PromptTemplate
    llms_base_module.LLM = LLM
    llms_module.base = llms_base_module
    langchain_module.schema = schema_module
    langchain_module.embeddings = embeddings_module
    langchain_module.text_splitter = text_splitter_module
    langchain_module.chains = chains_module
    langchain_module.prompts = prompts_module
    langchain_module.llms = llms_module
    sys.modules["langchain"] = langchain_module
    sys.modules["langchain.schema"] = schema_module
    sys.modules["langchain.embeddings"] = embeddings_module
    sys.modules["langchain.embeddings.base"] = embeddings_base_module
    sys.modules["langchain.text_splitter"] = text_splitter_module
    sys.modules["langchain.chains"] = chains_module
    sys.modules["langchain.prompts"] = prompts_module
    sys.modules["langchain.llms"] = llms_module
    sys.modules["langchain.llms.base"] = llms_base_module
else:
    Document = sys.modules["langchain.schema"].Document

if "langchain_community" not in sys.modules:
    community_module = types.ModuleType("langchain_community")
    vectorstores_module = types.ModuleType("langchain_community.vectorstores")
    document_loaders_module = types.ModuleType("langchain_community.document_loaders")

    class FakeFAISS:
        def __init__(self):
            self.docstore = types.SimpleNamespace(_dict={})
            self.index = types.SimpleNamespace(ntotal=0)

        @classmethod
        def from_texts(cls, texts, embedding, metadatas):
            instance = cls()
            instance.docstore._dict = {
                str(i): Document(page_content=text, metadata=meta)
                for i, (text, meta) in enumerate(zip(texts, metadatas))
            }
            instance.index.ntotal = len(texts)
            return instance

        @classmethod
        def load_local(cls, *args, **kwargs):
            return cls()

        def save_local(self, *args, **kwargs):
            return None

        def add_texts(self, texts, metadatas):
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                key = f"doc-{self.index.ntotal + i}"
                self.docstore._dict[key] = Document(page_content=text, metadata=meta)
            added_ids = [f"doc-{self.index.ntotal + i}" for i in range(len(texts))]
            self.index.ntotal += len(texts)
            return added_ids

        def similarity_search(self, *args, **kwargs):
            return []

        def similarity_search_with_score(self, *args, **kwargs):
            return []

    class _BaseLoader:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def load(self):  # pragma: no cover - simple stub
            return []

    class PyPDFLoader(_BaseLoader):
        pass

    class DirectoryLoader(_BaseLoader):
        pass

    vectorstores_module.FAISS = FakeFAISS
    community_module.vectorstores = vectorstores_module
    document_loaders_module.PyPDFLoader = PyPDFLoader
    document_loaders_module.DirectoryLoader = DirectoryLoader
    community_module.document_loaders = document_loaders_module
    sys.modules["langchain_community"] = community_module
    sys.modules["langchain_community.vectorstores"] = vectorstores_module
    sys.modules["langchain_community.document_loaders"] = document_loaders_module

if "faiss" not in sys.modules:
    sys.modules["faiss"] = types.ModuleType("faiss")
