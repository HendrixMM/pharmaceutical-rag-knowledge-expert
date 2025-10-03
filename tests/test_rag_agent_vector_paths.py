import json
import logging
import shutil
import sys
import types
from pathlib import Path
from typing import Optional
from typing import Union

langchain_module = types.ModuleType("langchain")
langchain_module.__path__ = []
langchain_module.__package__ = "langchain"
langchain_module.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
schema_module = types.ModuleType("langchain.schema")
chains_module = types.ModuleType("langchain.chains")
prompts_module = types.ModuleType("langchain.prompts")
llms_module = types.ModuleType("langchain.llms")
llms_module.__path__ = []
llms_module.__package__ = "langchain.llms"
llms_module.__spec__ = types.SimpleNamespace(submodule_search_locations=[])
llms_base_module = types.ModuleType("langchain.llms.base")
text_splitter_module = types.ModuleType("langchain.text_splitter")


class Document:
    def __init__(self, page_content: str, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class PromptTemplate:
    def __init__(self, *args, **kwargs):
        pass


class RetrievalQA:
    @classmethod
    def from_chain_type(cls, *args, **kwargs):  # pragma: no cover - placeholder stub
        return cls()


class LLM:
    pass


schema_module.Document = Document
prompts_module.PromptTemplate = PromptTemplate
chains_module.RetrievalQA = RetrievalQA
llms_base_module.LLM = LLM
llms_module.base = llms_base_module
text_splitter_module.RecursiveCharacterTextSplitter = type(
    "RecursiveCharacterTextSplitter",
    (),
    {
        "__init__": lambda self, *args, **kwargs: None,
        "split_text": lambda self, text: [text],
    },
)

langchain_module.schema = schema_module
langchain_module.chains = chains_module
langchain_module.prompts = prompts_module
langchain_module.llms = llms_module
langchain_module.text_splitter = text_splitter_module

sys.modules["langchain"] = langchain_module
sys.modules["langchain.schema"] = schema_module
sys.modules["langchain.chains"] = chains_module
sys.modules["langchain.prompts"] = prompts_module
sys.modules["langchain.llms"] = llms_module
sys.modules["langchain.llms.base"] = llms_base_module
sys.modules["langchain.text_splitter"] = text_splitter_module

from src.rag_agent import MEDICAL_DISCLAIMER, RAGAgent


class DummyEmbeddings:
    """Lightweight embeddings stub for path reconciliation tests."""

    def __init__(self, model_name: str, dimension: int = 512):
        self.model_name = model_name
        self._dimension = dimension

    def get_embedding_dimension(self) -> int:
        return self._dimension


class MinimalVectorDB:
    """Minimal vector DB stub with just enough surface for the agent helpers."""

    def __init__(self, path: Union[Path, str]):
        self.db_path = Path(path)
        self.index_name = "faiss_index"
        self.index_path: Optional[Path] = None
        self.metadata_path: Optional[Path] = None
        self.vectorstore = None
        self.load_calls = 0

    def save_index(self) -> bool:  # pragma: no cover - not used by these tests
        return True

    def load_index(self, **_kwargs) -> bool:  # pragma: no cover - default no-op
        self.load_calls += 1
        return True

    def update_base_path(self, new_path: Union[Path, str]) -> None:
        self.db_path = Path(new_path)


def test_check_and_migrate_legacy_index_reuses_existing_store(tmp_path, monkeypatch):
    base_path = tmp_path / "vector_db"
    base_path.mkdir()

    metadata = {"model_name": "dummy/model", "dimension": 384}
    (base_path / "embeddings_meta.json").write_text(json.dumps(metadata))

    per_model_path = base_path / "dummy_model"
    embeddings = DummyEmbeddings("dummy/model", dimension=384)

    class LegacyVectorDBStub:
        def __init__(self):
            self.db_path = base_path
            self.vectorstore = {"id": "legacy"}
            self.load_calls = 0

        def load_index(self, **_kwargs):
            self.load_calls += 1
            return True

        def save_index(self):  # pragma: no cover - not used by legacy stub
            raise AssertionError("legacy save_index should not be called")

    legacy_db_instance = LegacyVectorDBStub()

    def legacy_factory(embeddings_obj, db_path):
        assert Path(db_path) == base_path
        return legacy_db_instance

    monkeypatch.setattr("src.vector_database.VectorDatabase", legacy_factory)
    monkeypatch.setattr("src.rag_agent.VectorDatabase", legacy_factory)

    class AgentVectorDBStub:
        def __init__(self, path: Path):
            self.db_path = path
            self.index_name = "faiss_index"
            self.vectorstore = None
            self.save_calls = 0

        def save_index(self) -> bool:
            self.save_calls += 1
            return True

    agent_db = AgentVectorDBStub(per_model_path)
    agent_db.metadata_written = False

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = True
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(per_model_path)
    agent.embeddings = embeddings
    agent.vector_db = agent_db

    def record_metadata():
        agent_db.metadata_written = True

    agent._write_embeddings_metadata = record_metadata

    migrated = agent._check_and_migrate_legacy_index()

    assert migrated is True
    assert legacy_db_instance.load_calls == 1
    assert agent_db.vectorstore == legacy_db_instance.vectorstore
    assert agent_db.save_calls == 1
    assert agent_db.metadata_written is True


def test_ensure_vector_db_path_current_moves_matching_directory(tmp_path):
    base_path = tmp_path / "vector_db_root"
    legacy_path = tmp_path / "vector_db"
    legacy_path.mkdir()
    metadata = {"model_name": "dummy/model"}
    (legacy_path / "embeddings_meta.json").write_text(json.dumps(metadata))

    embeddings = DummyEmbeddings("dummy/model")

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = True
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(legacy_path)
    agent._active_vector_db_model = "legacy/model"
    agent.embeddings = embeddings
    agent.vector_db = MinimalVectorDB(legacy_path)

    reconciled, rebuild_required = agent._ensure_vector_db_path_current()

    expected_path = base_path / "dummy_model"

    assert reconciled is True
    assert rebuild_required is False
    assert Path(agent.vector_db_path) == expected_path
    assert agent.vector_db.db_path == expected_path
    assert expected_path.exists()
    assert not legacy_path.exists()


def test_check_embeddings_compatibility_detects_dimension_mismatch(tmp_path):
    vector_db_path = tmp_path / "per_model"
    vector_db_path.mkdir()

    metadata = {"model_name": "dummy/model", "dimension": 1024}
    (vector_db_path / "embeddings_meta.json").write_text(json.dumps(metadata))

    agent = object.__new__(RAGAgent)
    agent.vector_db_path = str(vector_db_path)
    agent.embeddings = DummyEmbeddings("dummy/model", dimension=768)

    assert agent._check_embeddings_compatibility() is False


def test_ensure_vector_db_path_current_handles_fallback_reconciliation(tmp_path):
    base_path = tmp_path / "vector_db"
    preferred_path = base_path / "preferred_model"
    preferred_path.mkdir(parents=True)
    (preferred_path / "embeddings_meta.json").write_text(json.dumps({"model_name": "preferred/model"}))

    embeddings = DummyEmbeddings("fallback/model")

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = True
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(preferred_path)
    agent._active_vector_db_model = "preferred/model"
    agent.embeddings = embeddings
    agent.vector_db = MinimalVectorDB(preferred_path)

    reconciled, rebuild_required = agent._ensure_vector_db_path_current()

    expected_path = base_path / "fallback_model"

    assert reconciled is True
    assert rebuild_required is True
    assert Path(agent.vector_db_path) == expected_path
    assert agent.vector_db.db_path == expected_path
    assert expected_path.exists()
    assert preferred_path.exists()


def test_setup_knowledge_base_reuses_legacy_index_in_legacy_mode(tmp_path, caplog):
    base_path = tmp_path / "vector_db"
    base_path.mkdir()

    class LegacyModeVectorDB(MinimalVectorDB):
        def load_index(self, **_kwargs) -> bool:
            self.load_calls += 1
            return True

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = False
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(base_path)
    agent.embeddings = DummyEmbeddings("dummy/model")
    agent.document_loader = None  # Not used when existing index loads
    agent.vector_db = LegacyModeVectorDB(base_path)
    agent._ensure_vector_db_path_current = lambda: (False, False)
    agent._check_embeddings_compatibility = lambda: True
    agent._check_and_migrate_legacy_index = lambda: False

    with caplog.at_level(logging.INFO):
        assert agent.setup_knowledge_base() is True

    assert agent.vector_db.load_calls == 1
    assert "reused compatible index" in caplog.text


def test_reconcile_before_query_reuses_matching_per_model_index(tmp_path):
    base_path = tmp_path / "vector_db"
    expected_path = base_path / "dummy_model"
    expected_path.mkdir(parents=True)

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = True
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(expected_path)
    agent._active_vector_db_model = "dummy/model"
    agent.embeddings = DummyEmbeddings("dummy/model")

    vector_db = MinimalVectorDB(expected_path)

    def fail_if_loaded(**_kwargs):  # pragma: no cover - should not run
        raise AssertionError("load_index should not be invoked when paths already match")

    vector_db.load_index = fail_if_loaded  # type: ignore[assignment]
    agent.vector_db = vector_db

    reconciled, rebuild_required, load_success, rebuild_message = agent._reconcile_before_query()

    assert reconciled is False
    assert rebuild_required is False
    assert load_success is True
    assert rebuild_message is None
    assert agent.vector_db.db_path == expected_path


def test_reconcile_before_query_switches_path_on_fallback_with_clear_logs(tmp_path, caplog):
    base_path = tmp_path / "vector_db"
    preferred_path = base_path / "preferred_model"
    preferred_path.mkdir(parents=True)
    (preferred_path / "embeddings_meta.json").write_text(json.dumps({"model_name": "preferred/model"}))

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = True
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(preferred_path)
    agent._active_vector_db_model = "preferred/model"
    agent.embeddings = DummyEmbeddings("fallback/model")
    agent.vector_db = MinimalVectorDB(preferred_path)

    with caplog.at_level(logging.INFO):
        reconciled, rebuild_required, load_success, rebuild_message = agent._reconcile_before_query()

    expected_path = base_path / "fallback_model"

    assert reconciled is True
    assert rebuild_required is True
    assert load_success is True
    assert rebuild_message is None
    assert Path(agent.vector_db_path) == expected_path
    assert agent.vector_db.db_path == expected_path
    assert (
        "Embedding model changed from 'preferred/model' to 'fallback/model'; reconciling vector DB path." in caplog.text
    )
    assert "Will rebuild at" in caplog.text


def test_reconcile_before_query_flags_metadata_mismatch(tmp_path, caplog):
    vector_db_path = tmp_path / "vector_db"
    vector_db_path.mkdir(parents=True)
    stored_metadata = {"model_name": "stored/model", "dimension": 1024}
    (vector_db_path / "embeddings_meta.json").write_text(json.dumps(stored_metadata))

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = False
    agent.base_vector_db_path = str(vector_db_path)
    agent.vector_db_path = str(vector_db_path)
    agent._active_vector_db_model = "stored/model"
    agent.embeddings = DummyEmbeddings("active/model", dimension=2048)
    agent.vector_db = MinimalVectorDB(vector_db_path)

    with caplog.at_level(logging.WARNING):
        reconciled, rebuild_required, load_success, rebuild_message = agent._reconcile_before_query()

    assert reconciled is False
    assert rebuild_required is True
    assert load_success is False
    assert rebuild_message is not None
    assert "rebuild the knowledge base" in rebuild_message
    assert "Stored embeddings metadata is incompatible" in caplog.text


def test_ensure_vector_db_path_current_flags_rebuild_when_move_fails(tmp_path, monkeypatch, caplog):
    base_path = tmp_path / "vector_db"
    previous_path = base_path / "legacy_model"
    previous_path.mkdir(parents=True)
    (previous_path / "embeddings_meta.json").write_text(json.dumps({"model_name": "dummy/model"}))

    agent = object.__new__(RAGAgent)
    agent.use_per_model_path = True
    agent.base_vector_db_path = str(base_path)
    agent.vector_db_path = str(previous_path)
    agent._active_vector_db_model = "legacy/model"
    agent.embeddings = DummyEmbeddings("dummy/model")
    agent.vector_db = MinimalVectorDB(previous_path)

    def fail_move(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "move", fail_move)

    with caplog.at_level(logging.INFO):
        reconciled, rebuild_required = agent._ensure_vector_db_path_current()

    expected_path = base_path / "dummy_model"

    assert reconciled is True
    assert rebuild_required is True
    assert Path(agent.vector_db_path) == expected_path
    assert agent.vector_db.db_path == expected_path
    assert "Failed to move vector DB directory" in caplog.text
    assert "index will be rebuilt" in caplog.text


def test_apply_disclaimer_honours_guardrail_metadata():
    agent = object.__new__(RAGAgent)
    agent.append_disclaimer_in_answer = True
    agent._guardrail_metadata = {}

    RAGAgent.set_guardrail_metadata(agent, {"disclaimer_added": True})

    base_text = "Evidence-based response"
    result = agent._apply_disclaimer(base_text)
    assert result == base_text

    # Guardrail flag should reset after being honoured so subsequent calls append the disclaimer
    follow_up = agent._apply_disclaimer(base_text)
    assert MEDICAL_DISCLAIMER in follow_up
