"""
Tests for query-aware input_type forwarding in EnhancedNeMoClient embeddings.

These tests avoid network by injecting a dummy cloud client that mimics the
OpenAIWrapper embeddings.create response shape.
"""

import os
import pytest

from src.clients.nemo_client_enhanced import EnhancedNeMoClient
from src.enhanced_config import EnhancedRAGConfig


class _DummyEmbItem:
    def __init__(self, vec):
        self.embedding = vec


class _DummyEmbResponse:
    def __init__(self, model: str, n: int = 1):
        self.data = [_DummyEmbItem([0.0, 0.1, 0.2])] * max(1, n)
        self.model = model
        self.usage = {"prompt_tokens": n, "total_tokens": n}


class _DummyEmbCloud:
    def __init__(self):
        self.calls = []

    def create_embeddings(self, texts, model=None, input_type=None, is_query=None, **kwargs):
        self.calls.append({
            "texts": texts,
            "model": model,
            "input_type": input_type,
            "is_query": is_query,
            "kwargs": kwargs,
        })
        return _DummyEmbResponse(model=model, n=len(texts) if isinstance(texts, list) else 1)


@pytest.mark.asyncio
async def test_embed_texts_forwards_query_input_type():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ENABLE_NVIDIA_BUILD_FALLBACK", "false")
        mp.setenv("ENABLE_NVIDIA_BUILD_EMBEDDING_INPUT_TYPE", "true")
        cfg = EnhancedRAGConfig.from_env()
        client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)
        dummy = _DummyEmbCloud()
        client.cloud_client = dummy
        # Use a question so embed_texts (is_query=True) is appropriate
        await client.embed_texts(["What is metformin?"], model="nvidia/nv-embedqa-e5-v5")
        assert dummy.calls, "No cloud embedding call recorded"
        call = dummy.calls[-1]
        assert call["is_query"] is True
        assert call["input_type"] == "query"


def test_create_embeddings_auto_detects_query_input_type():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ENABLE_NVIDIA_BUILD_FALLBACK", "false")
        mp.setenv("ENABLE_NVIDIA_BUILD_EMBEDDING_INPUT_TYPE", "true")
        cfg = EnhancedRAGConfig.from_env()
        client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)
        dummy = _DummyEmbCloud()
        client.cloud_client = dummy
        # Do not pass is_query; auto-detection should infer query for '?' and short text
        client.create_embeddings(["What is aspirin?"], model="nvidia/nv-embedqa-e5-v5")
        call = dummy.calls[-1]
        assert call["is_query"] is True
        assert call["input_type"] == "query"


def test_create_embeddings_uses_passage_for_non_query():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ENABLE_NVIDIA_BUILD_FALLBACK", "false")
        mp.setenv("ENABLE_NVIDIA_BUILD_EMBEDDING_INPUT_TYPE", "true")
        cfg = EnhancedRAGConfig.from_env()
        client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)
        dummy = _DummyEmbCloud()
        client.cloud_client = dummy
        long_passage = "This is a long passage about clinical pharmacology. " * 10  # no question mark, long text
        client.create_embeddings([long_passage], model="nvidia/nv-embedqa-e5-v5")
        call = dummy.calls[-1]
        assert call["is_query"] in (False, None)
        assert call["input_type"] == "passage"

