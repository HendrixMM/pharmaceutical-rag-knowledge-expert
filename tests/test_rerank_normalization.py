"""
Tests for rerank normalization contract via NeMoClientWrapper.

Verifies that regardless of cloud or NeMo shapes, wrapper returns
NeMoAPIResponse with data.reranked_passages as [{"text": str, "score": float}]
sorted by descending score.
"""

import os
import pytest

from src.nemo_retriever_client import NeMoClientWrapper, NeMoAPIResponse


class _DummyCloudRerank:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, candidates, top_n=None, model=None):
        # Simulate cloud returning items with 'score'
        self.calls += 1
        raw = [
            {"text": candidates[0], "score": 0.1},
            {"text": candidates[1], "score": 0.9},
        ]
        return raw[:top_n] if top_n else raw


class _DummyNemoRerank:
    async def rerank_passages(self, query, passages, model, top_k=None):
        # Simulate NeMo returning relevance-based items
        data = {
            "reranked_passages": [
                {"passage": passages[0], "relevance": 0.2, "index": 0},
                {"passage": passages[1], "relevance": 0.8, "index": 1},
            ][: (top_k or 2)]
        }
        return NeMoAPIResponse(success=True, data=data, service="reranking", model=model)


@pytest.mark.asyncio
async def test_wrapper_rerank_normalized_cloud():
    with pytest.MonkeyPatch.context() as mp:
        # Ensure wrapper can initialize without trying real cloud
        mp.setenv("ENABLE_NVIDIA_BUILD_FALLBACK", "false")
        w = NeMoClientWrapper()
        # Inject dummy cloud client
        dummy = _DummyCloudRerank()
        w._enhanced.cloud_client = dummy
        # Call wrapper
        res = await w.rerank_passages(
            query="q",
            passages=["a", "b"],
            model="meta/llama-3_2-nemoretriever-500m-rerank-v2",
            top_k=2,
        )
        assert res.success is True
        assert isinstance(res, NeMoAPIResponse)
        items = res.data.get("reranked_passages", [])
        assert all(set(x.keys()) == {"text", "score"} for x in items)
        # Sorted desc by score
        scores = [x["score"] for x in items]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_wrapper_rerank_normalized_nemo():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ENABLE_NVIDIA_BUILD_FALLBACK", "false")
        w = NeMoClientWrapper()
        # Force nemo path by clearing cloud client and injecting dummy nemo
        w._enhanced.cloud_client = None
        w._enhanced.nemo_client = _DummyNemoRerank()
        # Call wrapper
        res = await w.rerank_passages(
            query="q",
            passages=["x", "y"],
            model="nvidia/nv-rerankqa-mistral4b-v3",
            top_k=2,
        )
        assert res.success is True
        items = res.data.get("reranked_passages", [])
        assert all(set(x.keys()) == {"text", "score"} for x in items)
        scores = [x["score"] for x in items]
        assert scores == sorted(scores, reverse=True)

