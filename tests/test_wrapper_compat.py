"""
Compatibility tests for NeMoClientWrapper and NVIDIA Build credit tracker.

Focus areas:
- Legacy health_check return shape mapping
- Latency instrumentation in embed_texts
- Safe division in CreditUsageTracker monthly usage percent
"""
import os
from unittest.mock import patch

import pytest

from src.nemo_retriever_client import NeMoAPIResponse, NeMoClientWrapper


@pytest.mark.asyncio
async def test_wrapper_health_check_legacy_shape():
    # Ensure cloud-first so EnhancedNeMoClient swallows missing NVIDIA_API_KEY
    with patch.dict(os.environ, {"ENABLE_NVIDIA_BUILD_FALLBACK": "true"}, clear=False):
        wrapper = NeMoClientWrapper()

        # Mock get_service_status to a known legacy-like payload
        fake_status = {
            "services": {
                "embedding": {"status": "healthy"},
                "reranking": {"status": "healthy"},
            },
            "metrics": {},
        }
        with patch.object(wrapper, "get_service_status", return_value=fake_status):
            health = await wrapper.health_check()
            assert isinstance(health, dict)
            assert "embedding" in health
            assert health["embedding"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_wrapper_embed_texts_latency_and_normalization():
    with patch.dict(os.environ, {"ENABLE_NVIDIA_BUILD_FALLBACK": "true"}, clear=False):
        wrapper = NeMoClientWrapper()

        # Return dict with embeddings to exercise normalization path
        async def fake_embed_texts(texts, model):
            return {"embeddings": [[0.1, 0.2]], "model": model}

        wrapper._enhanced.embed_texts = fake_embed_texts  # type: ignore[attr-defined]

    result = await wrapper.embed_texts(["x"], model="nv-embedqa-e5-v5")
    assert isinstance(result, NeMoAPIResponse)
    assert result.success is True
    assert "embeddings" in result.data
    assert isinstance(result.response_time_ms, (int, float))
    assert result.response_time_ms >= 0.0


@pytest.mark.asyncio
async def test_wrapper_rerank_legacy_shape():
    with patch.dict(os.environ, {"ENABLE_NVIDIA_BUILD_FALLBACK": "true"}, clear=False):
        wrapper = NeMoClientWrapper()

        async def fake_rerank_async(query, passages, model=None, top_n=None):
            # Mimic ClientResponse shape expected by wrapper
            return type(
                "CR",
                (),
                {
                    "success": True,
                    "data": {"reranked_passages": [{"text": passages[0], "score": 0.9, "index": 0}]},
                    "error": None,
                },
            )()

        wrapper._enhanced.rerank_passages_async = fake_rerank_async  # type: ignore[attr-defined]

        result = await wrapper.rerank_passages("q", ["a", "b"], model="nv-rerankqa-mistral4b-v3", top_k=1)
        assert isinstance(result, NeMoAPIResponse)
        assert result.success is True
        assert "reranked_passages" in result.data
        assert isinstance(result.response_time_ms, (int, float))


def test_credit_usage_tracker_safe_division(monkeypatch):
    # Inject a dummy openai wrapper module to satisfy import-time dependency
    import sys
    import types

    dummy = types.ModuleType("src.clients.openai_wrapper")
    dummy.OpenAIWrapper = object

    class DummyCfg:
        pass

    dummy.NVIDIABuildConfig = DummyCfg

    class DummyErr(Exception):
        pass

    dummy.NVIDIABuildError = DummyErr
    sys.modules["src.clients.openai_wrapper"] = dummy
    sys.modules[".clients.openai_wrapper"] = dummy

    from src.nvidia_build_client import CreditUsageTracker

    tracker = CreditUsageTracker(monthly_free_requests=0)
    # Simulate some prior usage
    tracker.requests_this_month = 5
    summary = tracker.get_usage_summary()
    assert summary["monthly_limit"] == 0
    assert summary["monthly_usage_percent"] == 0
