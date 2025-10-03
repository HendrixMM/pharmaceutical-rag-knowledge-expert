"""
Tests for pharma disclaimer flags in EnhancedNeMoClient chat responses.

This test avoids network by injecting a dummy cloud client that mimics the
OpenAIWrapper chat response shape.
"""
import os
from unittest.mock import patch

import pytest

from src.clients.nemo_client_enhanced import EnhancedNeMoClient
from src.enhanced_config import EnhancedRAGConfig


class _DummyChoice:
    def __init__(self, content: str):
        self.message = type("_DummyMsg", (), {"content": content})


class _DummyChatResponse:
    def __init__(self, content: str, model: str):
        self.choices = [_DummyChoice(content)]
        self.model = model
        self.usage = {"prompt_tokens": 1, "total_tokens": 1}


class _DummyCloudClient:
    def __init__(self, model: str = "meta/llama-3.1-8b-instruct"):
        self._model = model

    def create_chat_completion(self, messages, model=None, max_tokens=None, temperature=None):
        # Return a stable dummy response
        return _DummyChatResponse("This is a medical summary.", model or self._model)


@pytest.mark.asyncio
async def test_enhanced_client_chat_adds_disclaimer_flag():
    with patch.dict(
        os.environ,
        {
            "PHARMA_REQUIRE_DISCLAIMER": "true",
            "APPEND_DISCLAIMER_IN_ANSWER": "true",
            "MEDICAL_DISCLAIMER": "[DISCLAIMER] Not medical advice.",
            # Ensure cloud-first strategy does not require actual OpenAI SDK
            "ENABLE_NVIDIA_BUILD_FALLBACK": "false",
        },
        clear=False,
    ):
        cfg = EnhancedRAGConfig.from_env()
        client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)

        # Inject dummy cloud client regardless of OpenAI availability
        client.cloud_client = _DummyCloudClient()

        resp = client.create_chat_completion(
            [{"role": "user", "content": "What are common interactions for metformin?"}]
        )

        assert resp.success is True
        assert isinstance(resp.data, dict)
        assert resp.data.get("disclaimer_added") is True
        assert "disclaimer_text" in resp.data
        # When APPEND_DISCLAIMER_IN_ANSWER=true, content should include disclaimer
        assert "[DISCLAIMER] Not medical advice." in resp.data.get("content", "")


@pytest.mark.asyncio
async def test_enhanced_client_chat_disclaimer_cached_and_count_once():
    # Verify disclaimer flags persist on cached responses and cloud call count is 1
    with patch.dict(
        os.environ,
        {
            "PHARMA_REQUIRE_DISCLAIMER": "true",
            "APPEND_DISCLAIMER_IN_ANSWER": "true",
            "MEDICAL_DISCLAIMER": "[DISCLAIMER] Not medical advice.",
            "ENABLE_NVIDIA_BUILD_FALLBACK": "false",
        },
        clear=False,
    ):
        cfg = EnhancedRAGConfig.from_env()
        client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)

        class _CountingDummy(_DummyCloudClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def create_chat_completion(self, *args, **kwargs):
                self.calls += 1
                return super().create_chat_completion(*args, **kwargs)

        dummy = _CountingDummy()
        client.cloud_client = dummy

        messages = [{"role": "user", "content": "Please summarize aspirin PK."}]
        # First call (populates cache)
        resp1 = client.create_chat_completion(messages)
        # Second call (should hit cache)
        resp2 = client.create_chat_completion(messages)

        assert dummy.calls == 1  # ensured single cloud invocation due to cache
        for resp in (resp1, resp2):
            assert resp.success is True
            assert resp.data.get("disclaimer_added") is True
            assert "[DISCLAIMER] Not medical advice." in resp.data.get("content", "")


@pytest.mark.asyncio
async def test_enhanced_client_chat_disclaimer_flag_without_append():
    # Verify flag is set but content not modified when append=false
    with patch.dict(
        os.environ,
        {
            "PHARMA_REQUIRE_DISCLAIMER": "true",
            "APPEND_DISCLAIMER_IN_ANSWER": "false",
            "MEDICAL_DISCLAIMER": "[DISCLAIMER] Not medical advice.",
            "ENABLE_NVIDIA_BUILD_FALLBACK": "false",
        },
        clear=False,
    ):
        cfg = EnhancedRAGConfig.from_env()
        client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)
        client.cloud_client = _DummyCloudClient()

        resp = client.create_chat_completion([{"role": "user", "content": "Interactions for ibuprofen?"}])

        assert resp.success is True
        assert resp.data.get("disclaimer_added") is True
        assert resp.data.get("disclaimer_text") == "[DISCLAIMER] Not medical advice."
        assert "[DISCLAIMER] Not medical advice." not in resp.data.get("content", "")
