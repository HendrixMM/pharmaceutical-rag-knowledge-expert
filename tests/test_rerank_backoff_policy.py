import importlib

from src.enhanced_config import EnhancedRAGConfig


def test_backoff_mapping_disables_jitter_when_false(monkeypatch):
    # Arrange environment for config
    env = {
        "ENABLE_NVIDIA_BUILD_FALLBACK": "true",
        "RERANK_RETRY_BACKOFF_BASE": "0.75",
        "RERANK_RETRY_MAX_ATTEMPTS": "5",
        "RERANK_RETRY_JITTER": "false",
    }
    cfg = EnhancedRAGConfig.from_env(env)

    # Create a dummy wrapper to capture config instead of initializing SDK
    captured = {}

    class DummyWrapper:
        def __init__(self, config):
            captured["config"] = config

    # Monkeypatch the OpenAIWrapper used by the enhanced client
    import src.clients.nemo_client_enhanced as nce
    original = nce.OpenAIWrapper
    nce.OpenAIWrapper = DummyWrapper
    try:
        client = nce.EnhancedNeMoClient(config=cfg, enable_fallback=True)
        # Assert cloud client created and config captured
        assert "config" in captured
        c = captured["config"]
        assert c.request_backoff_base == 0.75
        assert c.rerank_retry_max_attempts == 5
        # Jitter disabled -> amplitude 0.0
        assert c.request_backoff_jitter == 0.0
    finally:
        nce.OpenAIWrapper = original


def test_backoff_mapping_sets_jitter_amplitude_when_true(monkeypatch):
    env = {
        "ENABLE_NVIDIA_BUILD_FALLBACK": "true",
        "RERANK_RETRY_BACKOFF_BASE": "1.0",
        "RERANK_RETRY_MAX_ATTEMPTS": "2",
        "RERANK_RETRY_JITTER": "true",
    }
    cfg = EnhancedRAGConfig.from_env(env)

    captured = {}

    class DummyWrapper:
        def __init__(self, config):
            captured["config"] = config

    import src.clients.nemo_client_enhanced as nce
    original = nce.OpenAIWrapper
    nce.OpenAIWrapper = DummyWrapper
    try:
        client = nce.EnhancedNeMoClient(config=cfg, enable_fallback=True)
        assert "config" in captured
        c = captured["config"]
        assert c.request_backoff_base == 1.0
        assert c.rerank_retry_max_attempts == 2
        # Jitter enabled -> amplitude equals base
        assert c.request_backoff_jitter == 1.0
    finally:
        nce.OpenAIWrapper = original

