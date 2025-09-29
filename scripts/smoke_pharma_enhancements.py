"""
Offline smoke test for pharmaceutical enhancements.

This script avoids network by stubbing the OpenAI client used by OpenAIWrapper.
It validates:
- Config pharma flags and settings
- Query classification for pharma types
- Pharma-aware request optimization and retry path
- Cost metrics exposure
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import sys
import types as _types

# Inject a minimal fake 'openai' module to avoid dependency during smoke test
_fake_openai = _types.ModuleType("openai")
class _FakeOpenAIError(Exception):
    pass
_fake_exceptions = _types.ModuleType("openai._exceptions")
_fake_exceptions.OpenAIError = _FakeOpenAIError
_fake_types = _types.ModuleType("openai.types")
_fake_chat = _types.ModuleType("openai.types.chat")
class _FakeChatCompletion:  # placeholder to satisfy import
    pass
_fake_chat.ChatCompletion = _FakeChatCompletion
class _FakeCreateEmbeddingResponse:  # placeholder to satisfy import
    pass
_fake_types.CreateEmbeddingResponse = _FakeCreateEmbeddingResponse
_fake_types.chat = _fake_chat

class _FakeOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        pass

_fake_openai.OpenAI = _FakeOpenAI
sys.modules.setdefault("openai", _fake_openai)
sys.modules.setdefault("openai._exceptions", _fake_exceptions)
sys.modules.setdefault("openai.types", _fake_types)
sys.modules.setdefault("openai.types.chat", _fake_chat)

try:
    from src.enhanced_config import EnhancedRAGConfig
    from src.clients.openai_wrapper import OpenAIWrapper, NVIDIABuildConfig, NVIDIABuildError
except ModuleNotFoundError:
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    sys.path.insert(0, _os.path.join(_root, "src"))
    from enhanced_config import EnhancedRAGConfig  # type: ignore
    from clients.openai_wrapper import OpenAIWrapper, NVIDIABuildConfig, NVIDIABuildError  # type: ignore


class _FakeUsage:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens

    def dict(self) -> Dict[str, Any]:
        return {"total_tokens": self.total_tokens}


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, model: str, content: str, tokens: int, params: Dict[str, Any]) -> None:
        self.model = model
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(tokens)
        # For introspection
        self._params = params


class _FakeCompletions:
    def __init__(self, owner: "_FakeChat", fail_once: bool = False) -> None:
        self._owner = owner
        self._fail_once = fail_once
        self._called = 0

    def create(self, *, model: str, messages: List[Dict[str, str]], max_tokens: Optional[int], temperature: Optional[float], **kwargs):
        self._called += 1
        # Simulate first-call failure for retry if enabled
        if self._fail_once and self._called == 1:
            raise RuntimeError("simulated network error")

        params = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
        # Produce minimal content echo
        user_content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        content = f"OK: {user_content[:80]}"
        tokens = max(8, min(128, len(user_content.split())))
        return _FakeResponse(model, content, tokens, params)


class _FakeChat:
    def __init__(self, fail_once: bool = False) -> None:
        self.completions = _FakeCompletions(self, fail_once=fail_once)


class _FakeOpenAIClient:
    def __init__(self, fail_once: bool = False) -> None:
        self.chat = _FakeChat(fail_once=fail_once)
        self.models = type("_M", (), {"list": lambda self: type("_R", (), {"data": []})()})()


def main() -> None:
    # Load config and print pharma flags/settings
    conf = EnhancedRAGConfig.from_env()
    snapshot = {
        "feature_flags": conf.get_feature_flags(),
        "pharma_settings": conf.get_pharma_settings(),
        "nvidia_build": conf.get_nvidia_build_config(),
    }

    # Instantiate wrapper with dummy key and stub client
    wrapper = OpenAIWrapper(NVIDIABuildConfig(api_key="dummy", pharmaceutical_optimized=True))
    # Enable single retry path by failing once
    wrapper.client = _FakeOpenAIClient(fail_once=True)

    # Classification samples
    samples = {
        "ddi": "Does simvastatin interact with clarithromycin?",
        "pk": "What is the half-life and clearance of metformin?",
        "trial": "Summarize outcomes from a phase 3 RCT of semaglutide in T2D.",
        "general": "Explain role of beta blockers in hypertension.",
    }
    classes = {k: wrapper.classify_pharma_query(v) for k, v in samples.items()}

    # Exercise create_chat_completion with retry (first call fails)
    resp = wrapper.create_chat_completion([
        {"role": "user", "content": samples["ddi"]}
    ])

    output = {
        "config": snapshot,
        "classification": classes,
        "chat_result": {
            "model": resp.model,
            "content_preview": resp.choices[0].message.content[:60],
            "tokens": resp.usage.total_tokens,
            "params_used": getattr(resp, "_params", {}),
        },
        "cost_metrics": wrapper.get_cost_metrics(),
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
