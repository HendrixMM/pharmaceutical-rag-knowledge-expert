"""
Ollama Client Adapter

Lightweight HTTP client for local Ollama server, providing:
- Embeddings via nomic-embed-text (default)
- Chat/generation via llama3.1:8b (default)

API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    timeout_seconds: int = 60


class OllamaClientError(Exception):
    pass


class OllamaClient:
    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def list_models(self) -> List[str]:
        url = self._url("/api/tags")
        try:
            resp = requests.get(url, timeout=self.config.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            names = [m.get("name", "") for m in data.get("models", [])]
            logger.info("Ollama models available: %d", len(names))
            return names
        except Exception as e:
            raise OllamaClientError(f"Failed to list models: {e}")

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Create embeddings. Ollama API expects one prompt per request; loop for batch."""
        model = model or self.config.embed_model
        url = self._url("/api/embeddings")
        embeddings: List[List[float]] = []
        for t in texts:
            payload = {
                # Ollama embeddings support 'prompt' (commonly used); 'input' may also work for some versions
                "model": model,
                "prompt": t,
            }
            try:
                resp = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding") or []
                if not isinstance(emb, list):
                    raise ValueError("Invalid embedding format from Ollama")
                embeddings.append(emb)
            except Exception as e:
                raise OllamaClientError(f"Embedding request failed: {e}")
        return embeddings

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None,
             max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        model = model or self.config.chat_model
        url = self._url("/api/chat")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # Note: Ollama uses 'options' for generation params
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            # Response shape: { 'message': { 'role': 'assistant', 'content': '...' }, 'done': true }
            return data
        except Exception as e:
            raise OllamaClientError(f"Chat request failed: {e}")

