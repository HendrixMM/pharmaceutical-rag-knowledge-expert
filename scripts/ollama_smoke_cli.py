"""
Ollama Smoke CLI: exercise local chat and embeddings via EnhancedNeMoClient

Usage:
  python scripts/ollama_smoke_cli.py

It will:
- Load .env
- Print endpoint priority
- Run a pharma chat prompt
- Run an embedding request
"""

from __future__ import annotations

import json
import os

def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    try:
        from src.enhanced_config import EnhancedRAGConfig
        from src.clients.nemo_client_enhanced import EnhancedNeMoClient
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[1]
        sys.path.append(str(ROOT))
        from src.enhanced_config import EnhancedRAGConfig
        from src.clients.nemo_client_enhanced import EnhancedNeMoClient

    cfg = EnhancedRAGConfig.from_env()
    print("Ollama enabled:", cfg.enable_ollama)
    print("Cloud-first enabled:", cfg.enable_nvidia_build_fallback)
    print("Ollama config:")
    print(json.dumps(cfg.get_ollama_config(), indent=2))

    client = EnhancedNeMoClient(config=cfg, enable_fallback=True, pharmaceutical_optimized=True)
    status = client.get_endpoint_status()
    print("\nEndpoint Priority:")
    for ep in status.get("endpoint_priority", []):
        print(f"- {ep['name']}: {ep['base_url']} ({ep['priority']})")

    # Chat test
    print("\nChat test (metformin MoA):")
    chat = client.create_chat_completion([
        {"role": "user", "content": "In one sentence, explain metformin mechanism of action."}
    ])
    print("  success:", chat.success, "endpoint:", (chat.endpoint_type.value if chat.endpoint_type else None))
    if chat.success:
        print("  response:", (chat.data.get("content", "") or "")[:200])
    else:
        print("  error:", chat.error)

    # Embedding test
    print("\nEmbedding test:")
    emb = client.create_embeddings(["metformin pharmacokinetics and interactions"])
    print("  success:", emb.success, "endpoint:", (emb.endpoint_type.value if emb.endpoint_type else None))
    if emb.success:
        embs = emb.data.get("embeddings")
        dim = (len(embs[0]) if isinstance(embs, list) and embs and isinstance(embs[0], list) else "unknown")
        print("  dimension:", dim)
    else:
        print("  error:", emb.error)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
