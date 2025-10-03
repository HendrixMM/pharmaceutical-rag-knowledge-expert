#!/usr/bin/env python3
"""
Standalone test script for NVIDIA Build reranking via OpenAIWrapper.
"""
from __future__ import annotations

import json
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # nosec B110 - optional dotenv import should not fail tests
    pass

try:
    from src.clients.openai_wrapper import OpenAIWrapper
except ModuleNotFoundError:
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    from src.clients.openai_wrapper import OpenAIWrapper  # type: ignore


def main() -> int:
    if not os.getenv("NVIDIA_API_KEY"):
        print("NVIDIA_API_KEY not set; please export or add to .env")
        return 1

    wrapper = OpenAIWrapper()
    query = "metformin drug interactions"
    candidates = [
        "metformin lowers blood glucose",
        "metformin has interactions with cationic drugs",
        "metformin side effects include GI disturbance",
    ]
    try:
        rankings = wrapper.rerank(query=query, candidates=candidates, top_n=3)
        print("Rerank Results:")
        print(json.dumps(rankings, indent=2))
        return 0
    except Exception as e:
        print("Rerank failed:", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
