"""
Current Endpoint Verification Test

Tests the currently configured NVIDIA endpoints to verify they work
with the existing API key and compare against NVIDIA Build endpoints.

This helps understand which service tier the API key provides access to.

Usage:
  python scripts/current_endpoints_test.py

Prerequisites:
  - .env with a valid NVIDIA_API_KEY
"""
import os
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict

import requests

# Ensure local src is importable
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")


def test_current_embedding_endpoint(api_key: str) -> Dict[str, Any]:
    """Test the current ai.api.nvidia.com embedding endpoint."""
    print("Testing current embedding endpoint (ai.api.nvidia.com)...")

    url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Current-Endpoint-Test/1.0",
    }

    # Test with current configured model
    payload = {
        "input": ["This is a test for pharmaceutical research applications"],
        "model": "nvidia/nv-embedqa-e5-v5",
        "encoding_format": "float",
    }

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        end_time = time.time()

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time_ms": int((end_time - start_time) * 1000),
            "error": None if response.status_code == 200 else response.text,
            "endpoint": url,
            "model": "nvidia/nv-embedqa-e5-v5",
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "response_time_ms": None,
            "error": str(e),
            "endpoint": url,
            "model": "nvidia/nv-embedqa-e5-v5",
        }


def test_nv_embed_v1_on_current_endpoint(api_key: str) -> Dict[str, Any]:
    """Test nvidia/nv-embed-v1 on current endpoint."""
    print("Testing nvidia/nv-embed-v1 on current endpoint...")

    url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Current-Endpoint-Test/1.0",
    }

    payload = {
        "input": ["This is a test for pharmaceutical research applications"],
        "model": "nvidia/nv-embed-v1",
        "encoding_format": "float",
    }

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        end_time = time.time()

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time_ms": int((end_time - start_time) * 1000),
            "error": None if response.status_code == 200 else response.text,
            "endpoint": url,
            "model": "nvidia/nv-embed-v1",
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "response_time_ms": None,
            "error": str(e),
            "endpoint": url,
            "model": "nvidia/nv-embed-v1",
        }


def test_alternative_llm_endpoint(api_key: str) -> Dict[str, Any]:
    """Test LLM on different possible endpoints."""
    print("Testing meta/llama-3.1-8b-instruct on alternative endpoints...")

    # Try different endpoint variations
    endpoints_to_try = [
        "https://ai.api.nvidia.com/v1/chat/completions",
        "https://api.nvidia.com/v1/chat/completions",
        "https://integrate.api.nvidia.com/v1/chat/completions",
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Current-Endpoint-Test/1.0",
    }

    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "What are pharmaceutical drug interactions?"}],
        "max_tokens": 50,
        "temperature": 0.1,
    }

    for url in endpoints_to_try:
        print(f"  Trying: {url}")
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            end_time = time.time()

            if response.status_code == 200:
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "response_time_ms": int((end_time - start_time) * 1000),
                    "error": None,
                    "endpoint": url,
                    "model": "meta/llama-3.1-8b-instruct",
                    "sample_output": response.json(),
                }
            else:
                print(f"    Failed: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")

    return {
        "success": False,
        "status_code": None,
        "response_time_ms": None,
        "error": "All endpoints failed",
        "endpoint": "Multiple tried",
        "model": "meta/llama-3.1-8b-instruct",
    }


def print_test_results(test_name: str, results: Dict[str, Any]) -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"Model: {results['model']}")
    print(f"Endpoint: {results['endpoint']}")
    print(f"{'='*60}")

    if results["success"]:
        print("✅ SUCCESS")
        print(f"Response Time: {results['response_time_ms']}ms")
        if "sample_output" in results and results["sample_output"]:
            if "choices" in results["sample_output"]:
                content = results["sample_output"]["choices"][0]["message"]["content"]
                print(f"Sample Response: {content[:150]}...")
    else:
        print("❌ FAILED")
        print(f"Status Code: {results['status_code']}")
        print(f"Error: {results['error']}")


def main():
    print("Current NVIDIA Endpoint Verification Test")
    print(f"{'='*60}")

    # Get API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        return 1

    print(f"API Key: {api_key[:10]}...")

    # Test current working embedding endpoint
    current_embedding = test_current_embedding_endpoint(api_key)
    print_test_results("Current Embedding Endpoint", current_embedding)

    # Test nv-embed-v1 on current endpoint
    nv_embed_v1 = test_nv_embed_v1_on_current_endpoint(api_key)
    print_test_results("nv-embed-v1 on Current Endpoint", nv_embed_v1)

    # Test LLM on alternative endpoints
    llm_test = test_alternative_llm_endpoint(api_key)
    print_test_results("LLM Alternative Endpoints", llm_test)

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    if current_embedding["success"]:
        print("✅ Your current NeMo Retriever embedding setup works")
        print("   This indicates your API key has access to ai.api.nvidia.com")

    if nv_embed_v1["success"]:
        print("✅ nvidia/nv-embed-v1 is accessible on your current endpoint")
        print("   You can use this as a fallback embedding model")
    else:
        print("❌ nvidia/nv-embed-v1 not available on current endpoint")

    if llm_test["success"]:
        print(f"✅ meta/llama-3.1-8b-instruct works on: {llm_test['endpoint']}")
    else:
        print("❌ meta/llama-3.1-8b-instruct not accessible on tested endpoints")

    print(f"\nConclusion: Your API key appears to work with {current_embedding['endpoint'].split('/')[2]} domain")

    return 0


if __name__ == "__main__":
    sys.exit(main())
