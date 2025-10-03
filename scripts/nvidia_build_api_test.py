"""
NVIDIA Build API Access Verification Test

Tests access to specific models through NVIDIA Build API endpoints:
- nvidia/nv-embed-v1 (embedding model)
- meta/llama-3.1-8b-instruct (language model)

This script verifies that your NVIDIA_API_KEY has access to these models
through the NVIDIA Build platform (integrate.api.nvidia.com).

Usage:
  python scripts/nvidia_build_api_test.py

Prerequisites:
  - .env with a valid NVIDIA_API_KEY
  - Internet connection

Returns:
  - Access status for each model
  - Sample outputs for verification
  - Recommendations for integration
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


def test_embedding_model(api_key: str) -> Dict[str, Any]:
    """Test access to nvidia/nv-embed-v1 embedding model."""
    print("Testing nvidia/nv-embed-v1 embedding model...")

    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "NVIDIA-Build-Test/1.0",
        "accept": "application/json",
    }

    payload = {
        "model": "nvidia/nv-embed-v1",
        "input": ["This is a test for pharmaceutical research applications"],
        "input_type": "query",
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
            "sample_output": response.json() if response.status_code == 200 else None,
        }
    except Exception as e:
        return {"success": False, "status_code": None, "response_time_ms": None, "error": str(e), "sample_output": None}


def test_llm_model(api_key: str) -> Dict[str, Any]:
    """Test access to meta/llama-3.1-8b-instruct language model."""
    print("Testing meta/llama-3.1-8b-instruct language model...")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "NVIDIA-Build-Test/1.0",
        "accept": "application/json",
    }

    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": "What are the key considerations for pharmaceutical drug interactions?"}
        ],
        "max_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.7,
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
            "sample_output": response.json() if response.status_code == 200 else None,
        }
    except Exception as e:
        return {"success": False, "status_code": None, "response_time_ms": None, "error": str(e), "sample_output": None}


def print_test_results(model_name: str, results: Dict[str, Any]) -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    if results["success"]:
        print("✅ ACCESS CONFIRMED")
        print(f"Response Time: {results['response_time_ms']}ms")
        if results["sample_output"]:
            if "data" in results["sample_output"]:
                # Embedding response
                print(f"Embedding Dimensions: {len(results['sample_output']['data'][0]['embedding'])}")
                print(f"Usage: {results['sample_output'].get('usage', 'N/A')}")
            elif "choices" in results["sample_output"]:
                # LLM response
                content = results["sample_output"]["choices"][0]["message"]["content"]
                print(f"Sample Response: {content[:200]}...")
                print(f"Usage: {results['sample_output'].get('usage', 'N/A')}")
    else:
        print("❌ ACCESS FAILED")
        print(f"Status Code: {results['status_code']}")
        print(f"Error: {results['error']}")


def generate_recommendations(embedding_success: bool, llm_success: bool) -> None:
    """Generate recommendations based on test results."""
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    if embedding_success and llm_success:
        print("✅ Both models are accessible with your NVIDIA Build API key!")
        print("\nNext Steps:")
        print("1. Your current NeMo Retriever models are superior for pharmaceutical use")
        print("2. Consider using nvidia/nv-embed-v1 as a cost-effective fallback")
        print("3. meta/llama-3.1-8b-instruct is already configured in your .env")
        print("4. Add NVIDIA Build endpoint support for flexible model switching")

    elif embedding_success:
        print("✅ Embedding model accessible, ❌ LLM model failed")
        print("\nRecommendations:")
        print("1. Use nvidia/nv-embed-v1 as fallback embedding model")
        print("2. Check LLM model name format or API key permissions")
        print("3. Your current meta/llama-3.1-8b-instruct may work on different endpoints")

    elif llm_success:
        print("✅ LLM model accessible, ❌ Embedding model failed")
        print("\nRecommendations:")
        print("1. Use meta/llama-3.1-8b-instruct as configured")
        print("2. Check embedding model name format or API key permissions")
        print("3. Your current nv-embedqa-e5-v5 is superior for pharmaceutical use")

    else:
        print("❌ Both models failed to access")
        print("\nTroubleshooting:")
        print("1. Verify your NVIDIA_API_KEY is valid and active")
        print("2. Check if models are enabled for your account")
        print("3. Ensure you have NVIDIA Build platform access")
        print("4. Your current NeMo Retriever setup may be using different endpoints")


def main():
    print("NVIDIA Build API Access Verification Test")
    print(f"{'='*60}")

    # Get API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        print("Please set your NVIDIA API key in .env file")
        return 1

    print(f"API Key: {api_key[:10]}...")

    # Test embedding model
    embedding_results = test_embedding_model(api_key)
    print_test_results("nvidia/nv-embed-v1", embedding_results)

    # Test LLM model
    llm_results = test_llm_model(api_key)
    print_test_results("meta/llama-3.1-8b-instruct", llm_results)

    # Generate recommendations
    generate_recommendations(embedding_results["success"], llm_results["success"])

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"nvidia/nv-embed-v1: {'✅ ACCESSIBLE' if embedding_results['success'] else '❌ FAILED'}")
    print(f"meta/llama-3.1-8b-instruct: {'✅ ACCESSIBLE' if llm_results['success'] else '❌ FAILED'}")

    return 0 if (embedding_results["success"] or llm_results["success"]) else 1


if __name__ == "__main__":
    sys.exit(main())
