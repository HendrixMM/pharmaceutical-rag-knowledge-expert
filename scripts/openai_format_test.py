"""
OpenAI-Compatible Format Test for NVIDIA Build Platform

Tests using OpenAI SDK format which is the recommended approach
for NVIDIA Build platform integration.

Usage:
  python scripts/openai_format_test.py
"""
import os
import sys
from pathlib import Path

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

try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: openai package not installed")
    print("Install with: pip install openai")
    sys.exit(1)


def test_embedding_with_openai_format():
    """Test embedding using OpenAI format with NVIDIA Build."""
    print("🧪 Testing embedding with OpenAI format...")

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found")
        return False

    try:
        client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")

        response = client.embeddings.create(
            input=["Test pharmaceutical research applications"], model="nvidia/nv-embed-v1"
        )

        print("✅ Embedding successful!")
        print(f"   Dimensions: {len(response.data[0].embedding)}")
        print(f"   Usage: {response.usage}")
        return True

    except Exception as e:
        print(f"❌ Embedding failed: {str(e)}")
        return False


def test_chat_with_openai_format():
    """Test chat completions using OpenAI format."""
    print("\n🧪 Testing chat completions with OpenAI format...")

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found")
        return False

    try:
        client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")

        response = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "What are pharmaceutical drug interactions? Answer in 2 sentences."}],
            max_tokens=100,
            temperature=0.1,
        )

        print("✅ Chat completion successful!")
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Usage: {response.usage}")
        return True

    except Exception as e:
        print(f"❌ Chat completion failed: {str(e)}")
        return False


def main():
    print("NVIDIA Build Platform - OpenAI Format Compatibility Test")
    print("=" * 60)

    embedding_success = test_embedding_with_openai_format()
    chat_success = test_chat_with_openai_format()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Embedding Access: {'✅ SUCCESS' if embedding_success else '❌ FAILED'}")
    print(f"Chat Access: {'✅ SUCCESS' if chat_success else '❌ FAILED'}")

    if embedding_success or chat_success:
        print("\n🎯 RESULT: Your API key DOES work with NVIDIA Build platform!")
        print("💡 Use OpenAI SDK format for best compatibility")

        if embedding_success:
            print("📊 nvidia/nv-embed-v1 is accessible for embeddings")
        if chat_success:
            print("💬 meta/llama-3.1-8b-instruct is accessible for chat")

        print("\n⚙️  Integration recommendation:")
        print("   Add OpenAI-compatible client to your RAG system as fallback")

    else:
        print("\n❌ API access issues persist")
        print("🔧 Contact NVIDIA support with this test output")

    return 0 if (embedding_success or chat_success) else 1


if __name__ == "__main__":
    sys.exit(main())
