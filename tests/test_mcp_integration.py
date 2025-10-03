"""
Test script for Microsoft Learn MCP integration

This script verifies that the MCP client is working correctly
and can fetch up-to-date NeMo Retriever documentation.
"""
import logging
import sys
import time

from mcp_client import NeMoMCPClient, create_mcp_client

from prompt_generator import MCPPromptGenerator, generate_migration_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_mcp_client_creation():
    """Test basic MCP client creation and initialization."""
    print("\n" + "=" * 60)
    print("Testing MCP Client Creation")
    print("=" * 60)

    try:
        client = create_mcp_client()
        print("✅ MCP client created successfully")
        return client
    except Exception as e:
        print(f"❌ Failed to create MCP client: {e}")
        return None


def test_documentation_query(client: NeMoMCPClient):
    """Test querying Microsoft Learn for NeMo documentation."""
    print("\n" + "=" * 60)
    print("Testing Documentation Query")
    print("=" * 60)

    test_queries = [
        "nvidia nemo retriever embedding",
        "nemo retriever langchain integration",
        "nvidia nim embedding models",
        "faiss vector database nvidia",
    ]

    all_results = []
    for query in test_queries:
        print(f"\n🔍 Querying: '{query}'")
        try:
            docs = client.query_docs(query, max_results=3)
            print(f"📄 Found {len(docs)} documents")

            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc['title']}")
                print(f"     URL: {doc['url']}")
                if doc.get("relevance_score"):
                    print(f"     Score: {doc['relevance_score']:.2f}")
                print()

            all_results.extend(docs)

        except Exception as e:
            print(f"❌ Query failed: {e}")

    if all_results:
        print(f"✅ Successfully retrieved {len(all_results)} total documents")
        return True
    else:
        print("❌ No documents retrieved")
        return False


def test_nemo_context_generation(client: NeMoMCPClient):
    """Test NeMo Retriever context generation."""
    print("\n" + "=" * 60)
    print("Testing NeMo Context Generation")
    print("=" * 60)

    try:
        context = client.get_nemo_retriever_context()
        print(f"📝 Generated context ({len(context)} characters):")
        print("-" * 40)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 40)
        print("✅ Context generation successful")
        return True
    except Exception as e:
        print(f"❌ Context generation failed: {e}")
        return False


def test_prompt_generation():
    """Test enhanced prompt generation with MCP integration."""
    print("\n" + "=" * 60)
    print("Testing Prompt Generation")
    print("=" * 60)

    try:
        generator = MCPPromptGenerator()

        # Test migration prompt
        print("🔧 Generating migration prompt...")
        migration_prompt = generator.build_migration_prompt(
            "nemo_retriever", "Migrate 10M document corpus with custom preprocessing pipeline"
        )

        print(f"📝 Generated migration prompt ({len(migration_prompt)} characters):")
        print("-" * 40)
        print(migration_prompt[:600] + "..." if len(migration_prompt) > 600 else migration_prompt)
        print("-" * 40)

        # Test troubleshooting prompt
        print("\n🔧 Generating troubleshooting prompt...")
        troubleshooting_prompt = generator.build_troubleshooting_prompt(
            "CUDA out of memory error when processing large batches"
        )

        print(f"📝 Generated troubleshooting prompt ({len(troubleshooting_prompt)} characters):")
        print("-" * 40)
        print(troubleshooting_prompt[:400] + "..." if len(troubleshooting_prompt) > 400 else troubleshooting_prompt)
        print("-" * 40)

        print("✅ Prompt generation successful")
        return True

    except Exception as e:
        print(f"❌ Prompt generation failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions for quick prompt generation."""
    print("\n" + "=" * 60)
    print("Testing Convenience Functions")
    print("=" * 60)

    try:
        # Test quick migration prompt
        print("🚀 Testing quick migration prompt...")
        quick_prompt = generate_migration_prompt(
            "langchain_faiss", "Need to maintain existing API while improving performance"
        )

        print(f"📝 Quick prompt generated ({len(quick_prompt)} characters)")
        print(f"✅ Contains MCP header: {'<<use_mcp microsoft-learn>>' in quick_prompt}")

        print("✅ Convenience functions working")
        return True

    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
        return False


def run_performance_test(client: NeMoMCPClient):
    """Run basic performance tests."""
    print("\n" + "=" * 60)
    print("Running Performance Tests")
    print("=" * 60)

    try:
        queries = ["nvidia nemo retriever", "embedding models nvidia", "vector database integration"]

        total_time = 0
        total_docs = 0

        for query in queries:
            start_time = time.time()
            docs = client.query_docs(query, max_results=2)
            end_time = time.time()

            query_time = end_time - start_time
            total_time += query_time
            total_docs += len(docs)

            print(f"📊 Query '{query}': {len(docs)} docs in {query_time:.2f}s")

        avg_time = total_time / len(queries)
        print(f"\n📈 Performance Summary:")
        print(f"   Average query time: {avg_time:.2f}s")
        print(f"   Total documents: {total_docs}")
        print(f"   Docs per second: {total_docs/total_time:.2f}")

        if avg_time < 5.0:  # Reasonable threshold
            print("✅ Performance acceptable")
            return True
        else:
            print("⚠️  Performance may need optimization")
            return False

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Microsoft Learn MCP Integration Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Client creation
    client = test_mcp_client_creation()
    results.append(client is not None)

    if not client:
        print("❌ Cannot continue tests without working client")
        sys.exit(1)

    # Test 2: Documentation querying
    results.append(test_documentation_query(client))

    # Test 3: Context generation
    results.append(test_nemo_context_generation(client))

    # Test 4: Prompt generation
    results.append(test_prompt_generation())

    # Test 5: Convenience functions
    results.append(test_convenience_functions())

    # Test 6: Performance
    results.append(run_performance_test(client))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    test_names = [
        "MCP Client Creation",
        "Documentation Query",
        "Context Generation",
        "Prompt Generation",
        "Convenience Functions",
        "Performance Test",
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! MCP integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
