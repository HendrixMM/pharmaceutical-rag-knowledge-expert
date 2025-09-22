#!/usr/bin/env python3
"""
Usage Example: Microsoft Learn MCP Integration for NeMo Retriever

This script demonstrates how to use the MCP integration for enhanced
prompts with up-to-date NeMo Retriever documentation.
"""

import logging
from mcp_client import create_mcp_client
from prompt_generator import MCPPromptGenerator
from agent_integration import MCPEnhancedAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def example_1_basic_client_usage():
    """Example 1: Basic MCP client usage."""
    print("\n" + "="*60)
    print("Example 1: Basic MCP Client Usage")
    print("="*60)

    # Create MCP client
    client = create_mcp_client()

    # Query for documentation
    docs = client.query_docs("nvidia nemo retriever embedding", max_results=2)

    print(f"Found {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc['title']}")
        print(f"   URL: {doc['url']}")
        print(f"   Score: {doc['relevance_score']:.2f}")
        print()


def example_2_prompt_generation():
    """Example 2: Enhanced prompt generation."""
    print("\n" + "="*60)
    print("Example 2: Enhanced Prompt Generation")
    print("="*60)

    generator = MCPPromptGenerator()

    # Generate migration prompt
    migration_prompt = generator.build_migration_prompt(
        "nemo_retriever",
        "Migrate a large-scale document processing system with 10M documents"
    )

    print("Generated Migration Prompt:")
    print("-" * 40)
    print(migration_prompt[:500] + "...")
    print("-" * 40)


def example_3_troubleshooting_assistant():
    """Example 3: Troubleshooting with live docs."""
    print("\n" + "="*60)
    print("Example 3: Troubleshooting Assistant")
    print("="*60)

    generator = MCPPromptGenerator()

    # Generate troubleshooting prompt
    error_prompt = generator.build_troubleshooting_prompt(
        "Getting CUDA out of memory errors when processing batches of 1000 documents"
    )

    print("Generated Troubleshooting Prompt:")
    print("-" * 40)
    print(error_prompt[:400] + "...")
    print("-" * 40)


def example_4_enhanced_agent():
    """Example 4: Enhanced agent with automatic context."""
    print("\n" + "="*60)
    print("Example 4: Enhanced Agent with Auto Context")
    print("="*60)

    agent = MCPEnhancedAgent()

    # Test health check
    health = agent.health_check()
    print(f"Agent Health Check:")
    print(f"  MCP Client Active: {health['mcp_client_active']}")
    print(f"  Documentation Accessible: {health['documentation_accessible']}")

    # Ask with automatic context
    enhanced_query = agent.ask_with_context(
        "How do I optimize embedding performance for large document collections?",
        context_type="auto"
    )

    print(f"\nEnhanced Query (first 300 chars):")
    print(enhanced_query[:300] + "...")


def example_5_migration_workflow():
    """Example 5: Complete migration workflow."""
    print("\n" + "="*60)
    print("Example 5: Complete Migration Workflow")
    print("="*60)

    agent = MCPEnhancedAgent()

    # Generate migration code for LangChain + FAISS to NeMo
    migration_code = agent.generate_migration_code(
        "LangChain with FAISS vector store",
        "Need to maintain existing API while improving performance by 10x"
    )

    print("Migration Code Prompt:")
    print("-" * 40)
    print(migration_code[:400] + "...")
    print("-" * 40)


def example_6_feature_implementation():
    """Example 6: Feature implementation with docs."""
    print("\n" + "="*60)
    print("Example 6: Feature Implementation with Documentation")
    print("="*60)

    agent = MCPEnhancedAgent()

    # Implement a new feature with context
    feature_prompt = agent.implement_feature_with_context(
        "Batch Embedding Processor",
        "Create a system that can process 100k documents in batches with optimal GPU utilization"
    )

    print("Feature Implementation Prompt:")
    print("-" * 40)
    print(feature_prompt[:400] + "...")
    print("-" * 40)


def example_7_custom_configuration():
    """Example 7: Using custom MCP configuration."""
    print("\n" + "="*60)
    print("Example 7: Custom MCP Configuration")
    print("="*60)

    # This example shows how to extend the configuration
    # for additional MCP servers (when they become available)

    custom_config = {
        "servers": {
            "microsoft-learn": {
                "type": "sse",
                "url": "https://learn.microsoft.com/api/mcp"
            },
            "nvidia-docs": {
                "type": "stdio",
                "command": "nvidia-docs-mcp",
                "args": ["--model", "nemo-retriever"]
            },
            "github-docs": {
                "type": "sse",
                "url": "https://api.github.com/mcp",
                "headers": {
                    "Authorization": "Bearer ${GITHUB_TOKEN}"
                }
            }
        }
    }

    print("Example extended configuration:")
    import json
    print(json.dumps(custom_config, indent=2))


def example_8_performance_monitoring():
    """Example 8: Performance monitoring and optimization."""
    print("\n" + "="*60)
    print("Example 8: Performance Monitoring")
    print("="*60)

    import time

    client = create_mcp_client()

    # Test query performance
    queries = [
        "nvidia nemo retriever",
        "embedding optimization",
        "vector database performance",
        "langchain integration"
    ]

    total_time = 0
    total_docs = 0

    for query in queries:
        start_time = time.time()
        docs = client.query_docs(query, max_results=2)
        end_time = time.time()

        query_time = end_time - start_time
        total_time += query_time
        total_docs += len(docs)

        print(f"Query '{query}': {len(docs)} docs in {query_time:.3f}s")

    print(f"\nPerformance Summary:")
    print(f"  Average query time: {total_time/len(queries):.3f}s")
    print(f"  Total documents retrieved: {total_docs}")
    print(f"  Documents per second: {total_docs/total_time:.1f}")


def main():
    """Run all examples."""
    print("🚀 Microsoft Learn MCP Integration Examples")
    print("This demonstrates how to integrate MCP for up-to-date NeMo documentation")

    try:
        example_1_basic_client_usage()
        example_2_prompt_generation()
        example_3_troubleshooting_assistant()
        example_4_enhanced_agent()
        example_5_migration_workflow()
        example_6_feature_implementation()
        example_7_custom_configuration()
        example_8_performance_monitoring()

        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)

        print("\n📚 Next Steps:")
        print("1. Customize mcp_config.json for your specific MCP servers")
        print("2. Integrate with your existing Claude-Code workflow")
        print("3. Add authentication tokens for private MCP servers")
        print("4. Monitor performance and optimize batch sizes")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())