"""
Microsoft Learn MCP Client for NeMo Retriever Documentation Integration

This module provides an MCP client to fetch up-to-date documentation
from Microsoft Learn for NVIDIA NeMo Retriever and related technologies.

Note: This implementation provides a framework for MCP integration.
Since Microsoft Learn doesn't currently expose an MCP endpoint,
this includes fallback mechanisms for demonstration and future use.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mcp_use import MCPClient
except ImportError:
    raise ImportError("mcp-use not installed. Run: pip install mcp-use")

logger = logging.getLogger(__name__)


class NeMoMCPClient:
    """MCP Client for fetching NeMo Retriever documentation from Microsoft Learn."""

    def __init__(self, config_path: str = "mcp_config.json", enable_fallback: bool = True):
        """
        Initialize the MCP client with configuration.

        Args:
            config_path: Path to the MCP configuration file
            enable_fallback: Whether to enable fallback mechanisms when MCP servers are unavailable
        """
        self.config_path = Path(config_path)
        self.mcp_client = None
        self.sessions = {}
        self.enable_fallback = enable_fallback
        self._load_config()

    def _load_config(self):
        """Load MCP configuration from file."""
        if not self.config_path.exists():
            if self.enable_fallback:
                logger.warning(f"MCP config file not found: {self.config_path}, using fallback mode")
                self.mcp_client = None
                return
            else:
                raise FileNotFoundError(f"MCP config file not found: {self.config_path}")

        try:
            self.mcp_client = MCPClient.from_config_file(str(self.config_path))
            logger.info(f"Loaded MCP configuration from {self.config_path}")
        except Exception as e:
            if self.enable_fallback:
                logger.warning(f"Failed to load MCP configuration, using fallback: {e}")
                self.mcp_client = None
            else:
                logger.error(f"Failed to load MCP configuration: {e}")
                raise

    async def register_servers_async(self) -> bool:
        """
        Register all configured MCP servers asynchronously.

        Returns:
            True if successful, False otherwise
        """
        if not self.mcp_client:
            logger.warning("No MCP client available, skipping server registration")
            return False

        try:
            # Get server names from config
            self.mcp_client.get_server_names()

            # Create sessions for all servers
            sessions = await self.mcp_client.create_all_sessions()
            self.sessions.update(sessions)

            logger.info(f"Successfully registered {len(sessions)} MCP servers: {list(sessions.keys())}")
            return True
        except Exception as e:
            logger.error(f"Failed to register MCP servers: {e}")
            return False

    def register_servers(self) -> bool:
        """
        Register all configured MCP servers (synchronous wrapper).

        Returns:
            True if successful, False otherwise
        """
        try:
            return asyncio.run(self.register_servers_async())
        except Exception as e:
            logger.error(f"Failed to register MCP servers: {e}")
            return False

    async def query_docs_async(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query Microsoft Learn for documentation asynchronously.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of documentation results
        """
        if not self.sessions:
            logger.warning("No active MCP sessions, using fallback documentation")
            return self._get_fallback_docs(query, max_results)

        try:
            results = []
            # Try to use available sessions to query documentation
            for server_name, session in self.sessions.items():
                if "microsoft-learn" in server_name or "learn" in server_name:
                    try:
                        # List available tools and resources
                        tools = await session.list_tools()
                        resources = await session.list_resources()

                        # If there are search tools, use them
                        for tool in tools:
                            if "search" in tool.name.lower() or "query" in tool.name.lower():
                                result = await session.call_tool(tool.name, {"query": query, "limit": max_results})
                                if result and hasattr(result, "content"):
                                    # Parse the result and add to results
                                    doc_data = self._parse_tool_result(result, query)
                                    results.extend(doc_data[:max_results])
                                    break

                        # If there are relevant resources, read them
                        for resource in resources[:max_results]:
                            if query.lower() in resource.uri.lower() or "nemo" in resource.uri.lower():
                                resource_content = await session.read_resource(resource.uri)
                                results.append(
                                    {
                                        "title": f"Resource: {resource.name}",
                                        "url": str(resource.uri),
                                        "content": str(resource_content.contents),
                                        "relevance_score": 0.8,
                                    }
                                )

                    except Exception as e:
                        logger.warning(f"Failed to query session {server_name}: {e}")
                        continue

            if results:
                logger.info(f"Retrieved {len(results)} documents for query: {query}")
                return results[:max_results]
            else:
                logger.info("No results from MCP servers, using fallback")
                return self._get_fallback_docs(query, max_results)

        except Exception as e:
            logger.error(f"Failed to query docs via MCP: {e}")
            return self._get_fallback_docs(query, max_results)

    def query_docs(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query Microsoft Learn for documentation (synchronous wrapper).

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of documentation results
        """
        try:
            return asyncio.run(self.query_docs_async(query, max_results))
        except Exception as e:
            logger.error(f"Failed to query docs: {e}")
            return self._get_fallback_docs(query, max_results)

    def _parse_tool_result(self, result, query: str) -> List[Dict[str, Any]]:
        """Parse MCP tool result into standardized format."""
        docs = []
        try:
            if hasattr(result, "content") and isinstance(result.content, list):
                for item in result.content:
                    if hasattr(item, "text"):
                        # Try to parse as JSON or extract structured data
                        try:
                            data = json.loads(item.text)
                            if isinstance(data, list):
                                for doc in data:
                                    docs.append(
                                        {
                                            "title": doc.get("title", "Unknown Title"),
                                            "url": doc.get("url", "Unknown URL"),
                                            "content": doc.get("content", doc.get("summary", "")),
                                            "relevance_score": doc.get("score", 0.7),
                                        }
                                    )
                        except json.JSONDecodeError:
                            # Treat as plain text result
                            docs.append(
                                {
                                    "title": f"Search result for: {query}",
                                    "url": "https://learn.microsoft.com",
                                    "content": item.text[:500],
                                    "relevance_score": 0.6,
                                }
                            )
        except Exception as e:
            logger.warning(f"Failed to parse tool result: {e}")

        return docs

    def _get_fallback_docs(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Get fallback documentation when MCP is not available.

        This simulates what would be returned from Microsoft Learn
        for NeMo Retriever queries.
        """
        query_lower = query.lower()

        # Simulated documentation entries based on common queries
        fallback_docs = []

        if "nemo" in query_lower and "retriever" in query_lower:
            fallback_docs.extend(
                [
                    {
                        "title": "NVIDIA NeMo Retriever Overview",
                        "url": "https://docs.nvidia.com/nemo/retriever/overview.html",
                        "content": "NVIDIA NeMo Retriever is a cloud-native solution for building and deploying retrieval-augmented generation (RAG) applications. It provides optimized embedding models and vector search capabilities.",
                        "relevance_score": 0.9,
                    },
                    {
                        "title": "NeMo Retriever Embedding NIMs",
                        "url": "https://docs.nvidia.com/nemo/retriever/embedding-nims.html",
                        "content": "NeMo Retriever includes pre-built NVIDIA Inference Microservices (NIMs) for embedding generation, supporting various embedding models optimized for different use cases.",
                        "relevance_score": 0.85,
                    },
                ]
            )

        if "langchain" in query_lower:
            fallback_docs.append(
                {
                    "title": "Integrating NeMo Retriever with LangChain",
                    "url": "https://docs.nvidia.com/nemo/retriever/langchain-integration.html",
                    "content": "Learn how to integrate NVIDIA NeMo Retriever with LangChain frameworks for enhanced RAG applications. Includes examples for embedding and retrieval chain setup.",
                    "relevance_score": 0.8,
                }
            )

        if "embedding" in query_lower:
            fallback_docs.append(
                {
                    "title": "NeMo Retriever Embedding Models",
                    "url": "https://docs.nvidia.com/nemo/retriever/embedding-models.html",
                    "content": "Comprehensive guide to NVIDIA NeMo Retriever embedding models, including performance benchmarks and best practices for different document types.",
                    "relevance_score": 0.85,
                }
            )

        if "migration" in query_lower:
            fallback_docs.append(
                {
                    "title": "Migrating to NeMo Retriever",
                    "url": "https://docs.nvidia.com/nemo/retriever/migration-guide.html",
                    "content": "Step-by-step guide for migrating existing RAG systems to NVIDIA NeMo Retriever, including code examples and best practices.",
                    "relevance_score": 0.9,
                }
            )

        if "performance" in query_lower or "optimization" in query_lower:
            fallback_docs.append(
                {
                    "title": "NeMo Retriever Performance Optimization",
                    "url": "https://docs.nvidia.com/nemo/retriever/performance-optimization.html",
                    "content": "Best practices for optimizing NVIDIA NeMo Retriever performance, including batch processing, GPU utilization, and caching strategies.",
                    "relevance_score": 0.8,
                }
            )

        if "troubleshooting" in query_lower or "error" in query_lower:
            fallback_docs.append(
                {
                    "title": "NeMo Retriever Troubleshooting Guide",
                    "url": "https://docs.nvidia.com/nemo/retriever/troubleshooting.html",
                    "content": "Common issues and solutions when working with NVIDIA NeMo Retriever, including CUDA memory errors, authentication issues, and performance problems.",
                    "relevance_score": 0.85,
                }
            )

        # If no specific matches, provide general docs
        if not fallback_docs:
            fallback_docs = [
                {
                    "title": "NVIDIA NeMo Retriever Documentation",
                    "url": "https://docs.nvidia.com/nemo/retriever/",
                    "content": "Complete documentation for NVIDIA NeMo Retriever, including API references, tutorials, and best practices for building RAG applications.",
                    "relevance_score": 0.7,
                }
            ]

        logger.info(f"Using fallback documentation: {len(fallback_docs[:max_results])} docs for query '{query}'")
        return fallback_docs[:max_results]

    def build_migration_prompt(self, base_prompt: str, context_query: Optional[str] = None) -> str:
        """
        Build a migration prompt with MCP context.

        Args:
            base_prompt: The base prompt content
            context_query: Optional query to fetch relevant context

        Returns:
            Enhanced prompt with MCP context
        """
        mcp_header = "<<use_mcp microsoft-learn>>\n"

        if context_query:
            # Fetch relevant documentation
            docs = self.query_docs(context_query, max_results=3)
            if docs:
                context_section = "\n## Relevant Documentation Context:\n"
                for doc in docs:
                    context_section += f"- **{doc['title']}**: {doc['url']}\n"
                    if doc["content"]:
                        context_section += f"  {doc['content'][:200]}...\n"
                context_section += "\n"
                return mcp_header + context_section + base_prompt

        return mcp_header + base_prompt

    def get_nemo_retriever_context(self) -> str:
        """
        Get specific NeMo Retriever documentation context.

        Returns:
            Formatted context string with latest NeMo Retriever docs
        """
        queries = [
            "nvidia nemo retriever embedding",
            "nemo retriever langchain integration",
            "nvidia nim embedding models",
            "faiss vector database nvidia",
        ]

        all_docs = []
        for query in queries:
            docs = self.query_docs(query, max_results=2)
            all_docs.extend(docs)

        if not all_docs:
            return "No NeMo Retriever documentation found."

        context = "## Latest NeMo Retriever Documentation:\n\n"
        for doc in all_docs:
            context += f"### {doc['title']}\n"
            context += f"URL: {doc['url']}\n"
            if doc["content"]:
                context += f"Summary: {doc['content'][:300]}...\n"
            context += "\n"

        return context


def create_mcp_client() -> NeMoMCPClient:
    """
    Factory function to create and initialize MCP client.

    Returns:
        Initialized NeMoMCPClient instance
    """
    client = NeMoMCPClient()
    if client.register_servers():
        logger.info("MCP client ready for use")
    else:
        logger.warning("MCP client created but server registration failed")
    return client


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    try:
        client = create_mcp_client()

        # Test query
        docs = client.query_docs("nvidia nemo retriever embedding")
        print(f"Found {len(docs)} documents")

        for doc in docs[:3]:
            print(f"- {doc['title']}: {doc['url']}")

    except Exception as e:
        print(f"Error: {e}")
