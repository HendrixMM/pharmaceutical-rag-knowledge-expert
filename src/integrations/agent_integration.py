"""
Agent Integration Module for MCP-Enhanced RAG System

This module provides integration between your existing Claude-Code agent
and the Microsoft Learn MCP server for up-to-date documentation.
"""
import logging
from typing import Any, Dict, List, Optional

from mcp_client import NeMoMCPClient, create_mcp_client

from prompt_generator import MCPPromptGenerator

logger = logging.getLogger(__name__)


class MCPEnhancedAgent:
    """
    Enhanced agent that integrates MCP documentation fetching
    with your existing Claude-Code workflow.
    """

    def __init__(self, mcp_client: Optional[NeMoMCPClient] = None, enable_auto_context: bool = True):
        """
        Initialize the MCP-enhanced agent.

        Args:
            mcp_client: Optional MCP client instance
            enable_auto_context: Whether to automatically fetch context for queries
        """
        self.mcp_client = mcp_client or create_mcp_client()
        self.prompt_generator = MCPPromptGenerator(self.mcp_client)
        self.enable_auto_context = enable_auto_context

        logger.info("MCP-Enhanced Agent initialized")

    def ask_with_context(self, query: str, context_type: str = "auto", max_context_docs: int = 3) -> str:
        """
        Process a query with automatically fetched context.

        Args:
            query: The user query or task
            context_type: Type of context to fetch ("auto", "nemo", "langchain", "troubleshooting")
            max_context_docs: Maximum number of context documents to include

        Returns:
            Enhanced prompt with MCP context
        """
        if not self.enable_auto_context:
            return query

        try:
            # Determine context query based on type
            if context_type == "auto":
                context_query = self._extract_context_query(query)
            elif context_type == "nemo":
                context_query = "nvidia nemo retriever"
            elif context_type == "langchain":
                context_query = "nemo retriever langchain integration"
            elif context_type == "troubleshooting":
                context_query = f"nvidia nemo retriever troubleshooting {query[:50]}"
            else:
                context_query = context_type

            # Fetch relevant documentation
            docs = self.mcp_client.query_docs(context_query, max_results=max_context_docs)

            if docs:
                enhanced_query = self._build_enhanced_query(query, docs)
                logger.info(f"Enhanced query with {len(docs)} context documents")
                return enhanced_query
            else:
                logger.warning("No context documents found, using original query")
                return query

        except Exception as e:
            logger.error(f"Failed to enhance query with context: {e}")
            return query

    def _extract_context_query(self, query: str) -> str:
        """Extract appropriate context query from user input."""
        query_lower = query.lower()

        # Keywords that suggest specific context needs
        if any(word in query_lower for word in ["migrate", "migration", "convert"]):
            return "nvidia nemo retriever migration"
        elif any(word in query_lower for word in ["error", "fail", "issue", "problem", "debug"]):
            return "nvidia nemo retriever troubleshooting"
        elif any(word in query_lower for word in ["performance", "optimize", "speed", "latency"]):
            return "nvidia nemo retriever performance optimization"
        elif any(word in query_lower for word in ["langchain", "faiss", "vector store"]):
            return "nemo retriever langchain faiss integration"
        elif any(word in query_lower for word in ["embedding", "embed", "encode"]):
            return "nvidia nim embedding models"
        else:
            return "nvidia nemo retriever"

    def _build_enhanced_query(self, original_query: str, docs: List[Dict[str, Any]]) -> str:
        """Build enhanced query with context documents."""
        mcp_header = "<<use_mcp microsoft-learn>>\n\n"

        context_section = "## Relevant Documentation Context:\n\n"
        for i, doc in enumerate(docs, 1):
            context_section += f"{i}. **{doc['title']}**\n"
            context_section += f"   URL: {doc['url']}\n"
            if doc.get("content"):
                # Include a snippet of content
                content_snippet = doc["content"][:250].strip()
                context_section += f"   Summary: {content_snippet}...\n"
            context_section += "\n"

        user_query_section = f"## User Query:\n{original_query}\n\n"

        instruction_section = """## Instructions:
Use the latest documentation context above to provide accurate, up-to-date information.
Reference specific documentation URLs when relevant.
Ensure all code examples and recommendations align with current NVIDIA NeMo Retriever best practices.

"""

        return mcp_header + context_section + user_query_section + instruction_section

    def generate_migration_code(self, source_system: str, target_requirements: Optional[str] = None) -> str:
        """
        Generate migration code with up-to-date documentation.

        Args:
            source_system: Current system being migrated from
            target_requirements: Specific requirements for the target system

        Returns:
            Migration prompt with current documentation context
        """
        migration_type = self._determine_migration_type(source_system)
        return self.prompt_generator.build_migration_prompt(migration_type, target_requirements)

    def _determine_migration_type(self, source_system: str) -> str:
        """Determine migration type based on source system."""
        source_lower = source_system.lower()

        if any(word in source_lower for word in ["langchain", "faiss"]):
            return "langchain_faiss"
        elif any(word in source_lower for word in ["embedding", "vector"]):
            return "embedding_optimization"
        else:
            return "nemo_retriever"

    def debug_with_context(self, error_description: str) -> str:
        """
        Generate debugging assistance with current documentation.

        Args:
            error_description: Description of the error or issue

        Returns:
            Debugging prompt with relevant context
        """
        return self.prompt_generator.build_troubleshooting_prompt(error_description)

    def implement_feature_with_context(self, feature_name: str, requirements: str) -> str:
        """
        Generate feature implementation with current documentation.

        Args:
            feature_name: Name of the feature to implement
            requirements: Feature requirements

        Returns:
            Implementation prompt with relevant context
        """
        return self.prompt_generator.build_feature_implementation_prompt(feature_name, requirements)

    def get_latest_nemo_updates(self) -> str:
        """Get the latest NeMo Retriever updates and documentation."""
        return self.mcp_client.get_nemo_retriever_context()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the MCP integration.

        Returns:
            Health check results
        """
        results = {
            "mcp_client_active": False,
            "server_registered": False,
            "documentation_accessible": False,
            "last_error": None,
        }

        try:
            # Check if MCP client is active
            if self.mcp_client:
                results["mcp_client_active"] = True

                # Test server registration by attempting a simple query
                docs = self.mcp_client.query_docs("nvidia nemo retriever", max_results=1)
                if docs:
                    results["documentation_accessible"] = True
                    results["server_registered"] = True
                    logger.info("MCP health check passed")
                else:
                    logger.warning("MCP client active but no docs retrieved")

        except Exception as e:
            results["last_error"] = str(e)
            logger.error(f"MCP health check failed: {e}")

        return results


# Convenience functions for easy integration
def create_enhanced_agent() -> MCPEnhancedAgent:
    """
    Factory function to create MCP-enhanced agent.

    Returns:
        Configured MCPEnhancedAgent instance
    """
    return MCPEnhancedAgent()


def ask_with_live_docs(query: str, context_type: str = "auto") -> str:
    """
    Quick function to ask a question with live documentation context.

    Args:
        query: The question or task
        context_type: Type of context to include

    Returns:
        Enhanced query with documentation context
    """
    agent = create_enhanced_agent()
    return agent.ask_with_context(query, context_type)


def generate_migration_with_docs(source_system: str, requirements: Optional[str] = None) -> str:
    """
    Quick function to generate migration code with live docs.

    Args:
        source_system: System being migrated from
        requirements: Migration requirements

    Returns:
        Migration prompt with current documentation
    """
    agent = create_enhanced_agent()
    return agent.generate_migration_code(source_system, requirements)


# Example usage and integration patterns
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Create enhanced agent
    agent = create_enhanced_agent()

    # Example 2: Health check
    health = agent.health_check()
    print(f"MCP Health: {health}")

    # Example 3: Ask with context
    enhanced_query = agent.ask_with_context(
        "How do I optimize embedding performance for large document collections?", context_type="auto"
    )
    print(f"Enhanced Query Length: {len(enhanced_query)} characters")

    # Example 4: Generate migration code
    migration_prompt = agent.generate_migration_code(
        "LangChain with FAISS", "Need to handle 10M documents with custom preprocessing"
    )
    print(f"Migration Prompt Length: {len(migration_prompt)} characters")

    # Example 5: Debug with context
    debug_prompt = agent.debug_with_context("Getting CUDA out of memory errors when processing large batches")
    print(f"Debug Prompt Length: {len(debug_prompt)} characters")
