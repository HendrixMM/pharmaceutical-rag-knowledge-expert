"""
Enhanced Prompt Generation with MCP Integration

This module provides prompt generation utilities that integrate with
Microsoft Learn MCP server for up-to-date documentation.
"""
import logging
from typing import Optional

from mcp_client import create_mcp_client
from mcp_client import NeMoMCPClient

logger = logging.getLogger(__name__)


class MCPPromptGenerator:
    """Prompt generator with Microsoft Learn MCP integration."""

    def __init__(self, mcp_client: Optional[NeMoMCPClient] = None):
        """
        Initialize the prompt generator.

        Args:
            mcp_client: Optional MCP client instance. If None, creates a new one.
        """
        self.mcp_client = mcp_client or create_mcp_client()

    def build_migration_prompt(
        self, migration_type: str = "nemo_retriever", specific_requirements: Optional[str] = None
    ) -> str:
        """
        Build a comprehensive migration prompt with live documentation.

        Args:
            migration_type: Type of migration (e.g., "nemo_retriever", "langchain_faiss")
            specific_requirements: Specific migration requirements

        Returns:
            Enhanced prompt with MCP context
        """
        base_prompt = self._get_base_migration_prompt(migration_type, specific_requirements)
        context_query = self._get_context_query(migration_type)

        return self.mcp_client.build_migration_prompt(base_prompt, context_query)

    def _get_base_migration_prompt(self, migration_type: str, specific_requirements: Optional[str]) -> str:
        """Generate base migration prompt based on type."""

        prompts = {
            "nemo_retriever": """
You are migrating to NVIDIA NeMo Retriever. Based on the latest Microsoft Learn documentation, generate code that:

1. **Replaces existing embedding models** with NVIDIA NIM embedding endpoints
2. **Updates vector database integration** to work with NeMo Retriever's optimized pipelines
3. **Implements proper authentication** for NVIDIA API access
4. **Optimizes retrieval performance** using NeMo's advanced features
5. **Maintains backward compatibility** where possible

Key migration requirements:
- Use latest NeMo Retriever APIs and best practices
- Implement proper error handling and fallbacks
- Include comprehensive logging and monitoring
- Follow NVIDIA's recommended patterns and configurations

""",
            "langchain_faiss": """
You are migrating LangChain + FAISS implementation to NVIDIA NeMo Retriever. Based on the latest documentation, generate code that:

1. **Replaces LangChain embeddings** with NeMo Retriever embedding NIMs
2. **Migrates FAISS vector store** to NeMo's optimized vector database
3. **Updates retrieval chains** to use NeMo's retrieval pipelines
4. **Preserves existing interfaces** where possible for minimal disruption
5. **Enhances performance** with NeMo's GPU acceleration

Migration considerations:
- Maintain compatibility with existing LangChain workflows
- Implement gradual migration strategy
- Add performance benchmarking
- Include comprehensive testing

""",
            "embedding_optimization": """
You are optimizing embedding performance using NVIDIA NeMo Retriever. Based on the latest documentation, generate code that:

1. **Implements batch embedding** for improved throughput
2. **Uses GPU acceleration** with NVIDIA NIMs
3. **Optimizes vector indexing** with NeMo's advanced algorithms
4. **Implements caching strategies** for frequently accessed embeddings
5. **Monitors performance metrics** and provides optimization insights

Performance optimization goals:
- Maximize embedding throughput
- Minimize latency for retrieval
- Optimize memory usage
- Scale efficiently with data growth

""",
        }

        base = prompts.get(migration_type, prompts["nemo_retriever"])

        if specific_requirements:
            base += f"\n**Additional Requirements:**\n{specific_requirements}\n"

        return base

    def _get_context_query(self, migration_type: str) -> str:
        """Get appropriate context query for migration type."""

        queries = {
            "nemo_retriever": "nvidia nemo retriever embedding nim api migration",
            "langchain_faiss": "nemo retriever langchain faiss migration integration",
            "embedding_optimization": "nvidia nim embedding performance optimization gpu",
        }

        return queries.get(migration_type, "nvidia nemo retriever")

    def build_troubleshooting_prompt(self, error_description: str) -> str:
        """
        Build a troubleshooting prompt with relevant documentation.

        Args:
            error_description: Description of the error or issue

        Returns:
            Troubleshooting prompt with MCP context
        """
        base_prompt = f"""
Based on the latest NVIDIA NeMo Retriever documentation, help troubleshoot this issue:

**Error/Issue Description:**
{error_description}

Please provide:
1. **Root cause analysis** based on current NeMo Retriever behavior
2. **Step-by-step solution** with code examples
3. **Prevention strategies** to avoid similar issues
4. **Alternative approaches** if the primary solution doesn't work
5. **Links to relevant documentation** for further reading

Ensure all solutions are based on the most current NeMo Retriever documentation and best practices.
"""

        context_query = f"nvidia nemo retriever troubleshooting {error_description[:50]}"
        return self.mcp_client.build_migration_prompt(base_prompt, context_query)

    def build_feature_implementation_prompt(self, feature_name: str, requirements: str) -> str:
        """
        Build a feature implementation prompt with latest documentation.

        Args:
            feature_name: Name of the feature to implement
            requirements: Detailed requirements for the feature

        Returns:
            Implementation prompt with MCP context
        """
        base_prompt = f"""
Implement the "{feature_name}" feature using NVIDIA NeMo Retriever. Based on the latest documentation:

**Feature Requirements:**
{requirements}

Please provide:
1. **Architecture design** following NeMo Retriever best practices
2. **Complete implementation** with proper error handling
3. **Configuration examples** for different use cases
4. **Testing strategy** including unit and integration tests
5. **Performance considerations** and optimization tips
6. **Documentation** for the implemented feature

Ensure the implementation follows current NVIDIA guidelines and leverages the latest NeMo Retriever capabilities.
"""

        context_query = f"nvidia nemo retriever {feature_name} implementation"
        return self.mcp_client.build_migration_prompt(base_prompt, context_query)

    def get_latest_nemo_context(self) -> str:
        """
        Get the latest NeMo Retriever context for general use.

        Returns:
            Formatted context with latest NeMo documentation
        """
        return self.mcp_client.get_nemo_retriever_context()


# Convenience functions for common use cases
def generate_migration_prompt(migration_type: str = "nemo_retriever", requirements: Optional[str] = None) -> str:
    """
    Quick function to generate migration prompt.

    Args:
        migration_type: Type of migration
        requirements: Specific requirements

    Returns:
        Enhanced migration prompt
    """
    generator = MCPPromptGenerator()
    return generator.build_migration_prompt(migration_type, requirements)


def generate_troubleshooting_prompt(error_description: str) -> str:
    """
    Quick function to generate troubleshooting prompt.

    Args:
        error_description: Description of the error

    Returns:
        Enhanced troubleshooting prompt
    """
    generator = MCPPromptGenerator()
    return generator.build_troubleshooting_prompt(error_description)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    generator = MCPPromptGenerator()

    # Generate a migration prompt
    migration_prompt = generator.build_migration_prompt(
        "nemo_retriever", "Need to migrate a 10M document corpus with custom preprocessing"
    )

    print("Generated Migration Prompt:")
    print("=" * 50)
    print(migration_prompt[:500] + "...")

    # Get latest NeMo context
    context = generator.get_latest_nemo_context()
    print("\nLatest NeMo Context:")
    print("=" * 50)
    print(context[:300] + "...")
