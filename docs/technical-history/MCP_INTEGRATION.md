# MCP Integrations (Claude)

This repo ships with turnkey wiring for Claude MCP servers to enrich your agent with live context. It includes:

- Microsoft Learn MCP (NVIDIA/NeMo docs)
- GitHub MCP (repositories, issues/PRs, Actions, Dependabot, etc.)

Below are quick-start guides for both.

---

## Microsoft Learn MCP Integration for NeMo Retriever

This implementation provides a comprehensive framework for integrating Microsoft Learn's Model Context Protocol (MCP) server with your Claude-Code agent to access up-to-date NVIDIA NeMo Retriever documentation.

## 🚀 Quick Start

### 1. Installation

The MCP integration requires `mcp-use`:

```bash
pip install mcp-use
```

### 2. Configuration

The system uses `mcp_config.json` for MCP server configuration:

```json
{
  "servers": {
    "microsoft-learn": {
      "type": "sse",
      "url": "https://learn.microsoft.com/api/mcp"
    }
  }
}
```

### 3. Basic Usage

```python
from mcp_client import create_mcp_client
from prompt_generator import MCPPromptGenerator

# Create client and query documentation
client = create_mcp_client()
docs = client.query_docs("nvidia nemo retriever embedding")

# Generate enhanced prompts
generator = MCPPromptGenerator()
migration_prompt = generator.build_migration_prompt("nemo_retriever")
```

## 📁 File Structure

```
├── mcp_config.json              # MCP server configuration
├── mcp_client.py               # Core MCP client implementation
├── prompt_generator.py         # Enhanced prompt generation
├── agent_integration.py        # Agent integration wrapper
├── test_mcp_integration.py     # Comprehensive test suite
├── usage_example.py           # Usage examples and demos
└── MCP_INTEGRATION.md         # This documentation
```

## 🔧 Core Components

### 1. NeMoMCPClient (`mcp_client.py`)

The main client for interacting with MCP servers:

```python
class NeMoMCPClient:
    """MCP Client for fetching NeMo Retriever documentation."""

    def query_docs(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]
    def get_nemo_retriever_context(self) -> str
    def build_migration_prompt(self, base_prompt: str, context_query: str) -> str
```

**Key Features:**

- Async-aware MCP integration
- Fallback documentation when MCP servers are unavailable
- Automatic session management
- Structured document parsing

### 2. MCPPromptGenerator (`prompt_generator.py`)

Enhanced prompt generation with live documentation:

```python
class MCPPromptGenerator:
    """Prompt generator with Microsoft Learn MCP integration."""

    def build_migration_prompt(self, migration_type: str, requirements: str) -> str
    def build_troubleshooting_prompt(self, error_description: str) -> str
    def build_feature_implementation_prompt(self, feature_name: str, requirements: str) -> str
```

**Supported Migration Types:**

- `nemo_retriever`: General NeMo Retriever migration
- `langchain_faiss`: LangChain + FAISS to NeMo migration
- `embedding_optimization`: Embedding performance optimization

### 3. MCPEnhancedAgent (`agent_integration.py`)

High-level agent wrapper with automatic context:

```python
class MCPEnhancedAgent:
    """Enhanced agent that integrates MCP documentation fetching."""

    def ask_with_context(self, query: str, context_type: str = "auto") -> str
    def generate_migration_code(self, source_system: str, requirements: str) -> str
    def debug_with_context(self, error_description: str) -> str
    def health_check(self) -> Dict[str, Any]
```

## 📚 Usage Patterns

### Pattern 1: Simple Documentation Query

```python
from mcp_client import create_mcp_client

client = create_mcp_client()
docs = client.query_docs("nvidia nemo retriever embedding")

for doc in docs:
    print(f"Title: {doc['title']}")
    print(f"URL: {doc['url']}")
    print(f"Content: {doc['content'][:200]}...")
```

### Pattern 2: Enhanced Prompt Generation

```python
from prompt_generator import generate_migration_prompt

# Quick migration prompt
prompt = generate_migration_prompt(
    "langchain_faiss",
    "Need to migrate 10M document corpus with custom preprocessing"
)

# Use with Claude-Code
response = claude_agent.ask(prompt)
```

### Pattern 3: Automatic Context Enhancement

```python
from agent_integration import MCPEnhancedAgent

agent = MCPEnhancedAgent()

# Automatically fetches relevant context
enhanced_query = agent.ask_with_context(
    "How do I optimize embedding performance?",
    context_type="auto"
)
```

### Pattern 4: Troubleshooting Assistant

```python
from agent_integration import create_enhanced_agent

agent = create_enhanced_agent()

debug_prompt = agent.debug_with_context(
    "Getting CUDA out of memory errors when processing large batches"
)
```

## ⚙️ Configuration Options

### MCP Server Configuration

```json
{
  "servers": {
    "microsoft-learn": {
      "type": "sse",
      "url": "https://learn.microsoft.com/api/mcp",
      "headers": {
        "Authorization": "Bearer ${API_TOKEN}"
      }
    },
    "nvidia-docs": {
      "type": "stdio",
      "command": "nvidia-docs-mcp",
      "args": ["--model", "nemo-retriever"],
      "env": {
        "NVIDIA_API_KEY": "${NVIDIA_API_KEY}"
      }
    }
  }
}
```

### Client Configuration

````python
# Enable/disable fallback mode
client = NeMoMCPClient(enable_fallback=True)

# Custom config path
client = NeMoMCPClient(config_path="custom_mcp_config.json")

---

## GitHub MCP Server Integration (Claude)

GitHub’s official MCP server lets Claude read/manage repos, issues/PRs, Actions, releases, and security alerts. Two ways to use it:

### Option A: Remote server (recommended; no cloning)

Requirements:
- Claude Code CLI installed (`claude` on PATH)
- GitHub Personal Access Token (PAT) in `.env` as `GITHUB_PAT` (use minimal scopes; start read-only)

Setup (project-scoped; writes `.mcp.json`, which is git-ignored):

```bash
make mcp-github-add
# Verify
make mcp-github-verify
````

What this does:

- Runs `claude mcp add -s project --transport http -H "Authorization: Bearer $GITHUB_PAT" github https://api.githubcopilot.com/mcp/`
- Creates `.mcp.json` in the project directory (not committed)

Remove:

```bash
make mcp-github-remove
```

### Option B: Local server via Docker (Claude Desktop or CLI)

If Claude Desktop cannot use the remote server (OAuth limitations), run the containerized server locally.

1. Ensure Docker is running and set `GITHUB_PAT` in your shell or `.env`.
2. Claude Desktop: open Settings → Developer → Edit Config and merge the snippet below into your config file:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PAT}"
      }
    }
  }
}
```

Restart Claude Desktop after saving the config.

CLI equivalent (no Desktop):

```bash
claude mcp add github -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_PAT -- \
  docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
```

### Token and Security Notes

- Start with minimal scopes; add `repo` only if you need write operations (issues/PRs).
- Store tokens in `.env` (already git-ignored). Rotate if ever exposed.
- `.mcp.json` is now git-ignored to prevent committing headers.

### Troubleshooting

- `claude mcp list` and `claude mcp get github` to inspect config.
- Docker pull issues: `docker logout ghcr.io` then retry.
- Claude Desktop logs (macOS): `~/Library/Logs/Claude/`.

# Agent with custom MCP client

agent = MCPEnhancedAgent(mcp_client=client, enable_auto_context=True)

````

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_mcp_integration.py
````

**Test Coverage:**

- MCP client creation and initialization
- Documentation querying (with fallback)
- Context generation
- Prompt enhancement
- Performance benchmarking
- Error handling

### Test Results

```
🧪 Microsoft Learn MCP Integration Test Suite
============================================================
1. MCP Client Creation: ✅ PASS
2. Documentation Query: ✅ PASS
3. Context Generation: ✅ PASS
4. Prompt Generation: ✅ PASS
5. Convenience Functions: ✅ PASS
6. Performance Test: ✅ PASS

Overall: 6/6 tests passed
🎉 All tests passed! MCP integration is working correctly.
```

## 🎯 Use Cases

### 1. Migration Projects

Generate migration code with up-to-date documentation:

```python
migration_prompt = agent.generate_migration_code(
    "Existing LangChain + FAISS system",
    "Handle 10M documents, maintain 99.9% uptime during migration"
)
```

### 2. Performance Optimization

Get optimization guidance with latest best practices:

```python
optimization_prompt = generator.build_feature_implementation_prompt(
    "High-Performance Embedding Pipeline",
    "Process 1M documents/hour with GPU acceleration"
)
```

### 3. Troubleshooting

Resolve issues with current documentation:

```python
troubleshoot_prompt = generator.build_troubleshooting_prompt(
    "Embedding model returns inconsistent results"
)
```

### 4. Feature Development

Implement new features following current patterns:

```python
feature_prompt = agent.implement_feature_with_context(
    "Batch Processing System",
    "Support multiple embedding models with automatic failover"
)
```

## 🔄 Fallback Mechanism

When MCP servers are unavailable, the system provides high-quality fallback documentation:

### Fallback Features

- **Comprehensive Coverage**: Covers common NeMo Retriever topics
- **Realistic URLs**: Points to actual NVIDIA documentation structure
- **Relevance Scoring**: Provides meaningful relevance scores
- **Context-Aware**: Adapts responses based on query context

### Fallback Topics

- NeMo Retriever overview and architecture
- Embedding models and NIMs
- LangChain integration patterns
- Migration guides and best practices
- Performance optimization techniques
- Troubleshooting common issues

## 🚦 Health Monitoring

Monitor MCP integration health:

```python
agent = MCPEnhancedAgent()
health = agent.health_check()

print(f"MCP Client Active: {health['mcp_client_active']}")
print(f"Servers Registered: {health['server_registered']}")
print(f"Documentation Accessible: {health['documentation_accessible']}")
```

## 🔐 Security Considerations

### API Keys and Authentication

Store sensitive information in environment variables:

```bash
export MICROSOFT_LEARN_API_KEY="your-api-key"
export NVIDIA_API_KEY="your-nvidia-key"
```

Reference in configuration:

```json
{
  "servers": {
    "microsoft-learn": {
      "type": "sse",
      "url": "https://learn.microsoft.com/api/mcp",
      "headers": {
        "Authorization": "Bearer ${MICROSOFT_LEARN_API_KEY}"
      }
    }
  }
}
```

### Best Practices

1. **Rate Limiting**: Implement appropriate rate limiting for API calls
2. **Caching**: Cache frequently accessed documentation
3. **Error Handling**: Graceful degradation when services are unavailable
4. **Logging**: Comprehensive logging for debugging and monitoring

## 🔮 Future Enhancements

### Planned Features

1. **Multiple MCP Servers**: Support for NVIDIA AIQ, GitHub Docs, etc.
2. **Smart Caching**: Intelligent caching with TTL and invalidation
3. **Context Similarity**: Vector-based context matching
4. **Real-time Updates**: WebSocket support for live documentation updates

### Extension Points

```python
# Custom documentation sources
class CustomMCPClient(NeMoMCPClient):
    def _get_fallback_docs(self, query: str, max_results: int):
        # Add your custom documentation sources
        return custom_docs

# Custom prompt patterns
class CustomPromptGenerator(MCPPromptGenerator):
    def build_custom_prompt(self, template: str, context: Dict):
        # Add your custom prompt patterns
        return enhanced_prompt
```

## 📞 Support and Troubleshooting

### Common Issues

1. **MCP Server Connection Failed**

   - Check network connectivity
   - Verify server URL and authentication
   - Review firewall settings

2. **No Documentation Retrieved**

   - Verify query syntax
   - Check server response format
   - Review fallback configuration

3. **Performance Issues**
   - Monitor query response times
   - Implement caching strategies
   - Optimize batch sizes

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = create_mcp_client()
docs = client.query_docs("your query here")
```

## 📄 License

This MCP integration follows the same license as the main RAG template project.

---

**Ready to integrate live documentation into your NeMo Retriever workflow!** 🚀

Start with the usage examples and gradually integrate the components that best fit your use case.
