# AI Engineer Preflight

Preflight readiness validation for LLM apps, RAG pipelines, and agent systems.

## Instructions

1. Parse requirements from: $ARGUMENTS
2. Identify target providers (OpenAI/Anthropic/Cohere), vector DBs (Pinecone/Qdrant/Weaviate/Chroma/Milvus), and observability needs
3. Run through the preflight checklist (env, keys, network, vector DB, embeddings, RAG assets, safety, observability, performance)
4. Produce a PASS/FAIL table with fix commands for each failed item
5. Recommend concurrency via scripts/orchestrate_benchmarks.py and show invocation

## Usage Examples

```bash
/project:ai-engineer-preflight Validate OpenAI + Qdrant setup for hybrid RAG
/project:ai-engineer-preflight Check Anthropic + Pinecone and Cohere rerank readiness
/project:ai-engineer-preflight Recommend concurrency and save preflight to logs/
```

## Output Format

- Summary bullets of PASS/FAIL per category
- Code blocks with commands to fix issues
- Optional: a JSON suggestion for `.env` keys to add (redact secrets)

## Notes

- Do not echo secrets. Show variable names and placeholders only
- Prefer minimal, shell-friendly commands
- If a category is not used in this project, mark as N/A
