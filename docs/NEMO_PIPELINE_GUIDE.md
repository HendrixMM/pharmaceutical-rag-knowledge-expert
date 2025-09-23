# NVIDIA NeMo Retriever Pipeline Guide

## Overview

This guide covers the complete three-step NVIDIA NeMo Retriever pipeline implementation:

1. **Extraction** - Document processing using NV-Ingest VLM
2. **Embedding** - Text vectorization using NeMo embedding models
3. **Reranking** - Relevance-based reordering using NeMo reranking models

The pipeline is optimized for pharmaceutical and medical documents with domain-specific enhancements, regulatory compliance features, and enterprise-grade error handling.

## Quick Start

### Prerequisites

- NVIDIA API key (set `NVIDIA_API_KEY` environment variable)
- Python 3.8+
- Required dependencies (see `requirements.txt`)

### Basic Usage

```bash
# Set up environment
export NVIDIA_API_KEY="your_nvidia_api_key_here"
export ENABLE_NEMO_EXTRACTION=true
export EMBEDDING_MODEL="nvidia/nv-embedqa-e5-v5"
export RERANK_MODEL="llama-3_2-nemoretriever-500m-rerank-v2"

# Run the complete pipeline test
python scripts/nim_native_test.py --pdf path/to/your/document.pdf
```

## Pipeline Architecture

### Step 1: Document Extraction

**Service**: `NeMoExtractionService`
**Technology**: NVIDIA NV-Ingest VLM-based OCR
**Purpose**: Extract text, tables, and images from PDF documents

#### Key Features

- **VLM-based OCR**: Superior accuracy for complex pharmaceutical documents
- **Table Preservation**: Maintains dosing tables and clinical data structure
- **Image Processing**: Extracts molecular structures and charts
- **Pharmaceutical Analysis**: Domain-specific entity extraction
- **Fallback Strategy**: Unstructured → PyPDF if NeMo VLM unavailable

#### Configuration

```bash
# Enable NeMo extraction
ENABLE_NEMO_EXTRACTION=true

# Extraction strategy
NEMO_EXTRACTION_STRATEGY=nemo  # nemo, auto, unstructured

# Pharmaceutical domain analysis
NEMO_PHARMACEUTICAL_ANALYSIS=true

# Content preservation
NEMO_PRESERVE_TABLES=true
NEMO_EXTRACT_IMAGES=true

# Chunking strategy
NEMO_CHUNK_STRATEGY=semantic  # semantic, title, page

# Strict mode (production)
NEMO_EXTRACTION_STRICT=false
APP_ENV=development  # production enforces strict=true
```

#### Usage Example

```python
from src.nemo_extraction_service import NeMoExtractionService

# Initialize service
service = NeMoExtractionService()

# Extract from PDF
result = await service.extract_from_pdf("pharmaceutical_document.pdf")

if result.success:
    print(f"Extracted {len(result.documents)} documents")
    for doc in result.documents:
        print(f"Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")
else:
    print(f"Extraction failed: {result.metadata.get('error')}")
```

### Step 2: Text Embedding

**Service**: `NeMoEmbeddingService`
**Models**: Multiple NVIDIA embedding models
**Purpose**: Convert text into dense vector representations

#### Available Models

| Model | Dimensions | Best For | Use Case |
|-------|------------|----------|----------|
| `nv-embedqa-e5-v5` | 1024 | Q&A, Medical | Pharmaceutical search |
| `nv-embedqa-mistral7b-v2` | 4096 | Multilingual, Complex | Global drug information |
| `snowflake-arctic-embed-l` | 1024 | General text | Basic similarity |

#### Configuration

```bash
# Model selection
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5

# Pharmaceutical optimization
PHARMA_DOMAIN_OVERLAY=true
```

#### Usage Example

```python
from src.nemo_embedding_service import NeMoEmbeddingService, EmbeddingConfig

# Configure for pharmaceutical content
config = EmbeddingConfig(
    model="nv-embedqa-e5-v5",
    pharmaceutical_optimization=True,
    batch_size=100
)

# Initialize service
service = NeMoEmbeddingService(config=config)

# Embed documents
texts = ["Metformin indication", "Aspirin contraindications"]
embeddings = await service.embed_documents(texts)

print(f"Generated {len(embeddings)} embeddings of {len(embeddings[0])} dimensions")
```

### Step 3: Passage Reranking

**Service**: `NeMoRetrieverClient.rerank_passages()`
**Models**: NeMo reranking models
**Purpose**: Reorder search results by relevance

#### Available Models

| Model | Max Pairs | Best For | Use Case |
|-------|-----------|----------|----------|
| `nv-rerankqa-mistral4b-v3` | 1000 | General Q&A | Standard reranking |
| `llama-3_2-nemoretriever-500m-rerank-v2` | 1000 | Latest NeMo | Optimal performance |

#### Configuration

```bash
# Model selection
RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2
```

#### Usage Example

```python
from src.nemo_retriever_client import create_nemo_client

# Initialize client
client = await create_nemo_client()

# Rerank passages
result = await client.rerank_passages(
    query="drug interactions with metformin",
    passages=[
        "Metformin contraindicated with contrast agents",
        "Aspirin increases bleeding risk",
        "Metformin and alcohol interaction warning"
    ],
    model="llama-3_2-nemoretriever-500m-rerank-v2",
    top_k=3
)

if result.success:
    for i, passage in enumerate(result.data["reranked_passages"]):
        print(f"{i+1}. Score: {passage['score']:.3f} - {passage['text']}")
```

## Complete Pipeline Integration

### End-to-End Example

```python
import asyncio
from pathlib import Path
from src.nemo_extraction_service import NeMoExtractionService
from src.nemo_embedding_service import NeMoEmbeddingService, EmbeddingConfig
from src.nemo_retriever_client import create_nemo_client

async def pharmaceutical_pipeline(pdf_path: str, query: str):
    \"\"\"Complete pharmaceutical document processing pipeline.\"\"\"

    # Step 1: Extract documents
    print("🔍 Extracting documents...")
    extraction_service = NeMoExtractionService()
    extraction_result = await extraction_service.extract_from_pdf(pdf_path)

    if not extraction_result.success:
        raise Exception(f"Extraction failed: {extraction_result.metadata.get('error')}")

    documents = [doc.page_content for doc in extraction_result.documents]
    print(f"✅ Extracted {len(documents)} documents")

    # Step 2: Generate embeddings
    print("🧮 Generating embeddings...")
    embedding_config = EmbeddingConfig(
        model="nv-embedqa-e5-v5",
        pharmaceutical_optimization=True
    )
    embedding_service = NeMoEmbeddingService(config=embedding_config)
    embeddings = await embedding_service.embed_documents(documents)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Step 3: Rerank by relevance
    print("📊 Reranking by relevance...")
    client = await create_nemo_client()
    rerank_result = await client.rerank_passages(
        query=query,
        passages=documents,
        model="llama-3_2-nemoretriever-500m-rerank-v2",
        top_k=5
    )

    if rerank_result.success:
        print(f"✅ Reranked to top {len(rerank_result.data['reranked_passages'])} passages")
        return rerank_result.data["reranked_passages"]
    else:
        raise Exception(f"Reranking failed: {rerank_result.error}")

# Usage
results = await pharmaceutical_pipeline(
    pdf_path="metformin_prescribing_info.pdf",
    query="contraindications and drug interactions"
)

for i, passage in enumerate(results):
    print(f"\\n{i+1}. Relevance: {passage['score']:.3f}")
    print(f"Content: {passage['text'][:200]}...")
```

### Using the Automated Test Script

The repository includes a comprehensive test script that runs the complete pipeline:

```bash
# Basic usage
python scripts/nim_native_test.py

# With specific PDF
python scripts/nim_native_test.py --pdf path/to/document.pdf

# The script will:
# 1. Validate all NIM services are healthy
# 2. Extract content using NeMo VLM
# 3. Generate embeddings using configured model
# 4. Rerank passages for pharmaceutical relevance
# 5. Apply pharmaceutical domain overlay analysis
```

## Pharmaceutical Domain Features

### Enhanced Metadata Extraction

The pipeline automatically extracts pharmaceutical-specific metadata:

```python
# Automatic extraction includes:
{
    "drug_names": ["Metformin", "Aspirin"],
    "drug_canonical_names": ["metformin_hydrochloride", "acetylsalicylic_acid"],
    "dosages": ["500mg", "81mg"],
    "indications": ["Type 2 diabetes", "Cardiovascular protection"],
    "contraindications": ["Severe renal impairment", "Active bleeding"],
    "regulatory_status": [{"drug": "metformin", "agency": "FDA", "status": "Approved"}],
    "evidence_level": "high",
    "species": "human",
    "cyp_risk_label": "moderate"
}
```

### Regulatory Compliance

- **FDA 21 CFR Part 11**: Audit trail and data integrity
- **GMP/GLP Compliance**: Validated data processing
- **GDPR**: Patient data protection where applicable

### Safety-First Processing

- **Contraindications**: Always prioritized in reranking
- **Black Box Warnings**: Elevated in search results
- **Drug Interactions**: Enhanced detection and flagging

## Free Tier Optimization

### NVIDIA Build Credits Monitoring

```python
from src.nemo_retriever_client import NVIDIABuildCreditsMonitor

# Enable credits monitoring
monitor = NVIDIABuildCreditsMonitor("your_api_key")

# Use with pipeline
client = await create_nemo_client(credits_monitor=monitor)

# Monitor usage
print(f"Credits used: {monitor.credits_used}")
print(f"Credits remaining: {monitor.credits_remaining}")
```

### Free Tier Best Practices

1. **Batch Processing**: Group documents to minimize API calls
2. **Caching**: Enable embedding caching for repeated queries
3. **Smart Chunking**: Use semantic chunking to reduce redundancy
4. **Model Selection**: Use efficient models for your use case

## Environment Configuration

### Complete .env Example

```bash
# NVIDIA API Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here

# NeMo Extraction Configuration
ENABLE_NEMO_EXTRACTION=true
NEMO_EXTRACTION_STRATEGY=nemo
NEMO_PHARMACEUTICAL_ANALYSIS=true
NEMO_PRESERVE_TABLES=true
NEMO_EXTRACT_IMAGES=true
NEMO_CHUNK_STRATEGY=semantic
NEMO_EXTRACTION_STRICT=false
APP_ENV=development

# Model Selection
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2

# Pharmaceutical Domain Overlay
PHARMA_DOMAIN_OVERLAY=true
DRUG_SYNONYMS_CSV=./data/drug_synonyms.csv
REGULATORY_STATUS_CSV=./data/regulatory_status.csv

# Free Tier Configuration
NVIDIA_BUILD_FREE_TIER=false
NVIDIA_RATE_LIMIT_AWARE=true
NVIDIA_CREDITS_MONITORING=true
```

## Testing

### Running Unit Tests

```bash
# Run all NeMo tests
pytest tests/test_nemo_*.py -v

# Run specific service tests
pytest tests/test_nemo_retriever_client.py -v
pytest tests/test_nemo_embedding_service.py -v
pytest tests/test_nemo_extraction_service.py -v

# Run with coverage
pytest tests/test_nemo_*.py --cov=src --cov-report=html
```

### Integration Testing

```bash
# Full pipeline integration test
python scripts/nim_native_test.py

# With pharmaceutical overlay
PHARMA_DOMAIN_OVERLAY=true python scripts/nim_native_test.py
```

## Performance Optimization

### Embedding Optimization

- **Batch Size**: Adjust based on available memory and API limits
- **Model Selection**: Choose appropriate model for content type
- **Caching**: Enable for frequently processed content

### Extraction Optimization

- **Chunking Strategy**: Semantic for quality, page for speed
- **Table Preservation**: Enable only when needed
- **Image Processing**: Disable for text-only documents

### Reranking Optimization

- **Top-K Selection**: Limit to necessary number of results
- **Query Optimization**: Use specific, domain-relevant queries
- **Model Selection**: Use latest models for best performance

## Troubleshooting

### Common Issues

#### 1. API Key Issues
```bash
# Check API key is set
echo $NVIDIA_API_KEY

# Test API access
python -c "import os; from src.nemo_retriever_client import create_nemo_client; client = await create_nemo_client()"
```

#### 2. Model Availability
```bash
# Check service health
python scripts/nim_native_test.py
```

#### 3. Extraction Failures
```bash
# Enable fallback mode
export NEMO_EXTRACTION_STRICT=false
export NEMO_EXTRACTION_STRATEGY=auto
```

#### 4. Memory Issues
```bash
# Reduce batch sizes
export NEMO_BATCH_SIZE=50
```

### Error Handling

The pipeline includes comprehensive error handling:

- **Graceful Degradation**: Falls back to alternative methods
- **Detailed Logging**: Comprehensive error information
- **Retry Logic**: Automatic retries for transient failures
- **Validation**: Input validation at each stage

## Production Deployment

### Environment Setup

```bash
# Production environment
export APP_ENV=production
export NEMO_EXTRACTION_STRICT=true

# Monitoring
export NVIDIA_CREDITS_MONITORING=true
export LOG_LEVEL=INFO
```

### Health Checks

```python
from src.nemo_retriever_client import create_nemo_client

async def health_check():
    client = await create_nemo_client()
    health = await client.health_check(force=True)

    for service, status in health.items():
        if status["status"] != "healthy":
            print(f"⚠️  {service} unhealthy: {status.get('error')}")
            return False

    print("✅ All NIM services healthy")
    return True
```

### Scaling Considerations

- **Rate Limiting**: Implement request throttling
- **Concurrent Processing**: Use asyncio for parallelization
- **Caching Strategy**: Implement distributed caching
- **Load Balancing**: Distribute across multiple API keys

## MCP-Enhanced Documentation Context

Your implementation includes an advanced **Model Context Protocol (MCP)** integration that provides live, up-to-date documentation context for all NeMo operations.

### MCP Configuration

The MCP system is configured in `mcp_config.json` with multiple documentation servers:

```json
{
  "servers": {
    "microsoft_learn": "NVIDIA NeMo documentation from Microsoft Learn",
    "nvidia_api_docs": "NVIDIA NIM API documentation for retrieval services",
    "nvidia_nemo_models": "NVIDIA NeMo Retriever model documentation",
    "nvidia_llama_rerank": "LLaMA-based reranking model documentation",
    "nvidia_embedding_models": "NV-EmbedQA and embedding model documentation"
  },
  "contexts": {
    "pharmaceutical_nemo_optimization": [
      "microsoft_learn", "nvidia_nemo_models",
      "nvidia_embedding_models", "nvidia_llama_rerank"
    ]
  }
}
```

### Using MCP Context in Your Pipeline

```python
from src.mcp_documentation_context import get_nemo_context, get_nemo_pipeline_recommendations

# Get live documentation for specific pipeline steps
embedding_context = get_nemo_context('embedding', 'nv-embedqa-e5-v5')
reranking_context = get_nemo_context('reranking', 'llama-3_2-nemoretriever-500m-rerank-v2')

# Get pharmaceutical-optimized recommendations
recommendations = get_nemo_pipeline_recommendations()
print(f"Recommended embedding model: {recommendations['embedding_model']}")
print(f"Recommended reranking model: {recommendations['reranking_model']}")
```

### Live Documentation Benefits

- **Always Current**: Automatically fetches latest NVIDIA documentation
- **Pharmaceutical Context**: Domain-specific optimization guidance
- **Model-Specific**: Tailored advice for each model in your pipeline
- **Error Resolution**: Up-to-date troubleshooting information

## Support and Resources

### Documentation
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NeMo Retriever Guide](https://docs.nvidia.com/nemo/retriever/)
- [API Reference](https://docs.api.nvidia.com/)
- [MCP Documentation Context Service](../src/mcp_documentation_context.py) - Live documentation integration
- [Pipeline Configuration Guide](../mcp_config.json) - MCP server configuration

### Community
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)

### Enterprise Support
- Contact NVIDIA Enterprise Support for production deployments
- Consider NVIDIA AI Enterprise for enhanced support and features

---

**📋 Summary**: This pipeline provides enterprise-grade pharmaceutical document processing with NVIDIA's latest NeMo Retriever technology, optimized for accuracy, compliance, and performance.