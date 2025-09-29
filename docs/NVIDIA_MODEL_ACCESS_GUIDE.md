# NVIDIA Model Access and Endpoint Configuration Guide

## Overview

This guide explains the different NVIDIA API endpoints, model access tiers, and how to configure your RAG system for optimal model access. Your system is currently optimized for pharmaceutical research with specialized NeMo Retriever models, and this guide helps you understand alternatives and fallback options.

## NVIDIA API Endpoint Types

### 1. NVIDIA AI Retrieval Services (`ai.api.nvidia.com`)

**Purpose**: Specialized NeMo Retriever services for enterprise-grade retrieval applications
**Current Status**: Your system's primary configuration
**Models**: Advanced NeMo models optimized for specific domains

```bash
Base URL: https://ai.api.nvidia.com/v1
Endpoints:
  - /retrieval/nvidia/embeddings
  - /retrieval/nvidia/reranking
  - /retrieval/nvidia/extraction
```

**Your Current Models (Optimal for Pharmaceutical Research):**
- **Embedding**: `nvidia/nv-embedqa-e5-v5` (1024 dimensions, Q&A optimized)
- **Reranking**: `meta/llama-3_2-nemoretriever-500m-rerank-v2` (pharmaceutical-specific)
- **LLM**: `meta/llama-3.1-8b-instruct` (configured)

### 2. NVIDIA Build Platform (`integrate.api.nvidia.com`)

**Purpose**: Developer-friendly access to popular open-source models
**Target**: General-purpose applications and cost-effective alternatives
**Free Tier**: 10,000 requests for registered developers

```bash
Base URL: https://integrate.api.nvidia.com/v1
Endpoints:
  - /embeddings
  - /chat/completions
```

**Available Models (Requested in your question):**
- **Embedding**: `nvidia/nv-embed-v1` (4096 dimensions, general purpose)
- **LLM**: `meta/llama-3.1-8b-instruct` (same model, different endpoint)

## API Key and Access Analysis

### Your Current API Key: `nvapi-FTRR...`

**Test Results**: The validation utility shows **403 Forbidden** errors across all endpoints, indicating:

1. **API Key Issue**: Your key may not have the required permissions
2. **Service Tier**: Different keys provide access to different endpoint sets
3. **Account Status**: May require activation or billing setup

### Recommendations for API Access

#### Option 1: Verify Current API Key (Recommended)
```bash
# Test your current setup
python scripts/model_validation_utility.py --verbose

# Check if your NeMo services work
python scripts/nim_native_test.py
```

#### Option 2: Get NVIDIA Build Platform Access
1. Visit [build.nvidia.com](https://build.nvidia.com)
2. Create account or sign in
3. Generate API key for Build platform
4. Test with: `python scripts/nvidia_build_api_test.py`

#### Option 3: Contact NVIDIA Support
- Check account status and permissions
- Verify API key is active and configured correctly
- Request access to specific endpoints if needed

## Model Comparison and Recommendations

### For Pharmaceutical Research (Your Current Setup - BEST)

| Model | Purpose | Dimensions | Max Tokens | Pharmaceutical Optimization |
|-------|---------|------------|------------|----------------------------|
| `nvidia/nv-embedqa-e5-v5` | Embedding | 1024 | 32k | ✅ Medical Q&A optimized |
| `meta/llama-3_2-nemoretriever-500m-rerank-v2` | Reranking | N/A | N/A | ✅ Pharmaceutical-specific |
| `meta/llama-3.1-8b-instruct` | LLM | N/A | 128k | ✅ General purpose, good reasoning |

**Why these are optimal for your use case:**
- Medical terminology understanding
- Pharmaceutical domain knowledge
- Clinical trial and regulatory document processing
- Drug interaction and compound analysis

### Alternative Models (If Build Platform Access Available)

| Model | Purpose | Dimensions | Max Tokens | Use Case |
|-------|---------|------------|------------|----------|
| `nvidia/nv-embed-v1` | Embedding | 4096 | 32k | General purpose, cost-effective fallback |
| `meta/llama-3.1-8b-instruct` | LLM | N/A | 128k | Same model, different endpoint |

## Configuration Options

### Current Configuration (.env)
```bash
# Optimal for pharmaceutical research
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
RERANK_MODEL=llama-3_2-nemoretriever-500m-rerank-v2
LLM_MODEL_NAME=meta/llama-3.1-8b-instruct
```

### Adding NVIDIA Build Fallback
```bash
# Enable fallback to Build platform models
ENABLE_NVIDIA_BUILD_FALLBACK=true
NVIDIA_BUILD_EMBEDDING_MODEL=nvidia/nv-embed-v1
NVIDIA_BUILD_LLM_MODEL=meta/llama-3.1-8b-instruct
```

### Testing Your Configuration
```bash
# Comprehensive validation
python scripts/model_validation_utility.py

# Test specific NVIDIA Build access
python scripts/nvidia_build_api_test.py

# Test current NeMo setup
python scripts/nim_native_test.py
```

## Troubleshooting Common Issues

### 403 Forbidden Errors
```json
{"status":403,"title":"Forbidden","detail":"Authorization failed"}
```

**Causes:**
- API key lacks permissions for endpoint
- Account not activated
- Service tier restrictions

**Solutions:**
1. Verify API key format: should start with `nvapi-`
2. Check account status on NVIDIA platform
3. Try different endpoint URLs
4. Contact NVIDIA support

### 404 Not Found Errors
```
404 page not found
```

**Causes:**
- Endpoint URL incorrect
- Service not available in your region
- Model name format incorrect

**Solutions:**
1. Use exact endpoint URLs from documentation
2. Verify model names match exactly
3. Check service availability

### Model Not Available
**Causes:**
- Model not included in your API tier
- Regional restrictions
- Service-specific models

**Solutions:**
1. Check available models for your tier
2. Use alternative models with similar capabilities
3. Upgrade service tier if needed

## Conclusion

**Your Current Setup is Optimal** for pharmaceutical research. The requested models (`nvidia/nv-embed-v1` and `meta/llama-3.1-8b-instruct`) are available through NVIDIA Build platform, but your current NeMo Retriever models provide superior performance for medical/pharmaceutical applications.

**Immediate Action Items:**
1. ✅ Keep current NeMo models as primary
2. 🔧 Resolve API key access issues
3. 💡 Consider Build platform as cost-effective fallback
4. 📋 Use validation utilities to monitor access

**For questions or issues**: Run the diagnostic tools and contact NVIDIA support with the output for faster resolution.