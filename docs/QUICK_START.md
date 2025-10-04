# Quick Start Guide

---

**Last Updated**: 2025-10-03
**Owner**: Documentation Team
**Estimated Time**: 5-10 minutes

---

Get your first pharmaceutical query running in minutes with this step-by-step guide.

## Prerequisites

Before you begin, ensure you have:

- ✅ Python 3.10 or higher installed
- ✅ NVIDIA API key ([Sign up for free](https://build.nvidia.com/))
- ✅ Basic familiarity with command line

Haven't installed yet? See [Installation Guide](INSTALLATION.md).

---

## 5-Minute Setup

### Step 1: Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever.git
cd RAG-Template-for-NVIDIA-nemoretriever

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your NVIDIA API key
# NVIDIA_API_KEY=your_api_key_here
```

**Get your NVIDIA API key:**

1. Visit [NVIDIA Build](https://build.nvidia.com/)
2. Sign in or create account
3. Navigate to "API Keys" section
4. Generate new API key
5. Copy key to `.env` file

### Step 3: Verify Setup (30 seconds)

```bash
# Quick health check
python -c "from src.clients.nemo_client_enhanced import EnhancedNeMoClient; print('✅ Setup verified')"
```

---

## Your First Query

### Python API

```python
from src.enhanced_rag_agent import EnhancedRAGAgent
import asyncio

async def main():
    # Initialize agent
    agent = EnhancedRAGAgent()

    # Run pharmaceutical query
    result = await agent.query(
        "What are the drug interactions for warfarin with NSAIDs?"
    )

    # Display results
    print("Response:")
    print(result['response'])
    print(f"\nSources: {len(result['sources'])}")

    # Show source citations
    for i, source in enumerate(result['sources'][:3], 1):
        print(f"\n{i}. {source.get('title', 'Unknown')}")
        print(f"   PMID: {source.get('pmid', 'N/A')}")

# Run the query
asyncio.run(main())
```

**Expected output:**

```
Response:
Warfarin, an anticoagulant, has significant drug interactions with NSAIDs...
[Medical disclaimer and detailed information]

Sources: 5

1. Drug Interactions Between Warfarin and Nonsteroidal Anti-inflammatory Agents
   PMID: 12345678

2. Clinical Significance of NSAID-Warfarin Interactions
   PMID: 87654321

3. Bleeding Risk with Combined Warfarin and NSAID Therapy
   PMID: 11223344
```

### Web Interface

Launch the Streamlit web interface for interactive queries:

```bash
# Start web interface
streamlit run ui/streamlit_app.py

# Access at: http://localhost:8501
```

Features:
- Interactive query input
- Real-time results
- Source visualization
- Cost tracking dashboard
- Query history

---

## Common Query Examples

### Drug Information Query

```python
result = await agent.query(
    "What is the mechanism of action for metformin in type 2 diabetes?"
)
```

### Drug Safety Query

```python
result = await agent.query(
    "What are the contraindications and warnings for lisinopril?"
)
```

### Clinical Research Query

```python
result = await agent.query(
    "What are the latest clinical trials for Alzheimer's treatment?"
)
```

### Drug Interaction Query

```python
result = await agent.query(
    "Are there any interactions between atorvastatin and grapefruit juice?"
)
```

See [Examples](EXAMPLES.md) for more comprehensive use cases.

---

## Understanding the Response

### Response Structure

```python
{
    'response': str,           # Generated answer with medical disclaimers
    'sources': List[Dict],     # Source documents with metadata
    'metadata': {
        'query_time': float,   # Response time in seconds
        'cost': float,         # API cost in credits
        'validation': Dict,    # Guardrail validation results
        'model': str          # Model used for generation
    }
}
```

### Medical Guardrails

Every query passes through three validation layers:

1. **Input Rails**: PII/PHI detection, jailbreak prevention
2. **Retrieval Rails**: Source quality, medical relevance
3. **Output Rails**: Hallucination detection, regulatory compliance

```python
# Check validation results
print(result['metadata']['validation'])
# {
#     'input_valid': True,
#     'sources_filtered': 2,
#     'output_safe': True,
#     'disclaimer_added': True
# }
```

See [Architecture](ARCHITECTURE.md) for detailed guardrail documentation.

---

## Basic Configuration

### Environment Variables

Essential configuration in `.env`:

```ini
# NVIDIA API (Required)
NVIDIA_API_KEY=your_api_key_here

# Strategy (Recommended defaults)
ENABLE_CLOUD_FIRST_STRATEGY=true
ENABLE_PHARMACEUTICAL_OPTIMIZATION=true

# Safety (Recommended: enabled)
ENABLE_MEDICAL_GUARDRAILS=true
```

### Model Selection

```python
from src.enhanced_rag_agent import EnhancedRAGAgent

# Use specific models
agent = EnhancedRAGAgent(
    embedding_model="nvidia/nv-embedqa-e5-v5",
    llm_model="meta/llama-3.1-70b-instruct"
)
```

Available models:
- Embeddings: `nvidia/nv-embedqa-e5-v5` (recommended)
- LLM: `meta/llama-3.1-70b-instruct`, `meta/llama-3.1-8b-instruct`
- Reranker: `nvidia/nv-rerankqa-mistral-4b-v3`

See [Configuration Guide](CONFIGURATION.md) for all options.

---

## Cost Monitoring

Track your API usage:

```python
from src.cost_tracker import CostTracker

tracker = CostTracker()

# After running queries
summary = tracker.get_summary()
print(f"Total queries: {summary['total_queries']}")
print(f"Total cost: ${summary['total_cost']:.2f}")
print(f"Average cost: ${summary['avg_cost']:.2f} per query")
```

**Free Tier Limits:**
- Embedding: 1,000 requests/day
- Chat completion: 100 requests/day
- Reranking: 500 requests/day

See [FREE_TIER_MAXIMIZATION.md](FREE_TIER_MAXIMIZATION.md) for optimization strategies.

---

## Command Line Interface

For quick queries without writing code:

```bash
# Single query
python cli.py query "What are the side effects of aspirin?"

# Interactive mode
python cli.py interactive

# Batch processing
python cli.py batch --input queries.txt --output results.json
```

CLI features:
- JSON output format
- Query history
- Cost tracking
- Source export

---

## Troubleshooting

### Issue: Import Error

```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: API Authentication Failed

```bash
# Solution: Verify API key
echo $NVIDIA_API_KEY  # Should show your key

# Test API connectivity
curl -H "Authorization: Bearer $NVIDIA_API_KEY" \
     https://integrate.api.nvidia.com/v1/models
```

### Issue: Medical Guardrails Error

```bash
# Solution: Install medical dependencies
pip install -r requirements-medical.txt

# Or disable guardrails temporarily
export ENABLE_MEDICAL_GUARDRAILS=false
```

### Issue: Slow Response Times

```python
# Solution: Enable caching
from src.enhanced_rag_agent import EnhancedRAGAgent

agent = EnhancedRAGAgent(enable_cache=True)
```

See [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) for more issues.

---

## Next Steps

### Learn More

1. **Configuration**: Customize models, guardrails, and optimization
   - [Configuration Guide](CONFIGURATION.md)

2. **Advanced Features**: Explore batch processing, monitoring, and deployment
   - [API Reference](API_REFERENCE.md)
   - [Examples](EXAMPLES.md)

3. **Production Deployment**: Deploy to cloud environments
   - [Deployment Guide](DEPLOYMENT.md)

### Explore Use Cases

- 📚 **Literature Review**: [PHARMACEUTICAL_BEST_PRACTICES.md](PHARMACEUTICAL_BEST_PRACTICES.md)
- 🔬 **Clinical Research**: [EXAMPLES.md](EXAMPLES.md#pharmaceutical-examples)
- 💊 **Drug Safety**: [EXAMPLES.md](EXAMPLES.md#example-4-safety-alert-integration)
- 📊 **Regulatory Compliance**: [EXAMPLES.md](EXAMPLES.md#complete-workflows)

### Optimize Performance

- **Cost Optimization**: [FREE_TIER_MAXIMIZATION.md](FREE_TIER_MAXIMIZATION.md)
- **Performance Tuning**: [BENCHMARKS.md](BENCHMARKS.md)
- **Monitoring Setup**: [MONITORING.md](MONITORING.md)

---

## Getting Help

### Documentation

- 📖 **Full Documentation**: [docs/](.)
- 🏗️ **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- 🔧 **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- ❓ **Troubleshooting**: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

### Community Support

- 💬 **Discussions**: [GitHub Discussions](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/discussions)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/issues)
- 🔒 **Security**: [SECURITY.md](SECURITY.md)

### Additional Resources

- **Examples Repository**: [examples/](../examples/)
- **Video Tutorials**: [Coming soon]
- **API Documentation**: [API_REFERENCE.md](API_REFERENCE.md)

---

## Example Notebooks

Explore Jupyter notebooks in `examples/notebooks/`:

1. `01_basic_query.ipynb` - Simple pharmaceutical queries
2. `02_advanced_filtering.ipynb` - Clinical study filtering
3. `03_drug_interactions.ipynb` - Interaction analysis
4. `04_batch_processing.ipynb` - Large-scale processing
5. `05_cost_optimization.ipynb` - Free tier maximization

```bash
# Launch Jupyter
pip install jupyter
jupyter notebook examples/notebooks/
```

---

**Congratulations! You're now ready to use the Pharmaceutical RAG Template.** 🎉

For production deployment, see [Deployment Guide](DEPLOYMENT.md).

For advanced features and customization, see [Configuration Guide](CONFIGURATION.md).
