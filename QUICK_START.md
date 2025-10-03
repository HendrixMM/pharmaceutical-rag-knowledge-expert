# 🚀 Quick Start Guide

Get up and running with the RAG Template for NVIDIA NemoRetriever in **5 minutes**.

## Prerequisites

- Python 3.8 or higher
- NVIDIA API key ([get one here](https://build.nvidia.com))
- 2GB free disk space

## ⚡ 5-Minute Setup

### Step 1: Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/your-org/RAG-Template-for-NVIDIA-nemoretriever.git
cd RAG-Template-for-NVIDIA-nemoretriever

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment (1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your NVIDIA API key
# Minimum required:
#   NVIDIA_API_KEY=nvapi-your-actual-key-here
```

**Get your NVIDIA API key:**
1. Visit [build.nvidia.com](https://build.nvidia.com)
2. Sign in/create account
3. Navigate to API Keys
4. Copy your key

### Step 3: Validate Setup (30 seconds)

```bash
# Verify configuration
python scripts/validate_env.py

# Expected output:
# ✅ Environment validation passed!
```

### Step 4: Add Documents (1 minute)

```bash
# Add your PDF documents
cp your_documents/*.pdf Data/Docs/

# Or use example docs (pharmaceutical research papers)
# Already included in Data/Docs/
```

### Step 5: Launch! (30 seconds)

```bash
# Start the web interface
streamlit run streamlit_app.py

# Or use CLI mode
python main.py "What are the drug interactions?"
```

**Web UI will open at:** http://localhost:8501

---

## 🎯 Your First Query

### Using Web Interface

1. Open http://localhost:8501
2. Enter your question: "What are the main findings?"
3. Click "Submit" or press Enter
4. View AI-powered answers with source citations

### Using CLI

```bash
# Basic query
python main.py "What are the pharmacokinetics of aspirin?"

# With filters
python main.py "Drug interactions" --drug aspirin --species human

# Save results
python main.py "Clinical trials" --output results.json
```

---

## 🔧 Common Configurations

### Minimal Setup (What you just did)

```env
NVIDIA_API_KEY=nvapi-your-key-here
DOCS_FOLDER=Data/Docs
```

### Pharmaceutical Research

```env
NVIDIA_API_KEY=nvapi-your-key-here
NEMO_PHARMACEUTICAL_ANALYSIS=true
PHARMA_DOMAIN_OVERLAY=true
ENABLE_MEDICAL_GUARDRAILS=true
```

### PubMed Integration

```env
NVIDIA_API_KEY=nvapi-your-key-here
PUBMED_EMAIL=your.email@example.com
PUBMED_EUTILS_API_KEY=your-pubmed-key  # Optional
ENABLE_RAG_PUBMED_INTEGRATION=true
```

---

## 📚 Next Steps

### 1. Explore Features

- **Medical Guardrails**: Automatic safety validation
- **PubMed Search**: Live literature integration
- **Drug Filtering**: Species-specific research
- **Benchmarking**: Performance validation

### 2. Advanced Configuration

- [NGC Deprecation Immunity](docs/NGC_DEPRECATION_IMMUNITY.md)
- [NVIDIA Model Access Guide](docs/NVIDIA_MODEL_ACCESS_GUIDE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Pharmaceutical Best Practices](docs/PHARMACEUTICAL_BEST_PRACTICES.md)

### 3. Development Setup

```bash
# Install development tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guide.

---

## 🐛 Troubleshooting

### Issue: "NVIDIA_API_KEY appears to be a placeholder"

**Solution:**
```bash
# Edit .env and replace placeholder
vim .env  # Change your_nvidia_api_key_here to real key
python scripts/validate_env.py  # Verify
```

### Issue: "No module named 'streamlit'"

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt
```

### Issue: "No documents found"

**Solution:**
```bash
# Check documents directory
ls -la Data/Docs/

# Add PDF files
cp path/to/your/*.pdf Data/Docs/
```

### Issue: API Rate Limit

**Solution:**
```bash
# Enable rate limiting
echo "NVIDIA_RATE_LIMIT_AWARE=true" >> .env
echo "MAX_REQUESTS_PER_SECOND=3" >> .env
```

### Issue: SSL Certificate Error

**Solution:**
```bash
# Usually due to corporate proxy
export CURL_CA_BUNDLE=/path/to/ca-bundle.crt
# Or disable verification (not recommended for production)
export PYTHONHTTPSVERIFY=0
```

---

## 💡 Tips & Tricks

### Performance

```bash
# Enable GPU acceleration (if available)
pip install -r requirements-nemo.txt

# Use model-specific vector databases
echo "VECTOR_DB_PER_MODEL=true" >> .env
```

### Cost Optimization

```bash
# Maximize free tier
echo "NVIDIA_BUILD_FREE_TIER=true" >> .env
echo "NVIDIA_CREDITS_MONITORING=true" >> .env

# See: docs/FREE_TIER_MAXIMIZATION.md
```

### Pharmaceutical Research

```bash
# Enable all pharmaceutical features
cat >> .env << 'EOF'
NEMO_PHARMACEUTICAL_ANALYSIS=true
PHARMA_DOMAIN_OVERLAY=true
PHARMACEUTICAL_FEATURE_DRUG_INTERACTION_ANALYSIS=true
PHARMACEUTICAL_FEATURE_CLINICAL_TRIAL_PROCESSING=true
PHARMACEUTICAL_FEATURE_PHARMACOKINETICS_OPTIMIZATION=true
EOF
```

---

## 🆘 Getting Help

- **Documentation**: Check `/docs` directory
- **Issues**: [GitHub Issues](https://github.com/your-org/RAG-Template-for-NVIDIA-nemoretriever/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/RAG-Template-for-NVIDIA-nemoretriever/discussions)
- **Security**: See [SECURITY.md](SECURITY.md)

---

## ✅ Checklist

After completing this guide, you should have:

- [x] Installed dependencies
- [x] Configured `.env` with NVIDIA API key
- [x] Validated environment
- [x] Added documents
- [x] Run your first query
- [x] Explored web interface

**What's Next?**
- Explore [pharmaceutical features](docs/PHARMACEUTICAL_BEST_PRACTICES.md)
- Set up [PubMed integration](docs/API_INTEGRATION_GUIDE.md)
- Learn about [NGC deprecation immunity](docs/NGC_DEPRECATION_IMMUNITY.md)
- Read the [full README](README.md)

---

**Time to production**: 5 minutes ✓
**Difficulty**: Beginner-friendly ✓
**Next**: [Full Documentation](README.md)
