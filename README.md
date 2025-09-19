# 🏛️ RAG Template for NVIDIA NemoRetriever

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-API-green)](https://build.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful **Retrieval-Augmented Generation (RAG)** template built with **NVIDIA's embedding models** and **LangChain**. This template provides a complete solution for building AI-powered document Q&A systems with a beautiful web interface.

## 🌟 **Features**

- 🤖 **NVIDIA AI Integration**: Uses NVIDIA's high-quality embedding models
- 📄 **PDF Document Processing**: Automatic loading and intelligent chunking
- 🔍 **Vector Search**: FAISS-based similarity search with persistence
- 💬 **Interactive Web UI**: Beautiful Streamlit interface with chat functionality
- 📊 **Advanced Analytics**: Document statistics and source visualization
- 🧪 **PubMed Study Ranking**: Optional pharmaceutical study ranking (enable with `rank=True` or `ENABLE_STUDY_RANKING=true`)
- 🔒 **Secure**: Environment-based API key management
- 📱 **Responsive**: Mobile-friendly design
- 🚀 **Production Ready**: Comprehensive error handling and logging

## 🎯 **Perfect For**

- Legal document analysis
- Research paper Q&A systems
- Corporate knowledge bases
- Educational content exploration
- Technical documentation search
- Any domain-specific document collection

## 📋 **Prerequisites**

### **System Requirements**
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space

### **Required Accounts**
- **NVIDIA Developer Account**: For API access to embedding models
- **Git**: For cloning the repository

## 🔑 **Getting NVIDIA API Key**

### **Step 1: Create NVIDIA Developer Account**
1. Visit [build.nvidia.com](https://build.nvidia.com)
2. Click **"Sign Up"** or **"Log In"** if you have an account
3. Complete the registration process
4. Verify your email address

### **Step 2: Generate API Key**
1. After logging in, navigate to your **Dashboard**
2. Click on **"API Keys"** or **"Credentials"**
3. Click **"Generate New API Key"**
4. Give your key a descriptive name (e.g., "RAG-Template-Key")
5. **Copy and save** the API key securely
6. ⚠️ **Important**: Save this key immediately - you won't be able to see it again!

### **Step 3: Verify API Access**
1. Ensure you have access to:
   - **Embedding Models**: `nvidia/nv-embed-v1`
   - **LLM Models**: `meta/llama-3.1-8b-instruct`
2. Check the [NVIDIA API documentation](https://docs.api.nvidia.com) for current model availability

## 🚀 **Quick Start Guide**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever.git
cd RAG-Template-for-NVIDIA-nemoretriever
```

### **Step 2: Set Up Python Environment**

#### **Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

### **Step 3: Install Dependencies**
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install medical safety validation dependencies (for medical/pharmaceutical applications)
# These include advanced PII/PHI detection using Presidio and biomedical NLP with scispaCy
# To enable medical guardrails, run:
pip install -r requirements.txt && pip install -r requirements-medical.txt
```

**Medical Dependencies Details:**
- `presidio-analyzer` and `presidio-anonymizer`: Advanced PII/PHI detection and anonymization
- `spacy`: Industrial-strength NLP library
- `scispacy`: Biomedical NLP models built on spaCy
- `transformers`: State-of-the-art NLP models from Hugging Face

When medical dependencies are installed, the system automatically uses Presidio for more accurate PII/PHI detection. Otherwise, it falls back to regex-based detection.

**To enable medical guardrails**: Set `ENABLE_MEDICAL_GUARDRAILS=true` and run `pip install -r requirements-medical.txt` after the base install.

### **Step 4: Configure Environment**
1. **Copy the environment template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit the `.env` file** with your details:
   ```bash
   # Open .env file in your preferred editor
   notepad .env  # Windows
   nano .env     # Linux/macOS
   ```

3. **Add your NVIDIA API key**:
   ```env
   # NVIDIA API Configuration
   NVIDIA_API_KEY=your_nvidia_api_key_here
   
   # Configuration (optional - defaults provided)
   DOCS_FOLDER=Data/Docs
   VECTOR_DB_PATH=./vector_db
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   # Guardrails actions (optional)
   # Set only if you host a remote actions server; leave unset to use bundled actions
   ACTIONS_SERVER_URL=http://localhost:8001
   ```

### **Step 5: Add Your Documents**
1. **Create the documents folder** (if not exists):
   ```bash
   mkdir -p Data/Docs
   ```

2. **Add your PDF files** to the `Data/Docs` folder:
   - Copy your PDF documents into this folder
   - The system will automatically process all PDF files
   - Supported formats: `.pdf`

### **Step 6: Test the System**
```bash
# Run the test suite
python test_rag_system.py
```

### **Step 7: Launch the Web Interface**
```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py
```

Or use the convenient launcher:
```bash
python start_web_interface.py
```

### **Step 8: Access Your RAG System**
1. **Open your browser** and navigate to:
   - **Local**: `http://localhost:8501`
   - **Network**: `http://[your-ip]:8501`

2. **Wait for initialization**:
   - The system will load and process your documents
   - This may take a few minutes for large document collections

3. **Start asking questions**!
   - Type your questions in the chat interface
   - Explore the document statistics
   - View detailed source references

## 📁 **Project Structure**

```
RAG-Template-for-NVIDIA-nemoretriever/
├── 📄 README.md                     # This comprehensive guide
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env.template                 # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
├── 📄 streamlit_app.py             # Main web interface
├── 📄 main.py                      # CLI interface
├── 📄 test_rag_system.py           # System tests
├── 📄 start_web_interface.py       # Web interface launcher
├── 📁 src/                         # Core source code
│   ├── 📄 __init__.py
│   ├── 📄 document_loader.py       # PDF processing
│   ├── 📄 nvidia_embeddings.py     # NVIDIA API integration
│   ├── 📄 vector_database.py       # FAISS vector storage
│   └── 📄 rag_agent.py             # Main RAG pipeline
├── 📁 .streamlit/                  # Streamlit configuration
│   └── 📄 config.toml
├── 📁 Data/                        # Document storage
│   └── 📁 Docs/                    # Place your PDF files here
└── 📁 vector_db/                   # Vector database (auto-created)
```

## 🎮 **Usage Examples**

### **Command Line Interface**
```bash
# Use the CLI version
python main.py
```

### **Web Interface**
1. **Start the web interface**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Sample questions to try**:
   - "What is the main topic of the documents?"
   - "Summarize the key points from [specific document]"
   - "What are the requirements for [specific process]?"
   - "How does [concept A] relate to [concept B]?"

### **Programmatic Usage**
```python
from src.rag_agent import RAGAgent
from src.pubmed_scraper import PubMedScraper
import os

# Initialize RAG agent
api_key = os.getenv("NVIDIA_API_KEY")
rag_agent = RAGAgent("Data/Docs", api_key)

# Setup knowledge base
rag_agent.setup_knowledge_base()

# Ask questions
response = rag_agent.ask_question("Your question here")
print(response.answer)
print(f"Sources: {len(response.source_documents)}")
if response.disclaimer:
    print(response.disclaimer)

# Preserve raw PubMed ordering when fetching articles programmatically
scraper = PubMedScraper()
articles = scraper.search_pubmed("oncology pharmacology", rank=False)
print(f"Fetched {len(articles)} articles in canonical crawl order")

# When preserving order (rank=False), EasyAPI abstracts are omitted unless ranking is enabled to reduce latency/cost.
```

## 🔧 **Configuration Options**

### **Environment Variables**
| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | Your NVIDIA API key | **Required** |
| `DOCS_FOLDER` | Path to PDF documents | `Data/Docs` |
| `VECTOR_DB_PATH` | Vector database storage | `./vector_db` |
| `VECTOR_DB_PER_MODEL` | Store vector indexes in per-model subdirectories; auto-migrates compatible legacy data and rebuilds on incompatibilities | `false` |
| `DISABLE_PREFLIGHT_EMBEDDING` | Set to `true` to skip the per-question preflight embedding (lower latency/cost) at the expense of weaker detection when the runtime falls back to a different model | `false` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |
| `DRUG_GENERIC_LEXICON` | Optional path to newline-delimited generic drug names | `data/drugs_generic.txt` if present |
| `DRUG_BRAND_LEXICON` | Optional path to newline-delimited brand drug names | `data/drugs_brand.txt` if present |
| `ENABLE_MEDICAL_GUARDRAILS` | Enable medical safety validation features (requires additional dependencies) | `false` |

### **Medical Safety Features**
This template includes optional medical safety validation features that can be enabled for pharmaceutical and healthcare applications. These features provide:

- PII/PHI detection and anonymization using Presidio
- Medical context validation
- Regulatory compliance checking
- Advanced biomedical NLP pipelines using SciSpaCy

To use these features:
1. Install the medical dependencies: `pip install -r requirements-medical.txt`
2. Set `ENABLE_MEDICAL_GUARDRAILS=true` in your environment

### **Customization Options**
- **Chunk Size**: Adjust for different document types
- **Model Selection**: Switch between available NVIDIA models
- **UI Styling**: Modify Streamlit interface in `streamlit_app.py`
- **Processing Logic**: Customize RAG pipeline in `src/rag_agent.py`

### **Vector Database Management**
- **`VECTOR_DB_PER_MODEL`**: When set to `true`, FAISS indexes live in sanitized per-model folders (e.g., `./vector_db/nvidia_llama-3.2-nemoretriever-1b-vlm-embed-v1`). Leaving it `false` keeps the legacy single-folder layout in `VECTOR_DB_PATH`.
- **`embeddings_meta.json`**: Saved next to each index. It records the embedding model name and vector dimension so the agent can detect mismatches before loading older data.
- **Rebuilds & migrations**: On first run with per-model paths, compatible legacy data is migrated automatically; mismatched metadata or dimension changes trigger a rebuild at the reconciled path. Runtime fallback to another embedding model updates the vector DB base path, and logs will tell you to rerun `setup_knowledge_base(force_rebuild=True)` when a rebuild is the safest choice.

### **PubMed Scraper CLI**
Run the scraper directly to fetch and cache PubMed results:

```bash
# Module execution (original behavior)
python -m src.pubmed_scraper "metformin glycemic control" --max-items 40 --export-sidecars ./Data/Docs

# Preserve canonical PubMed ordering when ranking isn't desired
python -m src.pubmed_scraper "metformin glycemic control" --no-rank

# Direct script execution is also supported
python src/pubmed_scraper.py "metformin glycemic control"

# Environment toggle for repeated runs without CLI flags
PRESERVE_PUBMED_ORDER=true python -m src.pubmed_scraper "metformin glycemic control"
```

The CLI prints the number of articles found, cache directory statistics, and a preview of the first few results. When `--export-sidecars` is supplied, `.pubmed.json` files are created next to matching PDFs in the specified folder.

## 📚 **PubMed Workflow**

This template provides seamless integration with PubMed for academic and research workflows. Follow these steps to scrape PubMed articles and integrate them with your RAG system:

### **Step 1: Configure APIFY_TOKEN**
1. **Sign up for Apify**: Visit [https://console.apify.com](https://console.apify.com)
2. **Get your API token**: Navigate to Settings → Integrations → API tokens
3. **Add to environment**: Set `APIFY_TOKEN=your_apify_token_here` in your `.env` file

### **Step 2: Scrape PubMed Articles**
```bash
# Scrape articles and export sidecar metadata files
python -m src.pubmed_scraper "diabetes treatment" --export-sidecars Data/Docs

# Advanced usage with more articles
python -m src.pubmed_scraper "cancer immunotherapy" --max-items 50 --export-sidecars Data/Docs

# Preserve original PubMed ordering (no ranking)
python -m src.pubmed_scraper "covid-19 treatment" --no-rank --export-sidecars Data/Docs
```

### **Step 3: Build and Query Knowledge Base**
```bash
# Launch the RAG agent to build knowledge base from documents + sidecars
python -m src.rag_agent

# Or use the web interface
streamlit run streamlit_app.py
```

### **Key Features**
- **Automatic Metadata**: PubMed metadata (DOI, PMID, authors, MeSH terms) is automatically extracted
- **Sidecar Integration**: `.pubmed.json` files provide rich metadata alongside PDFs
- **Deduplication**: Automatic removal of duplicate articles by DOI/PMID and title
- **Medical Disclaimers**: Built-in medical disclaimers for healthcare applications

### **Environment Examples**
The `.env.template` file includes comprehensive PubMed configuration options:
- `APIFY_TOKEN`: Your Apify API token for PubMed scraping
- `PUBMED_CACHE_DIR`: Local cache directory for results
- `ENABLE_STUDY_RANKING`: Enable/disable automatic study quality ranking
- `ENABLE_DEDUPLICATION`: Control duplicate article removal

## 🧪 **Testing**

### **Run All Tests**
```bash
python test_rag_system.py
```

### **Test Individual Components**
```bash
# Test NVIDIA embeddings
python src/nvidia_embeddings.py

# Test document loader
python src/document_loader.py

# Test vector database
python src/vector_database.py

# Test RAG agent
python src/rag_agent.py
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. API Connection Failed**
```
❌ NVIDIA API connection failed
```
**Solutions**:
- Verify your API key is correct in `.env`
- Check internet connection
- Ensure API key has proper permissions
- Visit [NVIDIA API status page](https://status.nvidia.com)

#### **2. No Documents Found**
```
❌ No PDF files found in Data/Docs
```
**Solutions**:
- Ensure PDF files are in `Data/Docs` folder
- Check file permissions
- Verify files are valid PDFs

#### **3. Memory Issues**
```
❌ Out of memory during processing
```
**Solutions**:
- Reduce `CHUNK_SIZE` in `.env`
- Process fewer documents at once
- Increase system RAM

#### **4. Import Errors**
```
❌ ModuleNotFoundError: No module named 'xyz'
```
**Solutions**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version compatibility

### **Getting Help**
1. **Check the logs**: Look for detailed error messages in console output
2. **Run tests**: Use `python test_rag_system.py` to diagnose issues
3. **Verify setup**: Ensure all prerequisites are met
4. **Check documentation**: Review NVIDIA API documentation

## 🚀 **Deployment Options**

### **Local Development**
- Use the provided scripts for local testing and development

### **Docker Deployment**
```dockerfile
# Example Dockerfile (create as needed)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

### **Cloud Deployment**
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use the provided configuration
- **AWS/GCP/Azure**: Deploy using container services

## 🤝 **Contributing**

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/yourusername/RAG-Template-for-NVIDIA-nemoretriever.git

# Install development dependencies
pip install -r requirements.txt

# Run tests before committing
python test_rag_system.py
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **NVIDIA** for providing excellent AI models and APIs
- **LangChain** for the RAG framework
- **Streamlit** for the beautiful web interface framework
- **FAISS** for efficient vector search capabilities

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever/discussions)
- **Documentation**: Check the `docs/` folder for additional guides

---

**🎉 Happy Building! Create amazing RAG applications with NVIDIA's powerful AI models! 🚀**
