# 🚀 Complete Setup Guide - RAG Template for NVIDIA NemoRetriever

This guide provides detailed, step-by-step instructions for setting up and running the RAG template.

## 📋 **Pre-Setup Checklist**

Before you begin, ensure you have:
- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] NVIDIA Developer Account created
- [ ] NVIDIA API key obtained
- [ ] PDF documents ready for processing

## 🔑 **Detailed NVIDIA API Key Setup**

### **Creating Your NVIDIA Account**

1. **Visit NVIDIA Build Platform**
   - Go to [build.nvidia.com](https://build.nvidia.com)
   - Click "Sign Up" in the top right corner

2. **Complete Registration**
   - Fill in your details (name, email, password)
   - Verify your email address
   - Accept the terms of service

3. **Account Verification**
   - Check your email for verification link
   - Click the verification link
   - Log in to your new account

### **Generating Your API Key**

1. **Access Dashboard**
   - After logging in, go to your dashboard
   - Look for "API Keys" or "Credentials" section

2. **Create New Key**
   - Click "Generate New API Key" or "Create API Key"
   - Give it a descriptive name: "RAG-Template-Project"
   - Select appropriate permissions (if asked)

3. **Save Your Key**
   - **IMPORTANT**: Copy the key immediately
   - Store it in a secure location (password manager recommended)
   - You won't be able to see it again!

4. **Test Access**
   - Verify you can access the models:
     - `nvidia/nv-embed-v1` (embedding model)
     - `meta/llama-3.1-8b-instruct` (language model)

## 💻 **Step-by-Step Installation**

### **Step 1: Clone the Repository**

```bash
# Clone the repository
git clone https://github.com/zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever.git

# Navigate to the project directory
cd RAG-Template-for-NVIDIA-nemoretriever

# Verify the clone was successful
ls -la
```

### **Step 2: Python Environment Setup**

#### **Option A: Virtual Environment (Recommended)**

```bash
# Create virtual environment
python -m venv rag_env

# Activate the environment
# Windows:
rag_env\Scripts\activate
# macOS/Linux:
source rag_env/bin/activate

# Verify activation (should show rag_env in prompt)
which python  # Should point to rag_env
```


### **Step 3: Install Dependencies**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Optional: Install medical safety validation dependencies (for medical/pharmaceutical applications)
# pip install -r requirements-medical.txt

# Verify installation
pip list | grep -E "(streamlit|langchain|faiss|requests)"
```

### **Step 4: Environment Configuration**

1. **Create Environment File**
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Verify the file was created
   ls -la .env
   ```

2. **Edit the Environment File**
   
   **Windows (using Notepad):**
   ```bash
   notepad .env
   ```
   
   **macOS/Linux (using nano):**
   ```bash
   nano .env
   ```
   
   **Or use any text editor:**
   ```bash
   code .env  # VS Code
   vim .env   # Vim
   ```

3. **Configure Your Settings**
   ```env
   # NVIDIA API Configuration
   NVIDIA_API_KEY=nvapi-your-actual-api-key-here
   
   # Optional Configuration (defaults shown)
   DOCS_FOLDER=Data/Docs
   VECTOR_DB_PATH=./vector_db
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

### **Step 5: Document Preparation**

1. **Verify Folder Structure**
   ```bash
   # Check if Data/Docs exists
   ls -la Data/Docs
   ```

2. **Add Your PDF Documents**
   - Copy your PDF files to `Data/Docs/` folder
   - Ensure files are valid PDFs
   - Check file permissions (readable)

3. **Verify Document Setup**
   ```bash
   # List PDF files
   ls -la Data/Docs/*.pdf
   
   # Count PDF files
   ls Data/Docs/*.pdf | wc -l
   ```

### **Step 6: System Testing**

1. **Run Comprehensive Tests**
   ```bash
   # Run all system tests
   python test_rag_system.py
   ```

2. **Test Individual Components**
   ```bash
   # Test NVIDIA API connection
   python src/nvidia_embeddings.py
   
   # Test document loading
   python src/document_loader.py
   
   # Test vector database
   python src/vector_database.py
   
   # Test RAG agent
   python src/rag_agent.py
   ```

### **Step 7: Launch the Application**

#### **Option A: Web Interface (Recommended)**

```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py

# Or use the launcher script
python start_web_interface.py
```

### **Step 8: Access and Test**

1. **Open Web Interface**
   - Open browser to `http://localhost:8501`
   - Wait for system initialization
   - Check system status in sidebar

2. **Test with Sample Questions**
   - "What is the main topic of the documents?"
   - "Summarize the key points"
   - "What are the requirements mentioned?"

## 🏥 **Medical Safety Features Setup (Optional)**

This template includes optional medical safety validation features for pharmaceutical and healthcare applications. These features provide:

- PII/PHI detection and anonymization using Presidio
- Medical context validation
- Regulatory compliance checking
- Advanced biomedical NLP pipelines using SciSpaCy

### **Installing Medical Dependencies**

For automated CI/production deployments, use conditional installation:

```bash
# Install base requirements first
pip install -r requirements.txt

# Conditionally install medical dependencies if enabled
if [ "${ENABLE_MEDICAL_GUARDRAILS}" = "true" ]; then
  pip install -r requirements-medical.txt
  # Download required SpaCy model for SciSpaCy
  python -m spacy download en_core_web_sm
fi
```

For manual setup:

```bash
# Install medical safety validation dependencies
pip install -r requirements-medical.txt

# Download required SpaCy model for SciSpaCy
python -m spacy download en_core_web_sm
```

### **Enabling Medical Features**

1. **Set Environment Variable**
   ```bash
   # In your .env file, add:
   ENABLE_MEDICAL_GUARDRAILS=true
   ```

2. **Restart the Application**
   ```bash
   # Stop the current application (Ctrl+C)
   # Restart the web interface
   streamlit run streamlit_app.py
   ```

3. **Verify Medical Features**
   - Check logs for medical guardrails initialization messages
   - Test with medical-related questions to see enhanced safety features

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: Python Version Problems**
```bash
# Check Python version
python --version

# If version is < 3.8, install newer Python
# Windows: Download from python.org
# macOS: brew install python@3.9
# Linux: sudo apt install python3.9
```

### **Issue 2: Package Installation Failures**
```bash
# Clear pip cache
pip cache purge

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install packages one by one if batch fails
pip install streamlit
pip install langchain
pip install faiss-cpu
```

### **Issue 3: API Key Issues**
```bash
# Verify .env file exists and has content
cat .env

# Check for hidden characters
hexdump -C .env | head

# Test API key manually
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('NVIDIA_API_KEY')[:10] + '...')
"
```

### **Issue 4: Document Loading Problems**
```bash
# Check file permissions
ls -la Data/Docs/

# Verify PDF files are valid
file Data/Docs/*.pdf

# Test with a single PDF first
mkdir test_docs
cp "Data/Docs/your-file.pdf" test_docs/
# Update DOCS_FOLDER in .env to test_docs
```

### **Issue 5: Memory Issues**
```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# Reduce chunk size in .env
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

## 🎯 **Verification Checklist**

After setup, verify these items:

- [ ] Python environment is activated
- [ ] All packages installed successfully
- [ ] .env file contains valid API key
- [ ] PDF documents are in Data/Docs folder
- [ ] System tests pass
- [ ] Web interface loads at localhost:8501
- [ ] Can ask questions and get responses
- [ ] Source documents are displayed
- [ ] No error messages in console

## 📞 **Getting Help**

If you encounter issues:

1. **Check the logs** for detailed error messages
2. **Run the test suite** to identify specific problems
3. **Verify prerequisites** are met
4. **Check GitHub Issues** for similar problems
5. **Create a new issue** with detailed error information

## 🎉 **Success!**

Once everything is working:
- Bookmark the web interface URL
- Try different types of questions
- Explore the document statistics
- Customize the system for your needs

**You're now ready to build amazing RAG applications with NVIDIA's AI models!** 🚀
