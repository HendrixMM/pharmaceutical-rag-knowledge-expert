"""
DIFC Legal RAG - Streamlit Web Interface
Beautiful, user-friendly web interface for the DIFC Legal RAG system
"""

import streamlit as st
import sys
import os
import time
from typing import Any, Dict
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.enhanced_rag_agent import EnhancedRAGAgent
from src.enhanced_config import EnhancedRAGConfig
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.pharmaceutical_query_adapter import build_enhanced_rag_agent

# Page configuration
st.set_page_config(
    page_title="RAG Assistant - NVIDIA NemoRetriever",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #7b1fa2;
    }
    
    .source-card {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #4caf50;
    }
    
    .status-offline {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_agent():
    """Initialize the RAG agent (cached for performance)"""
    try:
        load_dotenv()

        # Load enhanced configuration
        config = EnhancedRAGConfig.from_env()

        api_key = os.getenv("NVIDIA_API_KEY")
        docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
        vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
        guardrails_config = os.getenv("MEDICAL_GUARDRAILS_CONFIG")
        enable_synthesis = os.getenv("ENABLE_SYNTHESIS", "true").strip().lower() in ("1", "true", "yes", "on")
        enable_ddi = os.getenv("ENABLE_DDI_ANALYSIS", "true").strip().lower() in ("1", "true", "yes", "on")
        safety_mode = os.getenv("ENHANCED_RAG_SAFETY_MODE", "balanced")

        # Production guard for NeMo strict configuration
        app_env = (os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
        if app_env in ("production", "prod"):
            strategy = (os.getenv("NEMO_EXTRACTION_STRATEGY") or "nemo").strip().lower()
            enable_nemo = (os.getenv("ENABLE_NEMO_EXTRACTION") or "true").strip().lower() in ("true","1","yes","on")
            strict = (os.getenv("NEMO_EXTRACTION_STRICT") or "true").strip().lower() in ("true","1","yes","on")
            if (strategy != "nemo") or (not enable_nemo) or (not strict):
                st.warning("Production requires NeMo strict: ENABLE_NEMO_EXTRACTION=true, NEMO_EXTRACTION_STRATEGY=nemo, NEMO_EXTRACTION_STRICT=true")

        if not api_key:
            st.error("❌ NVIDIA_API_KEY not found in environment variables")
            st.info("Please check your .env file and ensure the API key is set correctly.")
            return None

        # Test NVIDIA API connection first
        with st.spinner("🔌 Testing NVIDIA API connection..."):
            try:
                embeddings = NVIDIAEmbeddings(api_key)
                if not embeddings.test_connection():
                    st.error("❌ Failed to connect to NVIDIA API")
                    return None
            except Exception as e:
                st.error(f"❌ NVIDIA API connection failed: {str(e)}")
                return None

        # Initialize RAG agent with enhanced configuration using factory method
        with st.spinner("🤖 Initializing Enhanced RAG Agent..."):
            rag_agent = build_enhanced_rag_agent(
                docs_folder=docs_folder,
                api_key=api_key,
                vector_db_path=vector_db_path,
                guardrails_config_path=guardrails_config,
                enable_synthesis=enable_synthesis,
                enable_ddi_analysis=enable_ddi,
                safety_mode=safety_mode,
                config=config,
                append_disclaimer_in_answer=True,
            )

        # Setup knowledge base
        with st.spinner("📚 Loading knowledge base..."):
            if rag_agent.base_agent.setup_knowledge_base():
                st.success("✅ Enhanced RAG system initialized successfully!")
                if config.should_enable_pubmed():
                    st.info("🔬 PubMed integration enabled - Hybrid search available!")
                return rag_agent
            else:
                st.error("❌ Failed to setup knowledge base")
                st.info("Please ensure PDF files are in the Data/Docs folder.")
                return None

    except Exception as e:
        st.error(f"❌ Failed to initialize Enhanced RAG agent: {str(e)}")
        st.exception(e)
        return None


@st.cache_resource
def get_shared_credit_tracker():
    """Return a shared PharmaceuticalCreditTracker instance (cached)."""
    try:
        from src.monitoring.credit_tracker import PharmaceuticalCreditTracker
        return PharmaceuticalCreditTracker()
    except Exception:
        return None


def fetch_credit_burn_snapshot():
    """Fetch a compact snapshot of credits and burn metrics for UI display.

    Returns a dict with keys: used_today (int), burn_rate (float), service_counts (dict)
    or an empty dict when unavailable.
    """
    try:
        tracker = get_shared_credit_tracker()
        if tracker is None:
            return {}
        analytics = tracker.get_pharmaceutical_analytics()
        base = analytics.get("base_monitor_summary", {}) if isinstance(analytics, dict) else {}
        burn = analytics.get("daily_burn") if isinstance(analytics, dict) else None
        out = {}
        if isinstance(burn, dict):
            used_today = burn.get("used_today")
            burn_rate = burn.get("burn_rate")
            if used_today is not None and burn_rate is not None:
                out["used_today"] = used_today
                out["burn_rate"] = float(burn_rate)
        svc = base.get("by_service") if isinstance(base, dict) else None
        if isinstance(svc, dict):
            out["service_counts"] = svc
        return out
    except Exception:
        return {}

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Assistant - NVIDIA NemoRetriever</h1>
        <p>AI-Powered Document Q&A System with Advanced Retrieval</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(rag_agent):
    """Display the sidebar with system information"""
    st.sidebar.markdown("## 📊 System Status")
    
    if rag_agent:
        # System status
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="status-indicator status-online"></span>
            <strong>System Online</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Get knowledge base stats
        stats = rag_agent.base_agent.get_knowledge_base_stats()
        
        st.sidebar.markdown("### 📚 Knowledge Base")
        st.sidebar.metric("Documents", stats.get('document_count', 0))
        st.sidebar.metric("PDF Files", stats.get('pdf_files_available', 0))
        
        # Model information
        st.sidebar.markdown("### 🤖 AI Models")
        st.sidebar.info("**Embedding**: nvidia/nv-embed-v1\n**LLM**: meta/llama-3.1-8b-instruct")

        # Safety metrics snapshot
        safety_metrics = rag_agent.get_safety_metrics()
        st.sidebar.markdown("### 🛡️ Safety Metrics")
        st.sidebar.metric("Checks", safety_metrics.get("total_queries", 0))
        st.sidebar.metric("Blocked", safety_metrics.get("blocked_queries", 0))
        st.sidebar.caption(f"Mode: {safety_metrics.get('safety_mode', 'balanced').title()}")

        # Pharma mode and rerank retry/backoff policy
        st.sidebar.markdown("### 💊 Pharma & Retry Policy")
        try:
            cfg = rag_agent.config
            pharma_mode = bool(getattr(cfg, "pharmaceutical_research_mode", True))
            backoff_base = getattr(cfg, "rerank_retry_backoff_base", 0.5)
            retry_attempts = int(getattr(cfg, "rerank_retry_max_attempts", 3))
            jitter_enabled = bool(getattr(cfg, "rerank_retry_jitter", True))
            st.sidebar.caption(f"Pharma Mode: {'On' if pharma_mode else 'Off'}")
            st.sidebar.caption(f"Backoff Base: {backoff_base}s | Retries: {retry_attempts}")
            st.sidebar.caption(f"Jitter: {'On' if jitter_enabled else 'Off'}")
        except Exception:
            st.sidebar.caption("Pharma policy info unavailable")

        # Credits & burn (best-effort, uses shared cached tracker)
        st.sidebar.markdown("### 🧾 Credits & Burn")
        try:
            snap = fetch_credit_burn_snapshot()
            if not snap:
                st.sidebar.caption("No credit usage recorded yet this session")
            else:
                if "used_today" in snap and "burn_rate" in snap:
                    st.sidebar.metric("Used Today", snap["used_today"])
                    st.sidebar.metric("Burn Rate", f"{snap['burn_rate']:.2f}x/day budget")
                for_svc = snap.get("service_counts") or {}
                if for_svc:
                    st.sidebar.caption("Service usage (month):")
                    for name, cnt in for_svc.items():
                        st.sidebar.caption(f"• {name}: {cnt}")
        except Exception:
            st.sidebar.caption("Credit monitor unavailable")

        # PubMed integration status
        if rag_agent.config.should_enable_pubmed():
            st.sidebar.markdown("### 🔬 PubMed Integration")
            pubmed_status = rag_agent.get_system_status().get("pubmed", {})
            pubmed_metrics = pubmed_status.get("metrics", {})

            # Status indicator
            pubmed_enabled = pubmed_status.get("enabled", False)
            if pubmed_enabled:
                st.sidebar.success("🟢 PubMed Enabled")

                # Display metrics
                st.sidebar.metric("Queries", pubmed_metrics.get("total_queries", 0))
                st.sidebar.metric("Cache Hits", pubmed_metrics.get("cache_hits", 0))
                last_provider = pubmed_metrics.get("last_provider") or "eutils/openalex"
                last_latency = pubmed_metrics.get("last_latency_ms")
                st.sidebar.caption(f"Provider: {last_provider}")
                if isinstance(last_latency, (int, float)):
                    st.sidebar.caption(f"Last latency: {int(last_latency)} ms")

                # Show cache status if available
                component_health = rag_agent.component_health.get("pubmed_integration", {})
                cache_status = component_health.get("cache", {})
                if cache_status:
                    cache_status_text = cache_status.get("status", "unknown")
                    if cache_status_text == "ready":
                        st.sidebar.success("📦 Cache Ready")
                    else:
                        st.sidebar.warning("⚠️ Cache Issue")

                # Show rate limit status
                rate_limit_status = component_health.get("rate_limit", {})
                if rate_limit_status:
                    if rate_limit_status.get("status") == "ready":
                        st.sidebar.info("⏱️ Rate Limit OK")
                    else:
                        st.sidebar.warning("⏱️ Rate Limit Active")

                # Configuration summary
                config_flags = rag_agent.config.summarize_flags()
                if config_flags.get("hybrid"):
                    st.sidebar.info("🔄 Hybrid Mode")
            else:
                st.sidebar.warning("⚪ PubMed Disabled")

        # Document types supported
        st.sidebar.markdown("### 📖 Document Types Supported")
        doc_types = [
            "PDF Documents", "Research Papers", "Legal Documents",
            "Technical Manuals", "Corporate Policies", "Academic Papers",
            "Training Materials", "Compliance Documents", "Reports",
            "Contracts", "Specifications", "User Guides"
        ]

        for doc_type in doc_types:
            st.sidebar.markdown(f"• {doc_type}")
            
    else:
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="status-indicator status-offline"></span>
            <strong>System Offline</strong>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.error("RAG system not available")

def display_chat_interface(rag_agent):
    """Display the main chat interface"""
    st.markdown("## 💬 Ask Your Question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        initial_message = "Hello! I'm your RAG Assistant powered by NVIDIA NemoRetriever. I can help you find information from your document collection."
        if rag_agent and rag_agent.config.should_enable_pubmed():
            initial_message += "\n\n🔬 **PubMed Integration Available**\nI can also search PubMed for the latest medical research. Try asking about recent studies or clinical trials!"
        st.session_state.messages.append({
            "role": "assistant",
            "content": initial_message,
            "sources": [],
            "processing_time": 0
        })

    # Query mode selection (only if PubMed is enabled)
    if rag_agent and rag_agent.config.should_enable_pubmed():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            query_mode = st.selectbox(
                "🔍 Search Mode:",
                ["Auto (Hybrid)", "Local Documents Only", "PubMed Only"],
                index=0,
                help="Auto: Intelligently combines local documents with PubMed research\nPubMed Only: Searches only PubMed, optionally in strict mode (no fallback to local documents)"
            )
        with col2:
            if query_mode != "Local Documents Only":
                max_pubmed_results = st.slider(
                    "Max PubMed Results:",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Maximum number of PubMed articles to include"
                )
            else:
                max_pubmed_results = 0
        with col3:
            st.markdown("###")
            if rag_agent.config.is_rollout_active():
                rollout_pct = rag_agent.config.rollout_percentage
                st.info(f"🎲 Rollout: {rollout_pct}%")
    else:
        query_mode = "Local Documents Only"
        max_pubmed_results = 0
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if message.get("sources"):
                display_sources(
                    message["sources"],
                    message.get("processing_time", 0),
                    message.get("confidence_scores"),
                )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {prompt}
        </div>
        """, unsafe_allow_html=True)
        
        if rag_agent:
            try:
                # Show loading spinner with mode-specific text
                if query_mode == "PubMed Only":
                    spinner_text = "🔬 Searching PubMed..."
                elif query_mode == "Auto (Hybrid)":
                    spinner_text = "🔄 Searching documents & PubMed..."
                else:
                    spinner_text = "🔍 Searching documents..."

                with st.spinner(spinner_text):
                    # Choose the appropriate query method based on mode
                    if query_mode == "PubMed Only":
                        response_payload = rag_agent.ask_question_pubmed_only(
                            prompt,
                            max_external_results=max_pubmed_results
                        )
                    elif query_mode == "Auto (Hybrid)":
                        response_payload = rag_agent.ask_question_hybrid(
                            prompt,
                            k=5,
                            max_external_results=max_pubmed_results
                        )
                    else:
                        response_payload = rag_agent.ask_question_safe_sync(prompt, k=5)

                if not response_payload:
                    st.error("❌ Failed to get a response. Please try again.")
                    return

                error = response_payload.get("error")
                answer = response_payload.get("answer")
                sources = response_payload.get("sources", [])
                processing_time = response_payload.get("processing_time", 0.0)
                confidence_scores = response_payload.get("confidence_scores")

                if error:
                    # Handle both PubMed-style errors (with 'message') and guardrails-style errors (with 'issues')
                    if isinstance(error, dict):
                        if "message" in error:
                            # PubMed-style error
                            issues = [error["message"]]
                        else:
                            # Guardrails-style error
                            issues = error.get("issues") or [error.get("message", "An error occurred")]
                    else:
                        # String error or other format
                        issues = [str(error)]

                    st.warning("⚠️ " + " ".join(issues))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "\n".join(issues),
                        "sources": [],
                        "processing_time": 0,
                    })

                    # Only show recommendations for guardrails-style errors
                    if isinstance(error, dict) and "recommendations" in error:
                        recommendations = error.get("recommendations")
                        if recommendations:
                            st.info("\n".join(recommendations))
                    return

                if not answer:
                    st.error("❌ The system returned an empty answer. Please try again.")
                    return

                safety_warnings = response_payload.get("safety", {}).get("output_validation", {}).get("warnings")

                # Add assistant response (answer already includes disclaimer when append_disclaimer_in_answer=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "processing_time": processing_time,
                    "confidence_scores": confidence_scores,
                })

                # Display assistant response
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                if safety_warnings:
                    st.warning("\n".join(safety_warnings))

                # Display sources
                display_sources(sources, processing_time, confidence_scores)

            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                    "sources": [],
                    "processing_time": 0
                })

        else:
            st.error("❌ RAG system not available. Please check the system status.")
            st.info("Try refreshing the page or contact support if the issue persists.")

def _normalise_source_document(source: Any) -> Dict[str, Any]:
    """Return a unified dictionary describing the source document."""
    if hasattr(source, "metadata") and hasattr(source, "page_content"):
        return {
            "metadata": dict(source.metadata or {}),
            "page_content": source.page_content,
        }
    if isinstance(source, dict):
        return {
            "metadata": dict(source.get("metadata", {})),
            "page_content": source.get("page_content", ""),
        }
    return {"metadata": {}, "page_content": str(source)}


def display_sources(source_documents, processing_time, confidence_scores=None):
    """Display source documents in an elegant format"""
    if not source_documents:
        st.info("ℹ️ No specific sources found for this response.")
        return

    normalised_docs = [_normalise_source_document(doc) for doc in source_documents]

    # Check if we have PubMed articles
    has_pubmed = any(
        any(key in doc["metadata"] for key in ("pubmed_id", "pmid", "id")) or
        doc["metadata"].get("source_type") == "pubmed"
        for doc in normalised_docs
    )
    local_docs = [doc for doc in normalised_docs if not any(
        key in doc["metadata"] for key in ("pubmed_id", "pmid", "id")
    ) and doc["metadata"].get("source_type") != "pubmed"]
    pubmed_docs = [doc for doc in normalised_docs if any(
        key in doc["metadata"] for key in ("pubmed_id", "pmid", "id")
    ) or doc["metadata"].get("source_type") == "pubmed"]

    # Create a container for sources
    with st.container():
        st.markdown("---")

        # Header with metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("📄 Total Sources", len(normalised_docs))
        with col3:
            if local_docs:
                st.metric("📁 Local Docs", len(local_docs))
        with col4:
            if pubmed_docs:
                st.metric("🔬 PubMed", len(pubmed_docs))

        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            st.metric("🎯 Avg. Relevance", f"{avg_confidence:.2f}")

        st.markdown("### 📚 Sources & References")

        # Create tabs for different views
        if has_pubmed:
            tab1, tab2, tab3, tab4 = st.tabs(["📄 All Sources", "📁 Local Documents", "🔬 PubMed Articles", "📊 Analysis"])
        else:
            tab1, tab2, tab3 = st.tabs(["📄 Document Sources", "📊 Source Analysis", "🔍 Quick Search"])

        with tab1 if not has_pubmed else tab1:  # All Sources or Document Sources
            for i, doc in enumerate(normalised_docs, 1):
                metadata = doc["metadata"]
                page_content = doc["page_content"]

                # Check if this is a PubMed article
                if "pubmed_id" in metadata:
                    pubmed_id = metadata.get("pubmed_id", "")
                    title = metadata.get("title", "Untitled Article")
                    authors = metadata.get("authors", [])
                    journal = metadata.get("journal", "")
                    year = metadata.get("publication_date", "")[:4] if metadata.get("publication_date") else ""
                    url = metadata.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}")

                    # PubMed article display
                    st.markdown(f"""
                    <div class="source-card" style="border-left-color: #2e7d32;">
                        <h4>🔬 {title}</h4>
                        <p><strong>Authors:</strong> {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}</p>
                        <p><strong>Journal:</strong> {journal} ({year})</p>
                        <p><strong>PubMed ID:</strong> {pubmed_id}</p>
                        <a href="{url}" target="_blank">View on PubMed →</a>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show abstract if available
                    abstract = metadata.get("abstract", "")
                    if abstract and len(abstract) > 200:
                        with st.expander("Show Abstract"):
                            st.markdown(f"*{abstract[:500]}{'...' if len(abstract) > 500 else ''}*")
                else:
                    # Local document display
                    source_file = metadata.get("source_file") or metadata.get("title") or metadata.get("id", "Unknown")
                    page = metadata.get("page", "Unknown")
                    chunk_id = metadata.get("chunk_id", "Unknown")

                    display_name = str(source_file).replace("_", " ").replace("-", " ").title()
                    if display_name.lower().endswith(".pdf"):
                        display_name = display_name[:-4]

                    confidence_indicator = ""
                    if confidence_scores and i <= len(confidence_scores):
                        score = confidence_scores[i - 1]
                        if score > 0.8:
                            confidence_indicator = "🟢 High Relevance"
                        elif score > 0.6:
                            confidence_indicator = "🟡 Medium Relevance"
                        else:
                            confidence_indicator = "🔴 Low Relevance"

                    with st.expander(f"📖 Source {i}: {display_name} (Page {page}) {confidence_indicator}"):
                        col_left, col_right = st.columns([2, 1])

                        with col_left:
                            st.markdown(f"**📁 File**: {source_file}")
                            st.markdown(f"**📄 Page**: {page}")
                            st.markdown(f"**🔢 Chunk ID**: {chunk_id}")
                            if confidence_scores and i <= len(confidence_scores):
                                st.markdown(f"**🎯 Relevance Score**: {confidence_scores[i-1]:.3f}")

                        with col_right:
                            if st.button(f"📋 Copy Text {i}", key=f"copy_{i}"):
                                st.code(page_content, language="text")

                        st.markdown("**📝 Content Preview**:")
                        preview_text = page_content[:800]
                        if len(page_content) > 800:
                            preview_text += "..."
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 3px solid #2a5298;">
                                {preview_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # Local Documents Tab (only when PubMed is available)
        if has_pubmed:
            with tab2:
                if local_docs:
                    source_files = [doc["metadata"].get("source_file", "Unknown") for doc in local_docs]
                    file_counts: Dict[str, int] = {}
                    for file in source_files:
                        clean_name = str(file).replace("_", " ").replace("-", " ")
                        if clean_name.endswith(".pdf"):
                            clean_name = clean_name[:-4]
                        file_counts[clean_name] = file_counts.get(clean_name, 0) + 1

                    if file_counts:
                        fig_pie = px.pie(
                            values=list(file_counts.values()),
                            names=list(file_counts.keys()),
                            title="📊 Local Document Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                        )
                        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                        st.plotly_chart(fig_pie, use_container_width=True)

                        pages = [doc["metadata"].get("page", 0) for doc in local_docs]
                        if pages:
                            fig_bar = px.histogram(
                                x=pages,
                                title="📄 Page Distribution of Local Sources",
                                labels={"x": "Page Number", "y": "Number of Sources"},
                                color_discrete_sequence=['#2a5298'],
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("📁 No local documents used in this response")

            # PubMed Articles Tab
            with tab3:
                if pubmed_docs:
                    # Journal distribution
                    journals = [doc["metadata"].get("journal", "Unknown") for doc in pubmed_docs]
                    journal_counts: Dict[str, int] = {}
                    for journal in journals:
                        journal_counts[journal] = journal_counts.get(journal, 0) + 1

                    if journal_counts:
                        fig_journal = px.pie(
                            values=list(journal_counts.values()),
                            names=list(journal_counts.keys()),
                            title="🔬 PubMed Articles by Journal",
                            color_discrete_sequence=px.colors.qualitative.G10,
                        )
                        fig_journal.update_traces(textposition="inside", textinfo="percent+label")
                        st.plotly_chart(fig_journal, use_container_width=True)

                    # Publication years
                    years = []
                    for doc in pubmed_docs:
                        pub_date = doc["metadata"].get("publication_date", "")
                        if pub_date and len(pub_date) >= 4:
                            years.append(int(pub_date[:4]))

                    if years:
                        fig_years = px.histogram(
                            x=years,
                            title="📅 Publication Years",
                            labels={"x": "Year", "y": "Number of Articles"},
                            color_discrete_sequence=['#2e7d32'],
                        )
                        st.plotly_chart(fig_years, use_container_width=True)

                    # Article list with detailed info
                    st.markdown("### 📝 Article Details")
                    for i, doc in enumerate(pubmed_docs, 1):
                        metadata = doc["metadata"]
                        title = metadata.get("title", "Untitled")
                        authors = metadata.get("authors", [])
                        journal = metadata.get("journal", "")
                        year = metadata.get("publication_date", "")[:4] if metadata.get("publication_date") else ""
                        pubmed_id = metadata.get("pubmed_id", "")
                        provider = metadata.get("provider") or metadata.get("provider_family") or "pubmed"

                        with st.expander(f"{i}. {title} ({year})"):
                            st.markdown(f"**Authors**: {', '.join(authors[:5])}{' et al.' if len(authors) > 5 else ''}")
                            st.markdown(f"**Journal**: {journal}")
                            st.markdown(f"**PubMed ID**: {pubmed_id}")
                            st.caption(f"Provider: {provider}")
                            if metadata.get("url"):
                                st.markdown(f"[**View on PubMed**]({metadata['url']})")

                            # Show abstract if available
                            abstract = metadata.get("abstract", "")
                            if abstract:
                                st.markdown("**Abstract**:")
                                st.info(abstract[:1000] + "..." if len(abstract) > 1000 else abstract)
                else:
                    st.info("🔬 No PubMed articles used in this response")

            # Analysis Tab
            with tab4:
                st.markdown("### 📊 Source Analysis")

                # Comparison pie chart
                fig_comparison = px.pie(
                    values=[len(local_docs), len(pubmed_docs)],
                    names=["Local Documents", "PubMed Articles"],
                    title="📈 Source Type Distribution",
                    color_discrete_sequence=["#2a5298", "#2e7d32"],
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Additional metrics
                col1, col2 = st.columns(2)
                with col1:
                    if pubmed_docs:
                        # Calculate average publication year
                        years = []
                        for doc in pubmed_docs:
                            pub_date = doc["metadata"].get("publication_date", "")
                            if pub_date and len(pub_date) >= 4:
                                years.append(int(pub_date[:4]))
                        if years:
                            avg_year = sum(years) / len(years)
                            st.metric("📅 Avg. Publication Year", f"{int(avg_year)}")

                with col2:
                    if local_docs and pubmed_docs:
                        ratio = len(pubmed_docs) / len(normalised_docs)
                        st.metric("🔬 PubMed Ratio", f"{ratio:.1%}")

        else:
            # Original tabs when no PubMed
            with tab2:
                source_files = [doc["metadata"].get("source_file", "Unknown") for doc in normalised_docs]
                file_counts: Dict[str, int] = {}
                for file in source_files:
                    clean_name = str(file).replace("_", " ").replace("-", " ")
                    if clean_name.endswith(".pdf"):
                        clean_name = clean_name[:-4]
                    file_counts[clean_name] = file_counts.get(clean_name, 0) + 1

                if file_counts:
                    fig_pie = px.pie(
                        values=list(file_counts.values()),
                        names=list(file_counts.keys()),
                        title="📊 Source Document Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    pages = [doc["metadata"].get("page", 0) for doc in normalised_docs]
                    if pages:
                        fig_bar = px.histogram(
                            x=pages,
                            title="📄 Page Distribution of Sources",
                            labels={"x": "Page Number", "y": "Number of Sources"},
                            color_discrete_sequence=['#2a5298'],
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

            with tab3:
                st.markdown("#### 🔍 Search Within Sources")
                search_term = st.text_input("Search for specific terms in the source documents:")

                if search_term:
                    matches = []
                    for i, doc in enumerate(normalised_docs, 1):
                        page_content = doc["page_content"]
                        if search_term.lower() in page_content.lower():
                            content_lower = page_content.lower()
                            index = content_lower.find(search_term.lower())
                            start = max(0, index - 100)
                            end = min(len(page_content), index + len(search_term) + 100)
                            context = page_content[start:end]

                            metadata = doc["metadata"]
                            matches.append(
                                {
                                    "source": i,
                                    "file": metadata.get("source_file", "Unknown"),
                                    "page": metadata.get("page", "Unknown"),
                                    "context": context,
                                }
                            )

                    if matches:
                        st.success(f"Found {len(matches)} matches for '{search_term}':")
                        for match in matches:
                            st.markdown(
                                f"""
                                **Source {match['source']}** - {match['file']} (Page {match['page']})
                                > ...{match['context']}...
                                """
                            )
                    else:
                        st.info(f"No matches found for '{search_term}' in the source documents.")

def display_document_stats(rag_agent):
    """Display detailed document statistics"""
    if not rag_agent:
        st.error("RAG system not available")
        return

    st.markdown("## 📊 Document Statistics")

    stats = rag_agent.base_agent.get_knowledge_base_stats()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Documents", stats.get('document_count', 0))
    with col2:
        st.metric("📁 PDF Files", stats.get('pdf_files_available', 0))
    with col3:
        st.metric("🔍 Searchable Chunks", stats.get('document_count', 0))
    with col4:
        st.metric("💾 Index Size", "Ready" if stats.get('index_exists') else "Not Found")

    # Document types breakdown
    st.markdown("### 📖 Document Types & Use Cases")
    doc_categories = {
        "Research Papers": "📚",
        "Technical Documentation": "🔧",
        "Legal Documents": "⚖️",
        "Corporate Policies": "🏢",
        "Training Materials": "🎓",
        "User Manuals": "📖",
        "Compliance Documents": "✅",
        "Reports & Analysis": "📊",
        "Contracts & Agreements": "📝",
        "Specifications": "🔍",
        "Academic Papers": "🎓",
        "Reference Materials": "📚"
    }

    cols = st.columns(3)
    for i, (category, emoji) in enumerate(doc_categories.items()):
        with cols[i % 3]:
            st.markdown(f"{emoji} **{category}**")

    # System health
    st.markdown("### 🔧 System Health")
    health_col1, health_col2 = st.columns(2)

    with health_col1:
        st.markdown("**API Status**")
        if rag_agent:
            st.success("🟢 NVIDIA API Connected")
            st.success("🟢 Vector Database Loaded")
            st.success("🟢 LLM Model Ready")
        else:
            st.error("🔴 System Offline")

    with health_col2:
        st.markdown("**Performance Metrics**")
        st.info("📊 Embedding Dimension: 4096")
        st.info("🚀 Average Query Time: 2-8 seconds")
        st.info("💾 Storage: Local FAISS Index")

def main():
    """Main application function"""
    # Display header
    display_header()

    # Initialize RAG agent
    rag_agent = initialize_rag_agent()

    # Display sidebar
    display_sidebar(rag_agent)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Document Statistics"])

    with tab1:
        # Main content area
        col1, col2 = st.columns([3, 1])

        with col1:
            # Chat interface
            display_chat_interface(rag_agent)
    
    with col2:
        # Quick actions and tips
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **Sample Questions:**
        • What is the main topic of the documents?
        • Summarize the key points
        • What are the requirements mentioned?
        • How does [concept A] relate to [concept B]?
        • What are the benefits described?
        • Explain the process for [specific topic]
        """)

        # Advanced features
        st.markdown("### 🛠️ Actions")

        # Export chat history
        if st.button("📥 Export Chat History"):
            if st.session_state.messages:
                chat_export = []
                for msg in st.session_state.messages:
                    chat_export.append({
                        "timestamp": datetime.now().isoformat(),
                        "role": msg["role"],
                        "content": msg["content"],
                        "sources_count": len(msg.get("sources", [])),
                        "processing_time": msg.get("processing_time", 0)
                    })

                import json
                export_data = json.dumps(chat_export, indent=2)
                st.download_button(
                    label="💾 Download Chat History",
                    data=export_data,
                    file_name=f"difc_legal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No chat history to export")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # Knowledge base rebuild
        if st.button("🔄 Rebuild Knowledge Base"):
            st.cache_resource.clear()
            st.rerun()

        # Pharma capability checks
        if st.button("🧪 Run Pharma Benchmark"):
            try:
                from src.enhanced_config import EnhancedRAGConfig
                from src.clients.nemo_client_enhanced import EnhancedNeMoClient
                cfg = EnhancedRAGConfig.from_env()
                api_key = os.getenv("NVIDIA_API_KEY")
                with st.spinner("Running pharmaceutical capability checks..."):
                    client = EnhancedNeMoClient(
                        config=cfg,
                        enable_fallback=True,
                        pharmaceutical_optimized=True,
                        api_key=api_key,
                    )
                    report = client.test_pharmaceutical_capabilities()
                status = str(report.get("overall_status", "unknown")).title()
                if status.lower() == "failed":
                    st.error(f"Benchmark Status: {status}")
                elif status.lower() == "partial":
                    st.warning(f"Benchmark Status: {status}")
                else:
                    st.success(f"Benchmark Status: {status}")
                st.json(report)
            except Exception as e:
                st.error(f"Benchmark run failed: {e}")

        # System information
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI assistant is powered by:
        - **NVIDIA** embedding models
        - **Meta LLaMA** language model
        - **FAISS** vector database
        - **1,869** legal document chunks
        """)

        # Chat statistics
        if st.session_state.messages:
            st.markdown("### 📈 Session Stats")
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]

            st.metric("Questions Asked", len(user_messages))
            st.metric("Responses Given", len(assistant_messages))

            if assistant_messages:
                avg_time = sum(m.get("processing_time", 0) for m in assistant_messages) / len(assistant_messages)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")

    with tab2:
        # Document statistics page
        display_document_stats(rag_agent)

if __name__ == "__main__":
    main()
