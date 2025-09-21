"""
Main Application - Interactive RAG Agent Interface
Provides CLI interface for interacting with the RAG agent
"""

import os
import sys
import time
from typing import Any, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.enhanced_rag_agent import EnhancedRAGAgent
from src.nvidia_embeddings import NVIDIAEmbeddings


def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🤖 RAG Agent - NVIDIA NemoRetriever Template")
    print("=" * 60)
    print("Ask questions about your PDF documents!")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Type 'help' for available commands")
    print("Type 'stats' to see knowledge base statistics")
    print("Type 'rebuild' to rebuild the knowledge base")
    print("-" * 60)


def print_help():
    """Print help information"""
    print("\n📖 Available Commands:")
    print("  help     - Show this help message")
    print("  stats    - Show knowledge base statistics")
    print("  rebuild  - Rebuild the knowledge base from PDFs")
    print("  clear    - Clear the screen")
    print("  quit/exit/q - Exit the application")
    print("\n💡 Tips:")
    print("  - Ask specific questions about your documents")
    print("  - The system will show relevant source documents")
    print("  - Processing time is displayed for each query")
    print()


def print_stats(agent: EnhancedRAGAgent):
    """Print knowledge base statistics"""
    print("\n📊 Knowledge Base Statistics:")
    stats = agent.base_agent.get_knowledge_base_stats()
    
    for key, value in stats.items():
        if key == "status":
            status_emoji = "✅" if value == "Index loaded" else "❌"
            print(f"  {status_emoji} Status: {value}")
        elif key == "document_count":
            print(f"  📄 Documents: {value}")
        elif key == "pdf_files_available":
            print(f"  📁 PDF Files: {value}")
        elif key == "docs_folder":
            print(f"  📂 Docs Folder: {value}")
        elif key == "index_path":
            print(f"  💾 Index Path: {value}")
        else:
            print(f"  {key}: {value}")
    print()


def _normalise_source(source: Any) -> Tuple[Dict[str, Any], str]:
    """Return metadata and text for Document or dict source entries."""
    if hasattr(source, "metadata") and hasattr(source, "page_content"):
        return dict(source.metadata or {}), source.page_content
    if isinstance(source, dict):
        return dict(source.get("metadata", {})), source.get("page_content", "")
    return {}, str(source)


def format_response(response: Dict[str, Any]) -> None:
    """Format and display the enhanced response payload."""
    error = response.get("error")
    if error:
        print("\n⚠️  Safety block triggered")
        print("-" * 40)
        for issue in error.get("issues", []):
            print(f"• {issue}")
        recommendations = error.get("recommendations", [])
        if recommendations:
            print("\nRecommended actions:")
            for item in recommendations:
                print(f"  - {item}")
        print()
        return

    answer = response.get("answer") or "No answer returned."
    print("\n🤖 Answer:")
    print("-" * 40)
    print(answer)
    print("-" * 40)

    sources = response.get("sources") or []
    if sources:
        print(f"\n📚 Sources ({len(sources)} documents):")
        for idx, source in enumerate(sources, 1):
            metadata, content = _normalise_source(source)
            source_file = metadata.get("source_file") or metadata.get("title") or metadata.get("id", "Unknown")
            page = metadata.get("page", "Unknown")
            chunk_id = metadata.get("chunk_id", "Unknown")

            print(f"\n  [{idx}] {source_file} (Page: {page}, Chunk: {chunk_id})")
            preview = content[:150]
            if len(content) > 150:
                preview += "..."
            print(f"      \"{preview}\"")

    processing_time = response.get("processing_time") or 0.0
    print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")

    warnings = response.get("safety", {}).get("output_validation", {}).get("warnings")
    if warnings:
        print("\n⚠️  Safety warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    disclaimer = response.get("disclaimer")
    if disclaimer:
        print("\n📄 Disclaimer:")
        print(disclaimer)
    print()


def setup_rag_agent():
    """Initialize and setup the enhanced RAG agent"""
    print("🔧 Initializing Enhanced RAG Agent...")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY")
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    guardrails_config = os.getenv("MEDICAL_GUARDRAILS_CONFIG")
    enable_synthesis = os.getenv("ENABLE_SYNTHESIS", "true").strip().lower() in ("1", "true", "yes", "on")
    enable_ddi = os.getenv("ENABLE_DDI_ANALYSIS", "true").strip().lower() in ("1", "true", "yes", "on")
    safety_mode = os.getenv("ENHANCED_RAG_SAFETY_MODE", "balanced")
    
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        print("Please check your .env file and ensure the API key is set")
        return None
    
    # Check if docs folder exists
    if not Path(docs_folder).exists():
        print(f"📁 Creating documents folder: {docs_folder}")
        Path(docs_folder).mkdir(parents=True, exist_ok=True)
        print(f"📝 Please add your PDF files to: {Path(docs_folder).absolute()}")
    
    # Test NVIDIA API connection
    print("🔌 Testing NVIDIA API connection...")
    try:
        embeddings = NVIDIAEmbeddings(api_key)
        if not embeddings.test_connection():
            print("❌ Failed to connect to NVIDIA API")
            return None
        print("✅ NVIDIA API connection successful!")
    except Exception as e:
        print(f"❌ NVIDIA API connection failed: {str(e)}")
        return None
    
    # Initialize RAG agent
    try:
        rag_agent = EnhancedRAGAgent(
            docs_folder=docs_folder,
            api_key=api_key,
            vector_db_path=vector_db_path,
            guardrails_config_path=guardrails_config,
            enable_synthesis=enable_synthesis,
            enable_ddi_analysis=enable_ddi,
            safety_mode=safety_mode,
        )
        print("✅ Enhanced RAG Agent initialized successfully!")
        return rag_agent
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced RAG Agent: {str(e)}")
        return None


def main():
    """Main application loop"""
    print_banner()
    
    # Setup RAG agent
    agent = setup_rag_agent()
    if not agent:
        print("❌ Failed to initialize Enhanced RAG Agent. Exiting...")
        return

    # Setup knowledge base
    print("\n🔨 Setting up knowledge base...")
    if not agent.base_agent.setup_knowledge_base():
        print("❌ Failed to setup knowledge base.")
        print("Please ensure you have PDF files in the documents folder and try again.")

        # Ask if user wants to continue anyway
        response = input("\nWould you like to continue anyway? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            return
    else:
        print("✅ Knowledge base setup completed!")
        print_stats(agent)

    print("\n🚀 RAG Agent is ready! Ask me anything about your documents.")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            print("\n" + "="*60)
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using RAG Agent! Goodbye!")
                break
            
            elif question.lower() == 'help':
                print_help()
                continue
            
            elif question.lower() == 'stats':
                print_stats(agent)
                continue
            
            elif question.lower() == 'rebuild':
                print("\n🔨 Rebuilding knowledge base...")
                if agent.base_agent.setup_knowledge_base(force_rebuild=True):
                    print("✅ Knowledge base rebuilt successfully!")
                    print_stats(agent)
                else:
                    print("❌ Failed to rebuild knowledge base")
                continue
            
            elif question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            
            # Process question
            print(f"\n🔍 Searching knowledge base...")
            response = agent.ask_question(question)
            format_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or type 'help' for available commands.")


if __name__ == "__main__":
    main()
