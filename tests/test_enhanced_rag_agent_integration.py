"""Tests for Enhanced RAG Agent PubMed integration."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add src to path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.enhanced_config import EnhancedRAGConfig
from src.enhanced_pubmed_scraper import EnhancedPubMedScraper
from src.enhanced_rag_agent import EnhancedRAGAgent
from src.pharmaceutical_query_adapter import (
    SystemHealthReport,
    build_enhanced_rag_agent,
    build_integrated_system,
    check_system_health,
)
from src.query_engine import EnhancedQueryEngine


class TestEnhancedRAGAgentPubMedIntegration:
    """Test suite for Enhanced RAG Agent PubMed integration."""

    @pytest.fixture
    def temp_docs_folder(self):
        """Create a temporary documents folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            # Create a dummy PDF file
            (docs_dir / "test.pdf").write_text("Dummy PDF content")
            yield str(docs_dir)

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration with PubMed enabled."""
        config = Mock(spec=EnhancedRAGConfig)
        config.should_enable_pubmed.return_value = True
        config.should_use_hybrid_mode.return_value = True
        config.enable_enhanced_pubmed_scraper = True
        config.enable_rate_limiting = True
        config.enable_advanced_caching = True
        config.use_normalized_cache_keys = True
        config.enable_query_enhancement = True
        config.pubmed_cache_integration = True
        config.max_external_results = 10
        config.relevance_threshold = 0.7
        config.query_timeout_seconds = 30
        config.safe_mode = True
        config.error_handling_strategy = "graceful"
        config.fallback_to_local_on_error = True
        config.summarize_flags.return_value = {"hybrid": True}
        return config

    @pytest.fixture
    def mock_pubmed_scraper(self):
        """Create a mock PubMed scraper."""
        scraper = Mock(spec=EnhancedPubMedScraper)
        scraper.search_pubmed.return_value = [
            {
                "pubmed_id": "12345",
                "title": "Test Study",
                "abstract": "This is a test abstract",
                "authors": ["Author A", "Author B"],
                "journal": "Test Journal",
                "publication_date": "2023-01-01",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
            }
        ]
        scraper.combined_status_report.return_value = {
            "cache": {"status": "ready", "hit_rate": 0.8},
            "rate_limit": {"status": "ready", "requests_remaining": 100},
        }
        return scraper

    @pytest.fixture
    def mock_query_engine(self, mock_pubmed_scraper):
        """Create a mock query engine."""
        engine = Mock(spec=EnhancedQueryEngine)
        engine.process_pharmaceutical_query.return_value = {
            "results": [
                {
                    "pubmed_id": "12345",
                    "title": "Test Study",
                    "abstract": "This is a test abstract",
                    "authors": ["Author A", "Author B"],
                    "journal": "Test Journal",
                    "publication_date": "2023-01-01",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
                    "relevance_score": 0.85,
                }
            ],
            "query_info": {"total_results": 1, "search_time": 1.5, "cache_hits": 0},
        }
        return engine

    @pytest.fixture
    def rag_agent(self, temp_docs_folder, mock_config):
        """Create an Enhanced RAG agent with mocked components."""
        with patch("src.enhanced_rag_agent.RAGAgent"), patch("src.enhanced_rag_agent.MedicalGuardrails"), patch(
            "src.enhanced_rag_agent.SynthesisEngine"
        ), patch("src.enhanced_rag_agent.DDIProcessor"), patch("src.nvidia_embeddings.NVIDIAEmbeddings"):
            agent = EnhancedRAGAgent(
                docs_folder=temp_docs_folder,
                api_key="test_api_key",
                config=mock_config,
                enable_synthesis=False,
                enable_ddi_analysis=False,
            )
            # Mock the base agent
            agent.base_agent = Mock()
            agent.base_agent.ask_question.return_value = {
                "answer": "Test answer from local documents",
                "sources": [{"metadata": {"source_file": "test.pdf"}, "page_content": "Test content"}],
                "processing_time": 1.0,
            }
            agent.base_agent.get_knowledge_base_stats.return_value = {"document_count": 1, "pdf_files_available": 1}
            return agent

    def test_pubmed_integration_initialization(self, rag_agent, mock_config):
        """Test that PubMed integration initializes correctly."""
        assert rag_agent.config == mock_config
        assert rag_agent._pubmed_available() == False  # No components set yet

    def test_ensure_pubmed_components_creates_components(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test that PubMed components are created when needed."""
        # Mock the component creation
        with patch("src.enhanced_pubmed_scraper.EnhancedPubMedScraper", return_value=mock_pubmed_scraper), patch(
            "src.query_engine.EnhancedQueryEngine", return_value=mock_query_engine
        ):
            rag_agent._ensure_pubmed_components()

            assert rag_agent._pubmed_scraper == mock_pubmed_scraper
            assert rag_agent._pubmed_query_engine == mock_query_engine
            assert rag_agent._pubmed_available() == True

    def test_ask_question_pubmed_only(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test asking questions with PubMed only."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        response = rag_agent.ask_question_pubmed_only("test query", max_results=5)

        assert "answer" in response
        assert "sources" in response
        assert "processing_time" in response
        assert len(response["sources"]) == 1
        assert response["sources"][0]["metadata"]["pubmed_id"] == "12345"

    def test_ask_question_with_pubmed(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test asking questions with both local and PubMed sources."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        response = rag_agent.ask_question_with_pubmed("test query")

        assert "answer" in response
        assert "sources" in response
        # Should have both local and PubMed sources
        assert len(response["sources"]) >= 1

    def test_ask_question_hybrid(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test hybrid question answering."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        response = rag_agent.ask_question_hybrid("test query", local_k=3, pubmed_max=5)

        assert "answer" in response
        assert "sources" in response
        assert "query_mode" in response
        assert response["query_mode"] == "hybrid"

    def test_get_system_status_includes_pubmed(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test that system status includes PubMed information."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        status = rag_agent.get_system_status()

        assert "pubmed" in status
        assert status["pubmed"]["enabled"] == True
        assert "metrics" in status["pubmed"]
        assert "component_health" in status["pubmed"]

    def test_classify_query_type(self, rag_agent):
        """Test query classification."""
        # Medical query
        query_type = rag_agent._classify_query("What are the effects of aspirin?")
        assert query_type == "medical"

        # General query
        query_type = rag_agent._classify_query("What is the capital of France?")
        assert query_type == "general"

        # Pharmaceutical query
        query_type = rag_agent._classify_query("Drug interactions between warfarin and ibuprofen")
        assert query_type == "pharmaceutical"

    def test_pubmed_fallback_on_error(self, rag_agent):
        """Test fallback to local documents when PubMed fails."""
        rag_agent._pubmed_scraper = Mock()
        rag_agent._pubmed_query_engine = Mock()
        rag_agent._pubmed_query_engine.process_pharmaceutical_query.side_effect = Exception("PubMed error")

        response = rag_agent.ask_question_pubmed_only("test query")

        # Should fall back to local documents
        assert "answer" in response
        assert "error" in response
        assert "fallback" in response["error"].get("message", "").lower()

    def test_component_health_tracking(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test that component health is tracked correctly."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        # Initialize components
        rag_agent._ensure_pubmed_components()

        health = rag_agent.component_health
        assert "pubmed_integration" in health
        assert health["pubmed_integration"]["status"] == "ready"

    def test_cache_aware_querying(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test cache-aware query optimization."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        # Mock cache status
        mock_pubmed_scraper.get_cache_status.return_value = {"hit_rate": 0.8, "size_mb": 10.5, "total_entries": 100}

        response = rag_agent.ask_question_pubmed_only("test query")

        # Should consider cache status
        assert "answer" in response

    def test_rate_limit_aware_processing(self, rag_agent, mock_pubmed_scraper, mock_query_engine):
        """Test rate limit aware processing."""
        rag_agent._pubmed_scraper = mock_pubmed_scraper
        rag_agent._pubmed_query_engine = mock_query_engine

        # Mock rate limit status
        mock_pubmed_scraper.get_rate_limit_status.return_value = {
            "requests_remaining": 5,
            "reset_time": "2023-01-01T12:00:00Z",
            "wait_time_seconds": 10,
        }

        response = rag_agent.ask_question_pubmed_only("test query")

        # Should handle rate limiting
        assert "answer" in response


class TestBuildEnhancedRAGAgent:
    """Test factory methods for building enhanced RAG agents."""

    def test_build_enhanced_rag_agent_with_pubmed(self, temp_docs_folder, mock_config):
        """Test building enhanced RAG agent with PubMed integration."""
        with patch("src.enhanced_rag_agent.EnhancedRAGAgent") as mock_agent_class, patch(
            "src.enhanced_pubmed_scraper.EnhancedPubMedScraper"
        ) as mock_scraper_class, patch("src.query_engine.EnhancedQueryEngine") as mock_engine_class:
            agent = build_enhanced_rag_agent(docs_folder=temp_docs_folder, api_key="test_key", config=mock_config)

            # Should create PubMed components
            mock_scraper_class.assert_called_once()
            mock_engine_class.assert_called_once()
            mock_agent_class.assert_called_once()

    def test_build_integrated_system(self, temp_docs_folder, mock_config):
        """Test building complete integrated system."""
        with patch("src.pharmaceutical_query_adapter.build_enhanced_rag_agent") as mock_build:
            mock_agent = Mock()
            mock_agent._pubmed_scraper = Mock()
            mock_agent._pubmed_query_engine = Mock()
            mock_agent._pubmed_available.return_value = True
            mock_agent.component_health = {"pubmed_integration": {"status": "ready"}}
            mock_build.return_value = mock_agent

            system = build_integrated_system(docs_folder=temp_docs_folder, api_key="test_key", config=mock_config)

            assert "rag_agent" in system
            assert "config" in system
            assert "status" in system
            assert system["status"]["pubmed_enabled"] == True

    def test_check_system_health(self, temp_docs_folder):
        """Test system health checking."""
        with patch("src.pharmaceutical_query_adapter.EnhancedRAGConfig") as mock_config_class:
            mock_config = Mock()
            mock_config.should_enable_pubmed.return_value = True
            mock_config_class.from_env.return_value = mock_config

            # Create mock components
            mock_agent = Mock()
            mock_agent._ensure_components_initialized.return_value = None
            mock_agent._ensure_pubmed_components.return_value = None
            mock_agent.component_health = {
                "pubmed_integration": {"status": "ready"},
                "medical_guardrails": {"status": "ready"},
                "synthesis_engine": {"status": "ready"},
                "ddi_pk_processor": {"status": "ready"},
            }

            mock_scraper = Mock()
            mock_scraper.combined_status_report.return_value = {
                "cache": {"status": "ready"},
                "rate_limit": {"status": "ready"},
            }

            report = check_system_health(rag_agent=mock_agent, pubmed_scraper=mock_scraper)

            assert isinstance(report, SystemHealthReport)
            assert report.rag_agent_ready == True
            assert report.pubmed_integration_ready == True
            assert report.pubmed_scraper_ready == True

    def test_graceful_degradation_when_pubmed_disabled(self, temp_docs_folder):
        """Test graceful degradation when PubMed is disabled."""
        config = Mock(spec=EnhancedRAGConfig)
        config.should_enable_pubmed.return_value = False

        with patch("src.enhanced_rag_agent.EnhancedRAGAgent") as mock_agent_class:
            agent = build_enhanced_rag_agent(docs_folder=temp_docs_folder, api_key="test_key", config=config)

            # Should not create PubMed components
            mock_agent_class.assert_called_once()
            # Verify no PubMed components were created
            call_args = mock_agent_class.call_args
            assert call_args[1]["pubmed_query_engine"] is None
            assert call_args[1]["pubmed_scraper"] is None

    def test_configuration_validation(self, temp_docs_folder):
        """Test that configuration is properly validated."""
        config = Mock(spec=EnhancedRAGConfig)
        config.should_enable_pubmed.return_value = True
        config.enable_rate_limiting = False

        with patch("src.enhanced_rag_agent.EnhancedRAGAgent") as mock_agent_class, patch(
            "src.enhanced_pubmed_scraper.EnhancedPubMedScraper"
        ) as mock_scraper_class:
            agent = build_enhanced_rag_agent(docs_folder=temp_docs_folder, api_key="test_key", config=config)

            # Should pass configuration to scraper
            mock_scraper_class.assert_called_once_with(
                enable_rate_limiting=False,
                enable_advanced_caching=config.enable_advanced_caching,
                use_normalized_cache_keys=config.use_normalized_cache_keys,
            )


if __name__ == "__main__":
    pytest.main([__file__])
