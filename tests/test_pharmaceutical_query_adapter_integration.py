"""Tests for pharmaceutical query adapter integration features."""
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
from src.pharmaceutical_query_adapter import (
    SystemHealthReport,
    build_enhanced_pharmaceutical_query_engine,
    build_enhanced_rag_agent,
    build_integrated_system,
    check_system_health,
)
from src.query_engine import EnhancedQueryEngine


class TestBuildEnhancedPharmaceuticalQueryEngine:
    """Test building enhanced pharmaceutical query engines."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=EnhancedRAGConfig)
        config.enable_rate_limiting = True
        config.enable_advanced_caching = True
        config.use_normalized_cache_keys = True
        config.enable_enhanced_pubmed_scraper = True
        config.enable_query_enhancement = True
        config.pubmed_cache_integration = True
        return config

    @pytest.fixture
    def mock_scraper(self):
        """Create a mock PubMed scraper."""
        return Mock(spec=EnhancedPubMedScraper)

    def test_build_with_scraper_provided(self, mock_scraper, mock_config):
        """Test building with pre-configured scraper."""
        with patch("src.query_engine.EnhancedQueryEngine") as mock_engine_class:
            engine = build_enhanced_pharmaceutical_query_engine(scraper=mock_scraper, config=mock_config)

            mock_engine_class.assert_called_once_with(
                mock_scraper,
                enable_query_enhancement=mock_config.enable_query_enhancement,
                cache_filtered_results=mock_config.pubmed_cache_integration,
            )

    def test_build_creates_scraper(self, mock_config):
        """Test building creates scraper when not provided."""
        with patch("src.enhanced_pubmed_scraper.EnhancedPubMedScraper") as mock_scraper_class, patch(
            "src.query_engine.EnhancedQueryEngine"
        ) as mock_engine_class:
            mock_scraper = Mock()
            mock_scraper_class.return_value = mock_scraper

            build_enhanced_pharmaceutical_query_engine(config=mock_config)

            # Should create scraper with correct config
            mock_scraper_class.assert_called_once_with(
                enable_rate_limiting=mock_config.enable_rate_limiting,
                enable_advanced_caching=mock_config.enable_advanced_caching,
                use_normalized_cache_keys=mock_config.use_normalized_cache_keys,
            )

    def test_build_disables_advanced_features_when_flag_false(self, mock_config):
        """Test that advanced features are disabled when flag is false."""
        mock_config.enable_enhanced_pubmed_scraper = False

        with patch("src.enhanced_pubmed_scraper.EnhancedPubMedScraper") as mock_scraper_class, patch(
            "src.query_engine.EnhancedQueryEngine"
        ) as mock_engine_class:
            build_enhanced_pharmaceutical_query_engine(config=mock_config)

            # Should disable advanced caching
            call_args = mock_scraper_class.call_args[1]
            assert call_args["enable_advanced_caching"] == False
            assert call_args["use_normalized_cache_keys"] == False

    def test_build_with_default_config(self):
        """Test building with default configuration."""
        with patch("src.enhanced_config.EnhancedRAGConfig") as mock_config_class, patch(
            "src.enhanced_pubmed_scraper.EnhancedPubMedScraper"
        ) as mock_scraper_class, patch("src.query_engine.EnhancedQueryEngine") as mock_engine_class:
            mock_config = Mock()
            mock_config.enable_rate_limiting = True
            mock_config.enable_advanced_caching = True
            mock_config.use_normalized_cache_keys = True
            mock_config.enable_enhanced_pubmed_scraper = True
            mock_config.enable_query_enhancement = True
            mock_config.pubmed_cache_integration = True
            mock_config_class.from_env.return_value = mock_config

            build_enhanced_pharmaceutical_query_engine()

            # Should load config from environment
            mock_config_class.from_env.assert_called_once()


class TestBuildEnhancedRAGAgent:
    """Test building enhanced RAG agents."""

    @pytest.fixture
    def temp_docs_folder(self):
        """Create a temporary documents folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            yield str(docs_dir)

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=EnhancedRAGConfig)
        config.should_enable_pubmed.return_value = True
        config.enable_rate_limiting = True
        config.enable_advanced_caching = True
        config.use_normalized_cache_keys = True
        config.enable_enhanced_pubmed_scraper = True
        config.enable_query_enhancement = True
        config.pubmed_cache_integration = True
        return config

    def test_build_with_pubmed_integration(self, temp_docs_folder, mock_config):
        """Test building RAG agent with PubMed integration."""
        with patch("src.pharmaceutical_query_adapter.EnhancedRAGAgent") as mock_agent_class, patch(
            "src.pharmaceutical_query_adapter.EnhancedPubMedScraper"
        ) as mock_scraper_class, patch("src.pharmaceutical_query_adapter.EnhancedQueryEngine") as mock_engine_class:
            agent = build_enhanced_rag_agent(docs_folder=temp_docs_folder, api_key="test_key", config=mock_config)

            # Should create PubMed components
            mock_scraper_class.assert_called_once()
            mock_engine_class.assert_called_once()
            mock_agent_class.assert_called_once()

            # Verify agent created with components
            call_args = mock_agent_class.call_args[1]
            assert call_args["pubmed_query_engine"] is not None
            assert call_args["pubmed_scraper"] is not None

    def test_build_without_pubmed_integration(self, temp_docs_folder, mock_config):
        """Test building RAG agent without PubMed integration."""
        mock_config.should_enable_pubmed.return_value = False

        with patch("src.pharmaceutical_query_adapter.EnhancedRAGAgent") as mock_agent_class, patch(
            "src.pharmaceutical_query_adapter.EnhancedPubMedScraper"
        ) as mock_scraper_class, patch("src.pharmaceutical_query_adapter.EnhancedQueryEngine") as mock_engine_class:
            agent = build_enhanced_rag_agent(docs_folder=temp_docs_folder, api_key="test_key", config=mock_config)

            # Should not create PubMed components
            mock_scraper_class.assert_not_called()
            mock_engine_class.assert_not_called()

            # Verify agent created without components
            call_args = mock_agent_class.call_args[1]
            assert call_args["pubmed_query_engine"] is None
            assert call_args["pubmed_scraper"] is None

    def test_build_with_preprovided_components(self, temp_docs_folder, mock_config):
        """Test building with pre-provided PubMed components."""
        mock_scraper = Mock(spec=EnhancedPubMedScraper)
        mock_engine = Mock(spec=EnhancedQueryEngine)

        with patch("src.pharmaceutical_query_adapter.EnhancedRAGAgent") as mock_agent_class:
            agent = build_enhanced_rag_agent(
                docs_folder=temp_docs_folder,
                api_key="test_key",
                config=mock_config,
                pubmed_scraper=mock_scraper,
                pubmed_query_engine=mock_engine,
            )

            # Should use provided components
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args[1]
            assert call_args["pubmed_query_engine"] == mock_engine
            assert call_args["pubmed_scraper"] == mock_scraper

    def test_build_passes_all_parameters(self, temp_docs_folder, mock_config):
        """Test that all parameters are passed through correctly."""
        with patch("src.pharmaceutical_query_adapter.EnhancedRAGAgent") as mock_agent_class:
            build_enhanced_rag_agent(
                docs_folder=temp_docs_folder,
                api_key="test_key",
                vector_db_path="./test_db",
                chunk_size=500,
                chunk_overlap=100,
                embedding_model_name="test-model",
                enable_preflight_embedding=True,
                append_disclaimer_in_answer=False,
                force_disclaimer_in_answer=True,
                guardrails_config_path="test.yaml",
                enable_synthesis=True,
                enable_ddi_analysis=True,
                safety_mode="strict",
                config=mock_config,
            )

            # Verify all parameters passed
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs["docs_folder"] == temp_docs_folder
            assert call_kwargs["api_key"] == "test_key"
            assert call_kwargs["vector_db_path"] == "./test_db"
            assert call_kwargs["chunk_size"] == 500
            assert call_kwargs["chunk_overlap"] == 100
            assert call_kwargs["embedding_model_name"] == "test-model"
            assert call_kwargs["enable_preflight_embedding"] == True
            assert call_kwargs["append_disclaimer_in_answer"] == False
            assert call_kwargs["force_disclaimer_in_answer"] == True
            assert call_kwargs["guardrails_config_path"] == "test.yaml"
            assert call_kwargs["enable_synthesis"] == True
            assert call_kwargs["enable_ddi_analysis"] == True
            assert call_kwargs["safety_mode"] == "strict"
            assert call_kwargs["config"] == mock_config


class TestBuildIntegratedSystem:
    """Test building complete integrated systems."""

    @pytest.fixture
    def temp_docs_folder(self):
        """Create a temporary documents folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            yield str(docs_dir)

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=EnhancedRAGConfig)
        config.should_enable_pubmed.return_value = True
        config.should_use_hybrid_mode.return_value = True
        return config

    def test_build_returns_all_components(self, temp_docs_folder, mock_config):
        """Test that build returns all expected components."""
        with patch("src.pharmaceutical_query_adapter.build_enhanced_rag_agent") as mock_build:
            # Create mock agent with PubMed components
            mock_agent = Mock()
            mock_scraper = Mock()
            mock_engine = Mock()
            mock_agent._pubmed_scraper = mock_scraper
            mock_agent._pubmed_query_engine = mock_engine
            mock_agent._pubmed_available.return_value = True
            mock_agent.component_health = {"pubmed_integration": {"status": "ready"}}
            mock_build.return_value = mock_agent

            system = build_integrated_system(docs_folder=temp_docs_folder, api_key="test_key", config=mock_config)

            # Should return all components
            assert "rag_agent" in system
            assert "pubmed_scraper" in system
            assert "query_engine" in system
            assert "config" in system
            assert "status" in system

            # Verify components
            assert system["rag_agent"] == mock_agent
            assert system["pubmed_scraper"] == mock_scraper
            assert system["query_engine"] == mock_engine
            assert system["config"] == mock_config
            assert system["status"]["pubmed_enabled"] == True
            assert system["status"]["hybrid_mode"] == True
            assert system["status"]["components_ready"] == True

    def test_build_without_pubmed(self, temp_docs_folder, mock_config):
        """Test building system without PubMed integration."""
        mock_config.should_enable_pubmed.return_value = False

        with patch("src.pharmaceutical_query_adapter.build_enhanced_rag_agent") as mock_build:
            mock_agent = Mock()
            mock_agent._pubmed_available.return_value = False
            mock_build.return_value = mock_agent

            system = build_integrated_system(docs_folder=temp_docs_folder, api_key="test_key", config=mock_config)

            # Should not include PubMed components
            assert "pubmed_scraper" not in system
            assert "query_engine" not in system
            assert system["status"]["pubmed_enabled"] == False
            assert system["status"]["components_ready"] == False

    def test_build_passes_parameters(self, temp_docs_folder, mock_config):
        """Test that all parameters are passed correctly."""
        with patch("src.pharmaceutical_query_adapter.build_enhanced_rag_agent") as mock_build:
            build_integrated_system(
                docs_folder=temp_docs_folder,
                api_key="test_key",
                vector_db_path="./test_db",
                chunk_size=500,
                chunk_overlap=100,
                embedding_model_name="test-model",
                guardrails_config_path="test.yaml",
                enable_synthesis=True,
                enable_ddi_analysis=True,
                safety_mode="strict",
                config=mock_config,
            )

            # Verify parameters passed to build_enhanced_rag_agent
            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args[1]
            assert call_kwargs["docs_folder"] == temp_docs_folder
            assert call_kwargs["api_key"] == "test_key"
            assert call_kwargs["vector_db_path"] == "./test_db"
            assert call_kwargs["chunk_size"] == 500
            assert call_kwargs["chunk_overlap"] == 100
            assert call_kwargs["embedding_model_name"] == "test-model"
            assert call_kwargs["guardrails_config_path"] == "test.yaml"
            assert call_kwargs["enable_synthesis"] == True
            assert call_kwargs["enable_ddi_analysis"] == True
            assert call_kwargs["safety_mode"] == "strict"
            assert call_kwargs["config"] == mock_config


class TestCheckSystemHealth:
    """Test system health checking functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=EnhancedRAGConfig)
        config.should_enable_pubmed.return_value = True
        return config

    def test_healthy_system(self, mock_config):
        """Test health check for healthy system."""
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

        mock_engine = Mock()

        report = check_system_health(
            rag_agent=mock_agent, pubmed_scraper=mock_scraper, query_engine=mock_engine, config=mock_config
        )

        assert isinstance(report, SystemHealthReport)
        assert report.rag_agent_ready == True
        assert report.pubmed_integration_ready == True
        assert report.pubmed_scraper_ready == True
        assert report.query_engine_ready == True
        assert report.guardrails_ready == True
        assert report.synthesis_ready == True
        assert report.ddi_analysis_ready == True
        assert len(report.errors) == 0

    def test_missing_components(self, mock_config):
        """Test health check with missing components."""
        report = check_system_health(rag_agent=None, pubmed_scraper=None, query_engine=None, config=mock_config)

        assert report.rag_agent_ready == False
        assert report.pubmed_integration_ready == False
        assert report.pubmed_scraper_ready == False
        assert report.query_engine_ready == False
        assert len(report.warnings) > 0
        assert any("No RAG agent" in w for w in report.warnings)

    def test_unhealthy_components(self, mock_config):
        """Test health check with unhealthy components."""
        # Create mock components with issues
        mock_agent = Mock()
        mock_agent._ensure_components_initialized.side_effect = Exception("Init failed")
        mock_agent.component_health = {
            "pubmed_integration": {"status": "error"},
            "medical_guardrails": {"status": "not_ready"},
        }

        mock_scraper = Mock()
        mock_scraper.combined_status_report.side_effect = Exception("Scraper error")

        report = check_system_health(rag_agent=mock_agent, pubmed_scraper=mock_scraper, config=mock_config)

        assert report.rag_agent_ready == False
        assert report.pubmed_integration_ready == False
        assert report.pubmed_scraper_ready == False
        assert len(report.errors) > 0

    def test_pubmed_disabled_but_components_provided(self, mock_config):
        """Test health check when PubMed disabled but components provided."""
        mock_config.should_enable_pubmed.return_value = False

        mock_agent = Mock()
        mock_agent.component_health = {"pubmed_integration": {"status": "ready"}}

        mock_scraper = Mock()
        mock_engine = Mock()

        report = check_system_health(
            rag_agent=mock_agent, pubmed_scraper=mock_scraper, query_engine=mock_engine, config=mock_config
        )

        # Components should be considered ready even if disabled
        assert report.query_engine_ready == True
        assert len(report.errors) == 0


class TestSystemHealthReport:
    """Test SystemHealthReport dataclass."""

    def test_default_initialization(self):
        """Test default initialization of health report."""
        report = SystemHealthReport()

        assert report.rag_agent_ready == False
        assert report.pubmed_integration_ready == False
        assert report.pubmed_scraper_ready == False
        assert report.query_engine_ready == False
        assert report.guardrails_ready == False
        assert report.synthesis_ready == False
        assert report.ddi_analysis_ready == False
        assert report.warnings == []
        assert report.errors == []

    def test_custom_initialization(self):
        """Test custom initialization of health report."""
        report = SystemHealthReport(
            rag_agent_ready=True, pubmed_integration_ready=True, warnings=["Test warning"], errors=["Test error"]
        )

        assert report.rag_agent_ready == True
        assert report.pubmed_integration_ready == True
        assert report.warnings == ["Test warning"]
        assert report.errors == ["Test error"]

    def test_post_init_initializes_lists(self):
        """Test that post-init initializes lists."""
        report = SystemHealthReport(warnings=None, errors=None)

        assert report.warnings == []
        assert report.errors == []


if __name__ == "__main__":
    pytest.main([__file__])
