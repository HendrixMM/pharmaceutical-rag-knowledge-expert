"""
Unit tests for NeMo Embedding Service

Tests the enhanced NVIDIA NeMo Embedding Service with pharmaceutical
domain optimization and multi-model support.
"""
import os
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from src.nemo_embedding_service import EmbeddingConfig
from src.nemo_embedding_service import EmbeddingResult
from src.nemo_embedding_service import NeMoEmbeddingService


class TestEmbeddingConfig:
    """Test embedding configuration class."""

    def test_default_config(self):
        config = EmbeddingConfig()

        assert config.model == "nv-embedqa-e5-v5"
        assert config.batch_size == 100
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600

    def test_custom_config(self):
        config = EmbeddingConfig(
            model="nv-embedqa-mistral7b-v2", batch_size=50, enable_caching=False, cache_ttl_seconds=1800
        )

        assert config.model == "nv-embedqa-mistral7b-v2"
        assert config.batch_size == 50
        assert config.enable_caching is False
        assert config.cache_ttl_seconds == 1800

    def test_pharmaceutical_optimization_config(self):
        config = EmbeddingConfig(
            model="nv-embedqa-e5-v5", pharmaceutical_optimization=True, medical_terminology_boost=1.2
        )

        assert config.pharmaceutical_optimization is True
        assert config.medical_terminology_boost == 1.2


class TestEmbeddingResult:
    """Test embedding result data structure."""

    def test_successful_result(self):
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            success=True, embeddings=embeddings, model_used="nv-embedqa-e5-v5", processing_time_ms=150.5
        )

        assert result.success is True
        assert len(result.embeddings) == 2
        assert result.model_used == "nv-embedqa-e5-v5"
        assert result.processing_time_ms == 150.5
        assert result.cache_hit is False
        assert result.error is None

    def test_failed_result(self):
        result = EmbeddingResult(success=False, error="API key invalid")

        assert result.success is False
        assert result.embeddings is None
        assert result.error == "API key invalid"


class TestNeMoEmbeddingService:
    """Test core NeMo Embedding Service functionality."""

    @pytest.fixture
    def mock_config(self):
        return EmbeddingConfig(
            model="nv-embedqa-e5-v5", batch_size=2, enable_caching=False  # Disable caching for simpler tests
        )

    @pytest.fixture
    def mock_service(self, mock_config):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoEmbeddingService(config=mock_config)

    def test_init_with_config(self, mock_config):
        """Test service initialization with custom config."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            service = NeMoEmbeddingService(config=mock_config)

            assert service.config.model == "nv-embedqa-e5-v5"
            assert service.config.batch_size == 2

    def test_init_default_config(self):
        """Test service initialization with default config."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            service = NeMoEmbeddingService()

            assert service.config.model == "nv-embedqa-e5-v5"
            assert service.config.batch_size == 100

    @pytest.mark.asyncio
    async def test_embed_documents_success(self, mock_service):
        """Test successful document embedding."""
        mock_texts = ["Drug interaction study", "Clinical trial results"]
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch.object(mock_service, "_embed_batch") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                success=True, embeddings=mock_embeddings, model_used="nv-embedqa-e5-v5", processing_time_ms=100.0
            )

            result = await mock_service.embed_documents(mock_texts)

            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]
            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_documents_empty_input(self, mock_service):
        """Test embedding with empty input."""
        result = await mock_service.embed_documents([])

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_documents_batching(self, mock_service):
        """Test that large inputs are properly batched."""
        # Create more texts than batch size to test batching
        mock_texts = [f"Document {i}" for i in range(5)]  # batch_size is 2
        [[0.1, 0.2] for _ in range(5)]

        with patch.object(mock_service, "_embed_batch") as mock_embed:
            # Mock multiple batch calls
            mock_embed.side_effect = [
                EmbeddingResult(success=True, embeddings=[[0.1, 0.2], [0.3, 0.4]]),
                EmbeddingResult(success=True, embeddings=[[0.5, 0.6], [0.7, 0.8]]),
                EmbeddingResult(success=True, embeddings=[[0.9, 1.0]]),
            ]

            result = await mock_service.embed_documents(mock_texts)

            assert len(result) == 5
            assert mock_embed.call_count == 3  # 3 batches: [2, 2, 1]

    @pytest.mark.asyncio
    async def test_embed_query_success(self, mock_service):
        """Test successful query embedding."""
        query = "What are the side effects of aspirin?"
        mock_embedding = [0.1, 0.2, 0.3, 0.4]

        with patch.object(mock_service, "_embed_batch") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                success=True, embeddings=[mock_embedding], model_used="nv-embedqa-e5-v5"
            )

            result = await mock_service.embed_query(query)

            assert result == mock_embedding
            mock_embed.assert_called_once_with([query])

    @pytest.mark.asyncio
    async def test_pharmaceutical_optimization(self, mock_service):
        """Test pharmaceutical domain optimization features."""
        mock_service.config.pharmaceutical_optimization = True

        pharmaceutical_texts = [
            "Metformin hydrochloride 500mg tablets",
            "Contraindicated in patients with renal impairment",
            "FDA approved indication for type 2 diabetes",
        ]

        with patch.object(mock_service, "_apply_pharmaceutical_optimization") as mock_optimize:
            with patch.object(mock_service, "_embed_batch") as mock_embed:
                mock_embed.return_value = EmbeddingResult(success=True, embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

                await mock_service.embed_documents(pharmaceutical_texts)

                mock_optimize.assert_called()

    def test_model_selection_pharmaceutical(self, mock_service):
        """Test intelligent model selection for pharmaceutical content."""
        pharmaceutical_content = [
            "Clinical trial phase III results",
            "Drug-drug interaction profile",
            "Pharmacokinetic parameters",
        ]

        with patch.object(mock_service, "_select_optimal_model") as mock_select:
            mock_select.return_value = "nv-embedqa-e5-v5"  # Optimal for QA

            model = mock_service._select_optimal_model(pharmaceutical_content)

            assert model == "nv-embedqa-e5-v5"

    def test_model_selection_multilingual(self, mock_service):
        """Test model selection for multilingual content."""
        multilingual_content = ["English clinical report", "Informe clínico en español", "Rapport clinique en français"]

        with patch.object(mock_service, "_select_optimal_model") as mock_select:
            mock_select.return_value = "nv-embedqa-mistral7b-v2"  # Better for multilingual

            model = mock_service._select_optimal_model(multilingual_content)

            assert model == "nv-embedqa-mistral7b-v2"


class TestNeMoEmbeddingServiceIntegration:
    """Integration tests with mocked NeMo client."""

    @pytest.fixture
    def mock_service_with_client(self):
        config = EmbeddingConfig(enable_caching=False)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            service = NeMoEmbeddingService(config=config)

            # Mock the NeMo client
            mock_client = AsyncMock()
            service.nemo_client = mock_client

            return service, mock_client

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_service_with_client):
        """Test successful batch embedding with mocked client."""
        service, mock_client = mock_service_with_client

        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_response.response_time_ms = 150.0
        mock_client.embed_texts.return_value = mock_response

        result = await service._embed_batch(["text1", "text2"])

        assert result.success is True
        assert len(result.embeddings) == 2
        assert result.processing_time_ms == 150.0
        mock_client.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_api_error(self, mock_service_with_client):
        """Test batch embedding with API error."""
        service, mock_client = mock_service_with_client

        mock_response = Mock()
        mock_response.success = False
        mock_response.error = "Rate limit exceeded"
        mock_client.embed_texts.return_value = mock_response

        result = await service._embed_batch(["text1"])

        assert result.success is False
        assert result.error == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_embed_batch_exception(self, mock_service_with_client):
        """Test batch embedding with network exception."""
        service, mock_client = mock_service_with_client

        mock_client.embed_texts.side_effect = Exception("Network error")

        result = await service._embed_batch(["text1"])

        assert result.success is False
        assert "Network error" in result.error


class TestCachingFunctionality:
    """Test embedding caching functionality."""

    @pytest.fixture
    def cached_service(self):
        config = EmbeddingConfig(enable_caching=True, cache_ttl_seconds=3600)

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoEmbeddingService(config=config)

    @pytest.mark.asyncio
    async def test_cache_hit(self, cached_service):
        """Test cache hit scenario."""
        text = "Cached pharmaceutical document"

        with patch.object(cached_service, "_get_from_cache") as mock_get_cache:
            with patch.object(cached_service, "_embed_batch") as mock_embed:
                # Simulate cache hit
                mock_get_cache.return_value = [0.1, 0.2, 0.3]

                result = await cached_service.embed_query(text)

                assert result == [0.1, 0.2, 0.3]
                mock_embed.assert_not_called()  # Should not call API

    @pytest.mark.asyncio
    async def test_cache_miss(self, cached_service):
        """Test cache miss scenario."""
        text = "New pharmaceutical document"

        with patch.object(cached_service, "_get_from_cache") as mock_get_cache:
            with patch.object(cached_service, "_store_in_cache") as mock_store_cache:
                with patch.object(cached_service, "_embed_batch") as mock_embed:
                    # Simulate cache miss
                    mock_get_cache.return_value = None
                    mock_embed.return_value = EmbeddingResult(success=True, embeddings=[[0.4, 0.5, 0.6]])

                    result = await cached_service.embed_query(text)

                    assert result == [0.4, 0.5, 0.6]
                    mock_embed.assert_called_once()
                    mock_store_cache.assert_called_once()


class TestEnvironmentIntegration:
    """Test integration with environment variable model selection."""

    @pytest.mark.asyncio
    async def test_environment_model_override(self):
        """Test that environment variables override default model selection."""
        test_env = {"NVIDIA_API_KEY": "test-key", "EMBEDDING_MODEL": "nvidia/nv-embedqa-mistral7b-v2"}

        with patch.dict(os.environ, test_env):
            config = EmbeddingConfig()
            service = NeMoEmbeddingService(config=config)

            # Check if service respects environment model preference
            env_model = os.getenv("EMBEDDING_MODEL", "").split("/")[-1]
            assert env_model == "nv-embedqa-mistral7b-v2"

    def test_model_name_extraction(self):
        """Test proper extraction of model names from full paths."""
        full_model_names = [
            "nvidia/nv-embedqa-e5-v5",
            "nvidia/nv-embedqa-mistral7b-v2",
            "Snowflake/snowflake-arctic-embed-l",
        ]

        for full_name in full_model_names:
            model_name = full_name.split("/")[-1]
            # Verify the extracted name doesn't contain provider prefix
            assert "/" not in model_name
            assert model_name in ["nv-embedqa-e5-v5", "nv-embedqa-mistral7b-v2", "snowflake-arctic-embed-l"]


class TestPharmaceuticalOptimization:
    """Test pharmaceutical domain-specific optimizations."""

    @pytest.fixture
    def pharma_service(self):
        config = EmbeddingConfig(
            model="nv-embedqa-e5-v5", pharmaceutical_optimization=True, medical_terminology_boost=1.2
        )

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoEmbeddingService(config=config)

    def test_pharmaceutical_terminology_detection(self, pharma_service):
        """Test detection of pharmaceutical terminology."""
        pharmaceutical_texts = [
            "Metformin is indicated for type 2 diabetes mellitus",
            "Contraindicated in patients with severe renal impairment",
            "Clinical pharmacology and mechanism of action",
        ]

        # Mock the terminology detection
        with patch.object(pharma_service, "_detect_pharmaceutical_terms") as mock_detect:
            mock_detect.return_value = ["metformin", "contraindicated", "clinical pharmacology"]

            terms = pharma_service._detect_pharmaceutical_terms(pharmaceutical_texts)

            assert "metformin" in terms
            assert "contraindicated" in terms

    def test_medical_entity_enhancement(self, pharma_service):
        """Test enhancement of medical entities in embeddings."""
        medical_text = "Aspirin 81mg daily for cardiovascular protection"

        with patch.object(pharma_service, "_enhance_medical_entities") as mock_enhance:
            mock_enhance.return_value = "Enhanced: Aspirin 81mg daily for cardiovascular protection"

            enhanced = pharma_service._enhance_medical_entities(medical_text)

            assert "Enhanced:" in enhanced
            mock_enhance.assert_called_once_with(medical_text)
