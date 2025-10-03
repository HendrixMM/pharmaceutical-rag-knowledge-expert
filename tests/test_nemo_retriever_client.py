"""
Unit tests for NeMo Retriever Client

Tests the three-step NeMo Retriever pipeline:
1. Extraction - Document processing and text extraction
2. Embedding - Text to vector conversion
3. Reranking - Relevance-based reordering

These tests use mocks to avoid requiring live API access during testing.
"""
import os
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from src.nemo_retriever_client import create_nemo_client
from src.nemo_retriever_client import NeMoRetrieverClient
from src.nemo_retriever_client import NVIDIABuildCreditsMonitor


class TestNVIDIABuildCreditsMonitor:
    """Test credits monitoring for free tier usage."""

    def test_init_with_api_key(self):
        monitor = NVIDIABuildCreditsMonitor("test-key")
        assert monitor.api_key == "test-key"
        assert monitor.credits_used == 0
        assert monitor.credits_remaining == 10000

    def test_log_api_call_basic(self):
        monitor = NVIDIABuildCreditsMonitor("test-key")
        monitor.log_api_call("embedding", tokens_used=5)

        assert monitor.credits_used == 5
        assert monitor.credits_remaining == 9995

    def test_log_api_call_multiple(self):
        monitor = NVIDIABuildCreditsMonitor("test-key")
        monitor.log_api_call("embedding", tokens_used=100)
        monitor.log_api_call("reranking", tokens_used=50)

        assert monitor.credits_used == 150
        assert monitor.credits_remaining == 9850

    def test_log_api_call_low_credits_warning(self, caplog):
        monitor = NVIDIABuildCreditsMonitor("test-key")
        monitor.credits_used = 9950  # Set high usage
        monitor.credits_remaining = 50

        monitor.log_api_call("reranking", tokens_used=10)

        assert "Low NVIDIA Build credits remaining" in caplog.text


class TestNeMoRetrieverClient:
    """Test core NeMo Retriever client functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked dependencies."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            client = NeMoRetrieverClient()
            return client

    def test_init_with_api_key(self):
        client = NeMoRetrieverClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-key"

    def test_init_from_env_var(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"}):
            client = NeMoRetrieverClient()
            assert client.api_key == "env-key"

    def test_init_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="NVIDIA API key is required"):
                NeMoRetrieverClient()

    def test_available_models(self, mock_client):
        """Test that all expected models are available."""
        # Test embedding models
        assert "nv-embedqa-e5-v5" in mock_client.EMBEDDING_MODELS
        assert "nv-embedqa-mistral7b-v2" in mock_client.EMBEDDING_MODELS
        assert "snowflake-arctic-embed-l" in mock_client.EMBEDDING_MODELS

        # Test reranking models
        assert "nv-rerankqa-mistral4b-v3" in mock_client.RERANKING_MODELS
        assert "llama-3_2-nemoretriever-500m-rerank-v2" in mock_client.RERANKING_MODELS

    def test_model_recommendation_pharmaceutical(self, mock_client):
        """Test model recommendations for pharmaceutical use case."""
        recommendation = mock_client.recommend_model(use_case="pharmaceutical_qa", content_type="medical")

        assert recommendation["embedding"] == "nv-embedqa-e5-v5"
        # Newer default for pharmaceutical reranking prioritizes latest llama-based model
        assert recommendation["reranking"] == "llama-3_2-nemoretriever-500m-rerank-v2"
        assert "pharmaceutical" in recommendation["reasoning"].lower()

    def test_model_recommendation_multilingual(self, mock_client):
        """Test model recommendations for multilingual content."""
        recommendation = mock_client.recommend_model(use_case="multilingual_search", content_type="multi")

        assert recommendation["embedding"] == "nv-embedqa-mistral7b-v2"
        assert "multilingual" in recommendation["reasoning"].lower()


class TestNeMoEmbeddingOperations:
    """Test embedding operations with mocked API responses."""

    @pytest.fixture
    def mock_client(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoRetrieverClient()

    @pytest.mark.asyncio
    async def test_embed_texts_success(self, mock_client):
        """Test successful text embedding."""
        mock_response_data = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await mock_client.embed_texts(texts=["test text 1", "test text 2"], model="nv-embedqa-e5-v5")

            assert result.success is True
            assert "embeddings" in result.data
            assert len(result.data["embeddings"]) == 2
            assert result.service == "embedding"
            assert result.model == "nv-embedqa-e5-v5"

    @pytest.mark.asyncio
    async def test_embed_texts_api_error(self, mock_client):
        """Test embedding with API error response."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text.return_value = "Bad Request"
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await mock_client.embed_texts(texts=["test text"], model="nv-embedqa-e5-v5")

            assert result.success is False
            assert "400" in result.error

    @pytest.mark.asyncio
    async def test_embed_texts_with_credits_monitor(self, mock_client):
        """Test embedding with credits monitoring."""
        mock_client.credits_monitor = NVIDIABuildCreditsMonitor("test-key")

        mock_response_data = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response

            await mock_client.embed_texts(texts=["test text"], model="nv-embedqa-e5-v5")

            assert mock_client.credits_monitor.credits_used == 1


class TestNeMoRerankingOperations:
    """Test reranking operations with mocked API responses."""

    @pytest.fixture
    def mock_client(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoRetrieverClient()

    @pytest.mark.asyncio
    async def test_rerank_passages_success(self, mock_client):
        """Test successful passage reranking."""
        mock_response_data = {
            "rankings": [
                {"text": "passage 1", "score": 0.9, "index": 0},
                {"text": "passage 2", "score": 0.7, "index": 1},
            ]
        }

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await mock_client.rerank_passages(
                query="drug interactions",
                passages=["passage 1", "passage 2"],
                model="llama-3_2-nemoretriever-500m-rerank-v2",
            )

            assert result.success is True
            assert "reranked_passages" in result.data
            assert len(result.data["reranked_passages"]) == 2
            assert result.service == "reranking"

    @pytest.mark.asyncio
    async def test_rerank_passages_with_top_k(self, mock_client):
        """Test reranking with top_k parameter."""
        mock_response_data = {"rankings": [{"text": "passage 1", "score": 0.9, "index": 0}]}

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await mock_client.rerank_passages(
                query="test query",
                passages=["passage 1", "passage 2", "passage 3"],
                model="llama-3_2-nemoretriever-500m-rerank-v2",
                top_k=1,
            )

            assert result.success is True
            assert len(result.data["reranked_passages"]) == 1

    @pytest.mark.asyncio
    async def test_rerank_passages_invalid_model(self, mock_client):
        """Test reranking with invalid model name."""
        with pytest.raises(ValueError, match="Unknown reranking model"):
            await mock_client.rerank_passages(query="test", passages=["passage"], model="invalid-model")


class TestNeMoHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def mock_client(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            return NeMoRetrieverClient()

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_client):
        """Test health check when all services are healthy."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            health = await mock_client.health_check()

            assert "embedding" in health
            assert "reranking" in health
            assert "extraction" in health

            for service_health in health.values():
                assert service_health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_some_unhealthy(self, mock_client):
        """Test health check when some services are unhealthy."""

        def mock_get_side_effect(*args, **kwargs):
            mock_response = AsyncMock()
            url = args[0] if args else ""

            if "embedding" in url:
                mock_response.status = 200  # Healthy
            else:
                mock_response.status = 503  # Unhealthy

            return mock_response.__aenter__()

        with patch("aiohttp.ClientSession.get", side_effect=mock_get_side_effect):
            health = await mock_client.health_check()

            assert health["embedding"]["status"] == "healthy"
            assert health["reranking"]["status"] == "unhealthy"
            assert health["extraction"]["status"] == "unhealthy"


class TestCreateNemoClient:
    """Test the convenience factory function."""

    @pytest.mark.asyncio
    async def test_create_nemo_client_success(self):
        """Test successful client creation with health check."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            with patch("src.nemo_retriever_client.NeMoRetrieverClient.health_check") as mock_health:
                mock_health.return_value = {
                    "embedding": {"status": "healthy"},
                    "reranking": {"status": "healthy"},
                    "extraction": {"status": "healthy"},
                }

                client = await create_nemo_client()

                assert isinstance(client, NeMoRetrieverClient)
                assert client.api_key == "test-key"
                mock_health.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_create_nemo_client_with_credits_monitor(self):
        """Test client creation with credits monitoring."""
        monitor = NVIDIABuildCreditsMonitor("test-key")

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            with patch("src.nemo_retriever_client.NeMoRetrieverClient.health_check") as mock_health:
                mock_health.return_value = {"embedding": {"status": "healthy"}}

                client = await create_nemo_client(credits_monitor=monitor)

                assert client.credits_monitor is monitor


class TestEnvironmentVariableIntegration:
    """Test integration with environment variables for model selection."""

    @pytest.mark.asyncio
    async def test_environment_model_configuration(self):
        """Test that environment variables are properly used for model selection."""
        test_env = {
            "NVIDIA_API_KEY": "test-key",
            "EMBEDDING_MODEL": "nvidia/nv-embedqa-e5-v5",
            "RERANK_MODEL": "llama-3_2-nemoretriever-500m-rerank-v2",
        }

        with patch.dict(os.environ, test_env):
            client = NeMoRetrieverClient()

            # Verify the models are available
            assert "nv-embedqa-e5-v5" in client.EMBEDDING_MODELS
            assert "llama-3_2-nemoretriever-500m-rerank-v2" in client.RERANKING_MODELS

            # Test model extraction from environment
            embedding_model = os.getenv("EMBEDDING_MODEL", "").split("/")[-1]
            rerank_model = os.getenv("RERANK_MODEL", "")

            assert embedding_model == "nv-embedqa-e5-v5"
            assert rerank_model == "llama-3_2-nemoretriever-500m-rerank-v2"
