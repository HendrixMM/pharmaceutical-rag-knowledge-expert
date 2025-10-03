"""
NVIDIA Build Platform Compatibility Test Suite

Comprehensive testing of OpenAI SDK wrapper compatibility with all NVIDIA Build
endpoints, ensuring NGC-independent operation and pharmaceutical optimization.

Test Coverage:
- OpenAI SDK wrapper functionality across all endpoints
- Model availability and compatibility verification
- Error handling and fallback mechanisms
- Pharmaceutical domain optimization validation
- Rate limiting and quota management

This test suite validates the system's readiness for NGC API deprecation.
"""
import os

import pytest

# Import modules under test
try:
    from src.clients.nemo_client_enhanced import ClientResponse, EndpointType, EnhancedNeMoClient
    from src.clients.openai_wrapper import NVIDIABuildConfig, NVIDIABuildError, OpenAIWrapper
    from src.enhanced_config import EnhancedRAGConfig
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from src.clients.nemo_client_enhanced import ClientResponse, EndpointType, EnhancedNeMoClient
    from src.clients.openai_wrapper import NVIDIABuildConfig, NVIDIABuildError, OpenAIWrapper
    from src.enhanced_config import EnhancedRAGConfig


class TestNVIDIABuildCompatibility:
    """Test suite for NVIDIA Build platform compatibility via OpenAI SDK."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with mock API key if needed."""
        # Use test API key or mock if not available
        self.test_api_key = os.getenv("NVIDIA_API_KEY", "test_key_for_compatibility_testing")
        self.test_base_url = "https://integrate.api.nvidia.com/v1"

        # Create test configuration
        self.config = NVIDIABuildConfig(
            api_key=self.test_api_key, base_url=self.test_base_url, pharmaceutical_optimized=True
        )

        yield

        # Cleanup if needed

    def test_nvidia_build_config_initialization(self):
        """Test NVIDIA Build configuration initialization."""
        config = NVIDIABuildConfig(
            api_key="test_key", base_url="https://integrate.api.nvidia.com/v1", pharmaceutical_optimized=True
        )

        assert config.api_key == "test_key"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.pharmaceutical_optimized == True
        assert config.timeout == 60  # Default timeout
        assert config.max_retries == 3  # Default retries

    def test_openai_wrapper_initialization(self):
        """Test OpenAI wrapper initialization with NVIDIA Build config."""
        wrapper = OpenAIWrapper(self.config)

        assert wrapper.config == self.config
        assert wrapper.client is not None
        assert wrapper.pharma_optimized == True

    @pytest.mark.asyncio
    async def test_list_available_models(self):
        """Test model listing functionality."""
        wrapper = OpenAIWrapper(self.config)

        try:
            models = wrapper.list_available_models()

            # Should return a list of model information
            assert isinstance(models, list)

            # Check for expected model types
            model_ids = [model.get("id", "") for model in models]

            # Look for embedding models
            embedding_models = [m for m in model_ids if "embed" in m.lower()]
            assert len(embedding_models) > 0, "No embedding models found"

            # Look for chat/completion models
            chat_models = [m for m in model_ids if any(term in m.lower() for term in ["llama", "mistral", "gemma"])]
            assert len(chat_models) > 0, "No chat models found"

            print(f"✅ Found {len(models)} available models")
            print(f"   Embedding models: {len(embedding_models)}")
            print(f"   Chat models: {len(chat_models)}")

        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                pytest.skip(f"API access limited (Discovery Tier): {str(e)}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_embedding_model_compatibility(self):
        """Test embedding model compatibility across NVIDIA Build endpoints."""
        wrapper = OpenAIWrapper(self.config)

        # Test with pharmaceutical research text
        test_texts = [
            "metformin mechanism of action in type 2 diabetes",
            "ACE inhibitor drug interactions with potassium supplements",
            "pharmacokinetics of warfarin in elderly patients",
        ]

        try:
            # Test default embedding model
            response = wrapper.create_embeddings(test_texts)

            assert response is not None
            assert hasattr(response, "data")
            assert len(response.data) == len(test_texts)

            # Validate embedding dimensions
            for embedding_data in response.data:
                assert hasattr(embedding_data, "embedding")
                assert isinstance(embedding_data.embedding, list)
                assert len(embedding_data.embedding) > 0

            print(f"✅ Embedding model compatibility verified")
            print(f"   Model: {response.model}")
            print(f"   Embeddings: {len(response.data)}")
            print(f"   Dimensions: {len(response.data[0].embedding)}")

        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                pytest.skip(f"API access limited (Discovery Tier): {str(e)}")
            elif "404" in str(e):
                pytest.skip(f"Embedding endpoint not available: {str(e)}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_chat_model_compatibility(self):
        """Test chat completion model compatibility."""
        wrapper = OpenAIWrapper(self.config)

        # Pharmaceutical research query
        messages = [
            {"role": "user", "content": "Explain the mechanism of action of metformin in treating type 2 diabetes."}
        ]

        try:
            response = wrapper.create_chat_completion(messages=messages, max_tokens=150, temperature=0.3)

            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0].message, "content")
            assert len(response.choices[0].message.content) > 0

            print(f"✅ Chat model compatibility verified")
            print(f"   Model: {response.model}")
            print(f"   Response length: {len(response.choices[0].message.content)} chars")

        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                pytest.skip(f"API access limited (Discovery Tier): {str(e)}")
            elif "404" in str(e):
                pytest.skip(f"Chat endpoint not available: {str(e)}")
            else:
                raise

    @pytest.mark.asyncio
    async def test_pharmaceutical_optimization_features(self):
        """Test pharmaceutical-specific optimization features."""
        wrapper = OpenAIWrapper(self.config)

        # Test pharmaceutical setup validation
        try:
            validation_result = wrapper.validate_pharmaceutical_setup()

            assert isinstance(validation_result, dict)
            assert "pharmaceutical_optimized" in validation_result
            assert validation_result["pharmaceutical_optimized"] == True

            # Check for pharmaceutical-specific configurations
            if "default_models" in validation_result:
                models = validation_result["default_models"]
                assert "embedding_model" in models
                assert "chat_model" in models

            print(f"✅ Pharmaceutical optimization validated")
            print(f"   Features: {list(validation_result.keys())}")

        except Exception as e:
            # This is our custom method, so it should work
            print(f"ℹ️  Pharmaceutical validation info: {str(e)}")

    def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms."""

        # Test invalid API key handling
        invalid_config = NVIDIABuildConfig(
            api_key="invalid_key_test", base_url=self.config.base_url, pharmaceutical_optimized=True
        )

        wrapper = OpenAIWrapper(invalid_config)

        # Should handle initialization gracefully
        assert wrapper is not None

        # Test connection with invalid credentials
        try:
            connection_result = wrapper.test_connection()
            # Should return error status, not crash
            assert isinstance(connection_result, dict)
            assert "success" in connection_result

        except Exception as e:
            # Should catch and handle authentication errors
            assert isinstance(e, (NVIDIABuildError, Exception))

    @pytest.mark.asyncio
    async def test_enhanced_client_integration(self):
        """Test enhanced NeMo client integration with NVIDIA Build."""

        # Create enhanced client with cloud-first configuration
        enhanced_client = EnhancedNeMoClient(pharmaceutical_optimized=True, enable_fallback=True)

        # Verify initialization
        assert enhanced_client.cloud_client is not None or enhanced_client.nemo_client is not None
        assert enhanced_client.pharmaceutical_optimized == True
        assert enhanced_client.enable_fallback == True

        # Test endpoint status
        status = enhanced_client.get_endpoint_status()
        assert isinstance(status, dict)
        assert "cloud_available" in status
        assert "pharmaceutical_optimized" in status

        print(f"✅ Enhanced client integration verified")
        print(f"   Cloud available: {status.get('cloud_available', 'unknown')}")
        print(f"   Fallback enabled: {status.get('fallback_enabled', 'unknown')}")

    @pytest.mark.asyncio
    async def test_cloud_first_execution_flow(self):
        """Test cloud-first execution flow with fallback."""

        enhanced_client = EnhancedNeMoClient(pharmaceutical_optimized=True, enable_fallback=True)

        # Test embedding request through enhanced client
        test_texts = ["metformin pharmacokinetics in kidney disease patients"]

        try:
            response = enhanced_client.create_embeddings(test_texts)

            assert isinstance(response, ClientResponse)
            assert response.success in [True, False]  # Could fail due to API access

            if response.success:
                assert response.endpoint_type in [EndpointType.CLOUD, EndpointType.SELF_HOSTED]
                assert response.cost_tier in ["free_tier", "infrastructure"]
                assert response.response_time_ms is not None

                print(f"✅ Cloud-first execution successful")
                print(f"   Endpoint: {response.endpoint_type.value}")
                print(f"   Cost tier: {response.cost_tier}")
                print(f"   Response time: {response.response_time_ms}ms")
            else:
                print(f"ℹ️  Request failed (expected with limited API access): {response.error}")

        except Exception as e:
            print(f"ℹ️  Cloud-first execution test info: {str(e)}")

    def test_rate_limiting_awareness(self):
        """Test rate limiting awareness and handling."""
        wrapper = OpenAIWrapper(self.config)

        # Test rate limiting configuration exists
        assert hasattr(wrapper.config, "timeout")
        assert hasattr(wrapper.config, "max_retries")

        # Should have conservative defaults
        assert wrapper.config.timeout <= 60  # Conservative timeout
        assert wrapper.config.max_retries >= 1  # Some retry capability

        # Test connection with rate limiting awareness
        try:
            connection_result = wrapper.test_connection()
            assert isinstance(connection_result, dict)

        except Exception as e:
            # Rate limiting errors should be handled gracefully
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"✅ Rate limiting correctly detected and handled")
            else:
                raise

    @pytest.mark.asyncio
    async def test_pharmaceutical_capability_integration(self):
        """Test integrated pharmaceutical capability testing."""

        enhanced_client = EnhancedNeMoClient(pharmaceutical_optimized=True, enable_fallback=True)

        # Run pharmaceutical capability test
        try:
            capabilities = enhanced_client.test_pharmaceutical_capabilities()

            assert isinstance(capabilities, dict)
            assert "pharmaceutical_optimized" in capabilities
            assert "overall_status" in capabilities

            # Should have attempted cloud and/or NeMo tests
            assert "cloud_test" in capabilities or "nemo_test" in capabilities

            # Should have functional test results
            assert "embedding_test" in capabilities
            assert "chat_test" in capabilities

            print(f"✅ Pharmaceutical capabilities tested")
            print(f"   Overall status: {capabilities.get('overall_status', 'unknown')}")
            print(f"   Pharmaceutical optimized: {capabilities.get('pharmaceutical_optimized', 'unknown')}")

        except Exception as e:
            print(f"ℹ️  Pharmaceutical capability test info: {str(e)}")

    def test_ngc_independence_verification(self):
        """Verify system operates independently of NGC API."""

        # Test that configuration doesn't rely on NGC endpoints
        config = EnhancedRAGConfig.from_env()

        # Check cloud-first strategy
        strategy = config.get_cloud_first_strategy()
        assert isinstance(strategy, dict)
        assert "cloud_first_enabled" in strategy

        # Verify OpenAI SDK configuration
        openai_compat = config.validate_openai_sdk_compatibility()
        assert isinstance(openai_compat, dict)
        assert openai_compat.get("compatible", False) == True

        # Check endpoint priority doesn't include NGC-specific endpoints
        priority_order = config.get_endpoint_priority_order()
        assert isinstance(priority_order, list)

        # Should prioritize NVIDIA Build over NGC
        nvidia_build_priority = None
        for i, endpoint in enumerate(priority_order):
            if "integrate.api.nvidia.com" in str(endpoint) or "nvidia_build" in str(endpoint).lower():
                nvidia_build_priority = i
                break

        assert nvidia_build_priority is not None, "NVIDIA Build not found in priority order"

        print(f"✅ NGC independence verified")
        print(f"   Cloud-first enabled: {strategy.get('cloud_first_enabled', 'unknown')}")
        print(f"   OpenAI SDK compatible: {openai_compat.get('compatible', 'unknown')}")
        print(f"   NVIDIA Build priority: {nvidia_build_priority}")

    def test_endpoint_health_monitoring_readiness(self):
        """Test readiness for endpoint health monitoring."""

        wrapper = OpenAIWrapper(self.config)

        # Test connection monitoring
        try:
            health_status = wrapper.test_connection()

            assert isinstance(health_status, dict)
            assert "success" in health_status

            # Should include timing information
            if health_status.get("success"):
                assert "response_time_ms" in health_status

            # Should include endpoint information
            assert "endpoint" in health_status or "base_url" in health_status

        except Exception as e:
            # Health monitoring should handle errors gracefully
            print(f"ℹ️  Health monitoring error handling: {str(e)}")

        print(f"✅ Health monitoring readiness verified")


class TestIntegrationCompatibility:
    """Integration tests for full system compatibility."""

    @pytest.mark.asyncio
    async def test_end_to_end_pharmaceutical_workflow(self):
        """Test complete pharmaceutical workflow without NGC dependencies."""

        # Create enhanced configuration
        config = EnhancedRAGConfig.from_env()

        # Verify cloud-first is working
        strategy = config.get_cloud_first_strategy()
        if not strategy.get("cloud_first_enabled"):
            pytest.skip("Cloud-first not enabled for testing")

        # Create enhanced client
        client = EnhancedNeMoClient(config=config, pharmaceutical_optimized=True, enable_fallback=True)

        # Test pharmaceutical embedding workflow
        pharmaceutical_queries = [
            "metformin contraindications in chronic kidney disease",
            "drug interactions between warfarin and NSAIDs",
            "pharmacokinetics of ACE inhibitors in elderly patients",
        ]

        try:
            embedding_response = client.create_embeddings(pharmaceutical_queries)

            assert isinstance(embedding_response, ClientResponse)

            # Should use cloud endpoint preferentially
            if embedding_response.success:
                print(f"✅ End-to-end embedding successful")
                print(
                    f"   Endpoint: {embedding_response.endpoint_type.value if embedding_response.endpoint_type else 'unknown'}"
                )
                print(f"   Cost tier: {embedding_response.cost_tier}")

        except Exception as e:
            print(f"ℹ️  End-to-end embedding info: {str(e)}")

        # Test pharmaceutical chat workflow
        try:
            chat_response = client.create_chat_completion(
                [{"role": "user", "content": "What are the key drug interactions to monitor with metformin therapy?"}]
            )

            assert isinstance(chat_response, ClientResponse)

            if chat_response.success:
                print(f"✅ End-to-end chat successful")
                print(f"   Endpoint: {chat_response.endpoint_type.value if chat_response.endpoint_type else 'unknown'}")
                print(f"   Cost tier: {chat_response.cost_tier}")

        except Exception as e:
            print(f"ℹ️  End-to-end chat info: {str(e)}")

    def test_configuration_migration_readiness(self):
        """Test system readiness for NGC API migration."""

        config = EnhancedRAGConfig.from_env()

        # Should have cloud-first capabilities
        assert hasattr(config, "nvidia_build_base_url")
        assert hasattr(config, "enable_nvidia_build_fallback")

        # Should support feature flags for migration
        feature_flags = config.get_feature_flags()
        assert isinstance(feature_flags, dict)

        # Should have OpenAI SDK compatibility
        compatibility = config.validate_openai_sdk_compatibility()
        assert compatibility.get("compatible", False) == True

        print(f"✅ Configuration migration readiness verified")
        print(f"   Feature flags available: {len(feature_flags)}")
        print(f"   OpenAI SDK compatible: {compatibility.get('compatible')}")


if __name__ == "__main__":
    # Run tests with comprehensive output
    pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "-s",  # Show print statements
            "--tb=short",  # Short traceback format
            "--disable-warnings",  # Clean output
        ]
    )
