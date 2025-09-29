"""
NVIDIA Build Platform Client (DEPRECATED)

DEPRECATION NOTICE: This client is deprecated. Use OpenAIWrapper directly for new code.
This wrapper is maintained for backward compatibility only and delegates all operations
to OpenAIWrapper for consistency and reduced maintenance overhead.

For new implementations, use:
    from src.clients.openai_wrapper import OpenAIWrapper, NVIDIABuildConfig

Legacy Features:
- OpenAI SDK compatibility for NVIDIA Build endpoints
- Cost-effective cloud-first approach with free tier support
- Credit usage monitoring and optimization
- Pharmaceutical domain model support
- Future-proof architecture (NGC-independent)

Usage (DEPRECATED):
    from src.nvidia_build_client import NVIDBuildClient  # historical name
    # or
    from src.nvidia_build_client import NVIDIABuildClient  # backward-compatible alias

    client = NVIDBuildClient()

    # Embedding usage
    response = client.create_embeddings(
        texts=["pharmaceutical research query"],
        model="nvidia/nv-embed-v1"
    )

    # Chat completion usage
    response = client.create_chat_completion(
        messages=[{"role": "user", "content": "What are drug interactions?"}],
        model="meta/llama-3.1-8b-instruct"
    )
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from src.clients.openai_wrapper import OpenAIWrapper, NVIDIABuildConfig, NVIDIABuildError, OPENAI_AVAILABLE
except ImportError:
    try:
        from .clients.openai_wrapper import OpenAIWrapper, NVIDIABuildConfig, NVIDIABuildError, OPENAI_AVAILABLE
    except ImportError:
        # If wrapper itself is not available, this is a real error
        raise ImportError(
            "OpenAIWrapper not available. Ensure src.clients.openai_wrapper is installed."
        )

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

logger = logging.getLogger(__name__)

@dataclass
class CreditUsageTracker:
    """Legacy credit tracker for backward compatibility."""

    # Free tier limits (NVIDIA Build platform)
    monthly_free_requests: int = 10000
    daily_alert_threshold: float = 0.1  # 10% of monthly in a day
    weekly_alert_threshold: float = 0.25  # 25% of monthly in a week
    monthly_alert_threshold: float = 0.8  # 80% of monthly limit

    # Usage tracking
    requests_today: int = 0
    requests_this_week: int = 0
    requests_this_month: int = 0

    # Rate limiting baseline
    requests_per_minute_baseline: int = 60  # Conservative estimate

    last_reset_day: Optional[str] = None
    last_reset_week: Optional[str] = None
    last_reset_month: Optional[str] = None

    def track_request(self) -> None:
        """Track a new API request and check thresholds."""
        today = datetime.now().strftime('%Y-%m-%d')
        this_week = datetime.now().strftime('%Y-W%U')
        this_month = datetime.now().strftime('%Y-%m')

        # Reset counters if needed
        if self.last_reset_day != today:
            self.requests_today = 0
            self.last_reset_day = today

        if self.last_reset_week != this_week:
            self.requests_this_week = 0
            self.last_reset_week = this_week

        if self.last_reset_month != this_month:
            self.requests_this_month = 0
            self.last_reset_month = this_month

        # Increment counters
        self.requests_today += 1
        self.requests_this_week += 1
        self.requests_this_month += 1

        # Check alert thresholds
        self._check_usage_alerts()

    def _check_usage_alerts(self) -> None:
        """Check usage against alert thresholds."""
        daily_limit = self.monthly_free_requests * self.daily_alert_threshold
        weekly_limit = self.monthly_free_requests * self.weekly_alert_threshold
        monthly_limit = self.monthly_free_requests * self.monthly_alert_threshold

        if self.requests_today >= daily_limit:
            logger.warning(f"Daily credit usage alert: {self.requests_today} requests today "
                         f"(>{daily_limit:.0f} threshold)")

        if self.requests_this_week >= weekly_limit:
            logger.warning(f"Weekly credit usage alert: {self.requests_this_week} requests this week "
                         f"(>{weekly_limit:.0f} threshold)")

        if self.requests_this_month >= monthly_limit:
            logger.warning(f"Monthly credit usage alert: {self.requests_this_month} requests this month "
                         f"(>{monthly_limit:.0f} threshold)")

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current usage summary with safe division for usage percent."""
        limit = max(0, self.monthly_free_requests)
        usage_pct = (self.requests_this_month / limit * 100) if limit > 0 else 0
        return {
            "requests_today": self.requests_today,
            "requests_this_week": self.requests_this_week,
            "requests_this_month": self.requests_this_month,
            "monthly_limit": self.monthly_free_requests,
            "daily_burn_rate": self.requests_today,
            "weekly_burn_rate": self.requests_this_week,
            "monthly_usage_percent": usage_pct,
            "estimated_monthly_projection": self.requests_today * 30 if self.requests_today > 0 else 0
        }

class NVIDBuildClient:
    """
    DEPRECATED: NVIDIA Build Platform client delegating to OpenAIWrapper.

    This client is maintained for backward compatibility only.
    For new code, use OpenAIWrapper directly from src.clients.openai_wrapper.

    Provides cost-effective, cloud-first access to NVIDIA models with:
    - Free tier credit monitoring (10K requests/month)
    - NGC deprecation immunity (OpenAI SDK based)
    - Pharmaceutical domain model support
    - Intelligent request optimization
    """

    # NVIDIA Build endpoints
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    # Model catalog is sourced from OpenAIWrapper to avoid duplication

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 enable_credit_monitoring: bool = True,
                 pharmaceutical_optimized: bool = True):
        """
        Initialize NVIDIA Build client (DEPRECATED).

        Args:
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
            base_url: Custom base URL (defaults to NVIDIA Build)
            enable_credit_monitoring: Enable free tier credit tracking
            pharmaceutical_optimized: Use pharmaceutical-optimized defaults
        """
        logger.warning(
            "NVIDBuildClient is deprecated. Use OpenAIWrapper directly for new code. "
            "See src.clients.openai_wrapper.OpenAIWrapper"
        )

        # Create NVIDIABuildConfig from parameters
        config = NVIDIABuildConfig(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            pharmaceutical_optimized=pharmaceutical_optimized,
            enable_cost_per_query_tracking=enable_credit_monitoring
        )

        # Delegate to OpenAIWrapper
        self.wrapper = OpenAIWrapper(config)

        # Maintain credit tracker for legacy compatibility
        self.credit_tracker = CreditUsageTracker() if enable_credit_monitoring else None
        self.pharmaceutical_optimized = pharmaceutical_optimized

        logger.info(f"NVIDIA Build client initialized (delegating to OpenAIWrapper)")
        if self.pharmaceutical_optimized:
            logger.info("Pharmaceutical optimization enabled")

        # Log OpenAI SDK availability status for self-hosted installations
        if not OPENAI_AVAILABLE:
            logger.info(
                "OpenAI SDK not available - cloud operations will fail with clear error messages. "
                "This is normal for self-hosted installations. "
                "Install OpenAI SDK for cloud functionality: pip install 'openai>=1.0.0,<2.0.0'"
            )

    def create_embeddings(self,
                         texts: List[str],
                         model: str = "nvidia/nv-embed-v1",
                         encoding_format: str = "float",
                         **kwargs):
        """
        Create embeddings using NVIDIA Build models (delegates to OpenAIWrapper).

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            encoding_format: Encoding format for embeddings
            **kwargs: Additional parameters for embedding creation

        Returns:
            CreateEmbeddingResponse from OpenAI SDK
        """
        if self.credit_tracker:
            self.credit_tracker.track_request()

        # Pharmaceutical optimization: prefer Q&A optimized model
        if self.pharmaceutical_optimized and model == "nvidia/nv-embed-v1":
            try:
                from src.clients.openai_wrapper import get_model_catalog  # type: ignore
            except Exception:
                from .clients.openai_wrapper import get_model_catalog  # type: ignore
            catalog = get_model_catalog()
            if "nvidia/nv-embedqa-e5-v5" in catalog.get("embedding", {}):
                logger.info("Using pharmaceutical-optimized embedding model: nvidia/nv-embedqa-e5-v5")
                model = "nvidia/nv-embedqa-e5-v5"

        try:
            response = self.wrapper.create_embeddings(
                texts=texts,
                model=model,
                encoding_format=encoding_format,
                **kwargs
            )

            logger.info(f"Embeddings created successfully via wrapper: {len(texts)} texts, "
                       f"model: {model}")

            return response

        except NVIDIABuildError as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            raise Exception(str(e))  # Convert to generic Exception for legacy compatibility
        except Exception as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            raise

    def create_chat_completion(self,
                              messages: List[Dict[str, str]],
                              model: str = "meta/llama-3.1-8b-instruct",
                              max_tokens: int = 1000,
                              temperature: float = 0.1,
                              **kwargs):
        """
        Create chat completion using NVIDIA Build models (delegates to OpenAIWrapper).

        Args:
            messages: List of message dictionaries
            model: Chat model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters for chat completion

        Returns:
            ChatCompletion from OpenAI SDK
        """
        if self.credit_tracker:
            self.credit_tracker.track_request()

        # Pharmaceutical optimization: conservative temperature for medical accuracy
        if self.pharmaceutical_optimized and temperature > 0.2:
            logger.info("Pharmaceutical optimization: reducing temperature for medical accuracy")
            temperature = 0.1

        try:
            response = self.wrapper.create_chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            logger.info(f"Chat completion successful via wrapper: model: {model}")

            return response

        except NVIDIABuildError as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise Exception(str(e))  # Convert to generic Exception for legacy compatibility
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    def test_model_access(self, model_type: str = "embedding") -> Dict[str, Any]:
        """
        Test access to specific model types.

        Args:
            model_type: Type of model to test ("embedding" or "chat")

        Returns:
            Dictionary with test results
        """
        results = {
            "model_type": model_type,
            "success": False,
            "error": None,
            "response_time_ms": None,
            "model_tested": None
        }

        start_time = time.time()

        try:
            if model_type == "embedding":
                model = "nvidia/nv-embed-v1"
                response = self.create_embeddings(
                    texts=["Test pharmaceutical research query"],
                    model=model
                )
                results["model_tested"] = model
                results["dimensions"] = len(response.data[0].embedding) if response.data else 0
                results["success"] = True

            elif model_type == "chat":
                model = "meta/llama-3.1-8b-instruct"
                response = self.create_chat_completion(
                    messages=[{"role": "user", "content": "What are drug interactions? Answer in one sentence."}],
                    model=model,
                    max_tokens=50
                )
                results["model_tested"] = model
                results["response"] = response.choices[0].message.content if response.choices else ""
                results["success"] = True

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            results["error"] = str(e)
            results["success"] = False

        finally:
            results["response_time_ms"] = int((time.time() - start_time) * 1000)

        return results

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current credit usage summary."""
        if not self.credit_tracker:
            return {"credit_monitoring": "disabled"}

        # Merge legacy tracker with wrapper metrics
        legacy_summary = self.credit_tracker.get_usage_summary()
        wrapper_metrics = self.wrapper.get_cost_metrics()

        return {
            "credit_monitoring": "enabled",
            **legacy_summary,
            "wrapper_metrics": wrapper_metrics
        }

    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available pharmaceutical models."""
        try:
            from src.clients.openai_wrapper import get_model_catalog  # type: ignore
        except Exception:
            from .clients.openai_wrapper import get_model_catalog  # type: ignore
        catalog = get_model_catalog()

        wrapper_info = self.wrapper.get_model_info()

        return {
            "base_url": self.wrapper.config.base_url,
            "pharmaceutical_optimized": self.pharmaceutical_optimized,
            "models": catalog,
            "wrapper_info": wrapper_info
        }

# Backward-compatible alias for historical imports
NVIDIABuildClient = NVIDBuildClient

# Convenience function for quick testing
def test_nvidia_build_access() -> Dict[str, Any]:
    """
    Quick test function for NVIDIA Build API access (DEPRECATED).

    Returns:
        Dictionary with comprehensive test results
    """
    logger.warning("test_nvidia_build_access is deprecated. Use OpenAIWrapper.test_connection() instead.")

    try:
        client = NVIDBuildClient()

        # Test both embedding and chat models
        embedding_results = client.test_model_access("embedding")
        chat_results = client.test_model_access("chat")

        return {
            "client_initialized": True,
            "embedding_test": embedding_results,
            "chat_test": chat_results,
            "usage_summary": client.get_usage_summary(),
            "available_models": client.get_available_models()
        }

    except Exception as e:
        return {
            "client_initialized": False,
            "error": str(e),
            "embedding_test": {"success": False},
            "chat_test": {"success": False}
        }

if __name__ == "__main__":
    # Quick test when run directly
    import json
    results = test_nvidia_build_access()
    print(json.dumps(results, indent=2))
