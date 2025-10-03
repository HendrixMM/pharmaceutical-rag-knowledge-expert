"""
NVIDIA Embedding Integration for RAG Agent
Handles connection to NVIDIA LLaMA 3.2 NemoRetriever embedding model

Note: This implementation uses direct HTTP calls instead of langchain_nvidia_ai_endpoints
for the following reasons:
1. Fine-grained control over retry/fallback logic with model-specific error detection
2. Custom probing behavior for model availability during initialization
3. Sophisticated model name normalization and automatic fallback handling
4. Direct control over batch processing and embedding dimension caching
5. Specialized rate limiting and error handling for NVIDIA API patterns

While langchain_nvidia_ai_endpoints is available in requirements.txt, the direct approach
provides better control over the embedding lifecycle and error scenarios.
"""
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logger = logging.getLogger(__name__)


def _env_true(name: str, default: bool = True) -> bool:
    """Interpret boolean-like environment flags consistently."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


class NVIDIAEmbeddings(Embeddings):
    """NVIDIA LLaMA 3.2 NemoRetriever embedding model integration"""

    # Known model name aliases to canonical names
    MODEL_ALIASES = {
        "nvidia/llama-3_2-nemoretriever-1b-vlm-embed-v1": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "nvidia/llama-3-2-nemoretriever-1b-vlm-embed-v1": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "nvidia/llama3.2-nemoretriever-1b-vlm-embed-v1": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "llama-3_2-nemoretriever-1b-vlm-embed-v1": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        "llama-3-2-nemoretriever-1b-vlm-embed-v1": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        # Add more aliases here as needed
    }

    # Model capabilities based on model family patterns
    MODEL_CAPABILITIES = {
        # NemoRetriever models support input_type and don't need encoding_format
        "nemoretriever": {
            "supports_input_type": True,
            "supports_encoding_format": False,
            "supports_truncate": True,
            "default_truncate": "END",
        },
        # NV-Embed models support encoding_format but not input_type
        "nv-embed": {
            "supports_input_type": False,
            "supports_encoding_format": True,
            "supports_truncate": False,
            "default_encoding_format": "float",
        },
        # Default capabilities for unknown models
        "default": {
            "supports_input_type": False,
            "supports_encoding_format": True,
            "supports_truncate": False,
            "default_encoding_format": "float",
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 10,
        probe_on_init: Optional[bool] = None,
        fallback_model_name: Optional[str] = None,
    ):
        """
        Initialize NVIDIA embeddings

        Args:
            api_key: NVIDIA API key
            embedding_model_name: Name of the embedding model (optional)
            base_url: Base URL for NVIDIA API (overridden by EMBEDDING_BASE_URL when set)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            batch_size: Number of texts to process in each batch (default: 10)
            probe_on_init: Whether to probe the model during initialization. When None,
                uses the EMBEDDING_PROBE_ON_INIT environment variable (defaults to True).
            fallback_model_name: Fallback model name (optional). Defaults to
                EMBEDDING_FALLBACK_MODEL environment variable or "nvidia/nv-embed-v1".
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key is required. Set NVIDIA_API_KEY environment variable.")

        # Set preferred model from parameter, environment, or default
        provided_model_name = (
            embedding_model_name
            or os.getenv("EMBEDDING_MODEL_NAME")
            or "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
        )
        self.model_name = self._normalize_model_name(provided_model_name)
        fallback_model_name = fallback_model_name or os.getenv("EMBEDDING_FALLBACK_MODEL", "nvidia/nv-embed-v1")
        self._fallback_model = self._normalize_model_name(fallback_model_name)
        self.model_selection_reason = "preferred"

        # Log both provided and normalized model names if different
        if provided_model_name != self.model_name:
            logger.info(f"Normalized model name: '{provided_model_name}' -> '{self.model_name}'")
        else:
            logger.info(f"Using model name: '{self.model_name}'")

        env_base_url = os.getenv("EMBEDDING_BASE_URL")
        effective_base_url = env_base_url.strip() if env_base_url else base_url
        if not effective_base_url:
            effective_base_url = "https://integrate.api.nvidia.com/v1"
        self.base_url = effective_base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set batch size from parameter or environment variable
        env_batch_size = os.getenv("EMBEDDING_BATCH_SIZE")
        if env_batch_size:
            try:
                self.batch_size = int(env_batch_size)
                logger.info(f"Using batch size from environment: {self.batch_size}")
            except ValueError:
                logger.warning(f"Invalid EMBEDDING_BATCH_SIZE '{env_batch_size}', using default: {batch_size}")
                self.batch_size = batch_size
        else:
            self.batch_size = batch_size

        # Set probe_on_init from parameter or environment variable
        falsey_values = {"false", "0", "no", "off"}
        if probe_on_init is not None:
            self.probe_on_init = probe_on_init
        else:
            env_probe_on_init = os.getenv("EMBEDDING_PROBE_ON_INIT")
            if env_probe_on_init is None:
                self.probe_on_init = True
            else:
                self.probe_on_init = env_probe_on_init.strip().lower() not in falsey_values

        # Set timeouts from environment variables
        self.probe_timeout = int(os.getenv("EMBEDDING_PROBE_TIMEOUT_SECONDS", "30"))
        self.request_timeout = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))

        # Track fallback status to avoid infinite loops
        self._already_fallback = False

        # Cache for embedding dimension to avoid duplicate network calls
        self._dimension: Optional[int] = None

        # Check if input_type feature is enabled
        self.input_type_enabled = _env_true("EMBEDDING_INPUT_TYPE_ENABLED", False)

        # Detect model family for capability mapping
        self.model_family = self._detect_model_family(self.model_name)
        self.model_capabilities = self._get_model_capabilities(self.model_family)

        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Traycer-RAG/1.0 (+https://traycer.ai)",
        }

        # Set up HTTP session with retry adapter
        self._session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update(self.headers)

        # Probe and set model with fallback (if enabled)
        if self.probe_on_init:
            self._probe_and_set_model()
            logger.info(f"Initialized NVIDIA embeddings with final model: {self.model_name}")
        else:
            logger.info(f"Initialized NVIDIA embeddings with model (no probe): {self.model_name}")
            logger.info("Model probing disabled - first embedding request will validate model availability")

    def _is_nvidia_embedding_model(self, model_name: str) -> bool:
        """
        Check if a model name is a NVIDIA embedding model that should be normalized.

        Args:
            model_name: The model name to check

        Returns:
            bool: True if this is a NVIDIA embedding model
        """
        # Must start with 'nvidia/' or be a known alias
        if model_name.startswith("nvidia/"):
            return True

        # Check if it's a known alias that maps to a NVIDIA model
        if model_name in self.MODEL_ALIASES:
            canonical_name = self.MODEL_ALIASES[model_name]
            if canonical_name.startswith("nvidia/"):
                return True

        # Check for NVIDIA embedding model patterns
        embedding_indicators = ["nemoretriever", "nv-embed", "embed-v1"]
        has_embedding_indicator = any(indicator in model_name.lower() for indicator in embedding_indicators)

        # Additional check: must be from a known NVIDIA embedding family
        nvidia_families = ["llama", "nemoretriever", "nv-embed"]
        has_nvidia_family = any(family in model_name.lower() for family in nvidia_families)

        return has_embedding_indicator and has_nvidia_family

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name by applying known aliases and converting recognized model-name patterns.
        Restricts normalization to NVIDIA embedding models only to avoid unintended rewrites.

        Args:
            model_name: Raw model name that might contain variants

        Returns:
            Normalized model name
        """
        # First, check if this is a known alias
        if model_name in self.MODEL_ALIASES:
            canonical_name = self.MODEL_ALIASES[model_name]
            logger.info(f"Normalized model alias: '{model_name}' -> '{canonical_name}'")
            return canonical_name

        # Only apply normalization to NVIDIA embedding models
        if not self._is_nvidia_embedding_model(model_name):
            # For non-NVIDIA models, return unchanged to avoid unintended rewrites
            return model_name

        # Only apply normalization to recognized LLaMA model name patterns
        # Pattern: "llama-X_Y" where X and Y are digits
        if re.match(r".*llama-\d+_\d+.*", model_name, re.IGNORECASE):
            # Convert underscore to dot in version numbers for LLaMA models
            # e.g., "llama-3_2" -> "llama-3.2"
            normalized = re.sub(r"(llama-\d+)_(\d+)", r"\1.\2", model_name, flags=re.IGNORECASE)
            return normalized

        # Handle compact variants such as "llama3.2" or "llama32"
        compact_pattern = re.compile(r"(llama)[\-_]?(\d)(?:[\.\-_]?(\d))?", re.IGNORECASE)
        match = compact_pattern.search(model_name)
        if match:
            prefix, major, minor = match.groups()
            if minor:
                normalized_variant = f"{prefix.lower()}-{major}.{minor}"
            else:
                normalized_variant = f"{prefix.lower()}-{major}"
            candidate = compact_pattern.sub(normalized_variant, model_name, count=1)
            logger.debug("Normalized compact model name variant '%s' -> '%s'", model_name, candidate)
            return candidate

        # For other model names, return unchanged
        return model_name

    def _detect_model_family(self, model_name: str) -> str:
        """
        Detect the model family based on model name patterns.

        Args:
            model_name: Normalized model name

        Returns:
            Model family key for MODEL_CAPABILITIES lookup
        """
        model_name_lower = model_name.lower()

        if "nemoretriever" in model_name_lower:
            return "nemoretriever"
        elif "nv-embed" in model_name_lower:
            return "nv-embed"
        else:
            return "default"

    def _get_model_capabilities(self, model_family: str) -> Dict:
        """
        Get capabilities for a model family.

        Args:
            model_family: Model family key

        Returns:
            Dictionary of model capabilities
        """
        return self.MODEL_CAPABILITIES.get(model_family, self.MODEL_CAPABILITIES["default"])

    def _parse_embeddings_response(self, json_result: Dict) -> List[List[float]]:
        """
        Parse embeddings response and cache dimension.

        Args:
            json_result: JSON response from NVIDIA API

        Returns:
            List of embedding vectors
        """
        if not isinstance(json_result, dict) or "data" not in json_result:
            raise Exception(f"Unexpected embeddings response format: {json_result}")

        try:
            embeddings = [item["embedding"] for item in json_result["data"]]
        except Exception as parsing_error:
            logger.error("Malformed embedding payload: %s", parsing_error)
            raise

        # Cache embedding dimension on first successful request
        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])
            logger.debug(f"Cached embedding dimension: {self._dimension}")

        return embeddings

    def _extract_model_unavailable_reason(self, response: requests.Response) -> Optional[str]:
        """Return reason string when response indicates the model is unavailable."""
        try:
            error_response = response.json()
        except json.JSONDecodeError:
            logger.debug("Could not decode error response JSON while evaluating fallback criteria.")
            return None

        error_block = error_response.get("error")

        if isinstance(error_block, dict):
            message = error_block.get("message", "")
            code = error_block.get("code", "")
        else:
            message = str(error_block) if error_block is not None else ""
            code = ""

        combined = f"{code} {message}".lower()

        fallback_indicators = [
            "unknown model",
            "model not found",
            "not available",
            "does not exist",
            "unsupported model",
            "disabled",
            "retired",
            "not enabled",
            "access denied for model",
            "unsupported",
        ]

        for indicator in fallback_indicators:
            if indicator in combined:
                reason = message or (code if code else indicator)
                logger.debug(f"Fallback indicator '{indicator}' found in error response.")
                return reason

        code_lower = str(code).lower()
        fallback_codes = {
            "model_not_found",
            "model_not_available",
            "model_disabled",
            "model_retired",
            "model_not_enabled",
            "access_denied_for_model",
            "model_archived",
        }

        if code_lower in fallback_codes:
            reason = message or code
            logger.debug("Fallback triggered by error code '%s'.", code_lower)
            return reason

        if response.status_code in (422, 501):
            reason = message or code or f"http_{response.status_code}"
            logger.debug(
                "Treating HTTP %s as model unavailability due to message '%s'",
                response.status_code,
                message,
            )
            return reason

        return None

    def _probe_and_set_model(self):
        """
        Probe the preferred model with retry/backoff and fallback to EMBEDDING_FALLBACK_MODEL if needed
        """
        preferred = self.model_name
        fallback = self._fallback_model

        # Try probe with retry/backoff like _make_request
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Probing model {preferred} (attempt {attempt + 1}/{self.max_retries})")

                # Try a quick request with preferred model
                url = f"{self.base_url}/embeddings"
                payload = self._build_probe_payload(self.model_name)

                response = self._session.post(url, json=payload, timeout=self.probe_timeout)

                if response.status_code == 200:
                    logger.info(f"Successfully using preferred model: {preferred}")
                    self.model_selection_reason = "preferred"
                    return
                elif response.status_code in (401, 403):  # Authentication/Authorization errors
                    logger.error(f"Authentication error during probe (status {response.status_code}): {response.text}")
                    raise Exception(
                        f"Authentication failed with status {response.status_code}. Check your NVIDIA API key."
                    )
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limited during probe. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [400, 404, 422, 501]:
                    reason = self._extract_model_unavailable_reason(response)

                    if reason:
                        logger.warning(f"Fallback triggered during probe: {reason}")
                        logger.warning(f"Falling back to {fallback}")
                        self.model_name = fallback
                        self._dimension = None
                        logger.debug("Embedding dimension cache cleared after probe fallback")
                        self._already_fallback = True
                        self.model_selection_reason = reason
                        logger.info(f"Probe-based permanent fallback to default model: {self.model_name}")
                        return

                    logger.warning(f"Model probe received {response.status_code} with non-fallback error.")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        break
                else:
                    # For 5xx errors and other status codes, treat as transient
                    logger.warning(f"Model probe failed with status {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        break

            except (requests.exceptions.RequestException, Exception) as e:
                logger.warning(f"Model probe attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break

        # If we reach here, all preferred-model probes hit transient errors; attempt a final soft probe with fallback
        logger.warning(
            f"Model probe failed after {self.max_retries} attempts with transient errors. Keeping model: {preferred}"
        )
        try:
            url = f"{self.base_url}/embeddings"
            payload = self._build_probe_payload(fallback)
            response = self._session.post(url, json=payload, timeout=self.probe_timeout)
            if response.status_code == 200:
                logger.info(
                    "Fallback model '%s' responded successfully during soft probe. Will continue with preferred model but fallback is warm.",
                    fallback,
                )
            else:
                logger.debug(
                    "Fallback soft probe returned status %s: %s",
                    response.status_code,
                    response.text,
                )
        except Exception as soft_probe_error:
            logger.debug("Fallback soft probe failed: %s", soft_probe_error)

        logger.info("First embedding request will validate model availability with fallback if needed")

    def _make_request(self, texts: List[str], input_type: Optional[str] = None) -> List[List[float]]:
        """
        Make request to NVIDIA API for embeddings

        Args:
            texts: List of texts to embed
            input_type: Optional type specification ("query" or "passage") to avoid scoring drift

        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/embeddings"

        # Build payload based on model capabilities
        payload = {"input": texts, "model": self.model_name}

        # Add encoding_format if supported by model
        if self.model_capabilities["supports_encoding_format"]:
            payload["encoding_format"] = self.model_capabilities.get("default_encoding_format", "float")
            logger.debug(f"Using encoding_format='{payload['encoding_format']}' for {self.model_family}")

        # Add input_type if supported by model and enabled
        if self.model_capabilities["supports_input_type"] and self.input_type_enabled and input_type:
            payload["input_type"] = input_type
            logger.debug(f"Using input_type='{input_type}' for {self.model_family}")

        # Add truncate if supported by model
        if self.model_capabilities["supports_truncate"]:
            payload["truncate"] = self.model_capabilities.get("default_truncate", "END")
            logger.debug(f"Using truncate='{payload['truncate']}' for {self.model_family}")

        logger.debug(f"Built payload for {self.model_family} model: {list(payload.keys())}")

        last_fallback_error: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Making embedding request with model '{self.model_name}' (attempt {attempt + 1}/{self.max_retries})"
                )

                response = self._session.post(url, json=payload, timeout=self.request_timeout)

                if response.status_code == 200:
                    result = response.json()
                    embeddings = self._parse_embeddings_response(result)
                    logger.debug(f"Successfully got embeddings for {len(texts)} texts using model '{self.model_name}'")
                    return embeddings

                elif response.status_code in (401, 403):  # Authentication/Authorization errors
                    logger.error(f"Authentication error (status {response.status_code}): {response.text}")
                    raise Exception(
                        f"Authentication failed with status {response.status_code}. Check your NVIDIA API key."
                    )
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                else:
                    handled = False

                    if response.status_code in (400, 404, 422):
                        if not self._already_fallback:
                            reason = self._extract_model_unavailable_reason(response)
                            if reason:
                                handled = True
                                logger.warning(f"Fallback triggered for request: {reason}")
                                logger.warning(f"Falling back to {self._fallback_model}")
                                self.model_name = self._fallback_model
                                self._dimension = None
                                logger.debug("Embedding dimension cache cleared after request-triggered fallback")
                                self._already_fallback = True
                                self.model_selection_reason = reason

                                # Update model family and capabilities after fallback
                                self.model_family = self._detect_model_family(self.model_name)
                                self.model_capabilities = self._get_model_capabilities(self.model_family)
                                logger.debug(f"Updated model family to '{self.model_family}' after fallback")

                                payload["model"] = self.model_name

                                # Rebuild payload with new model capabilities
                                payload = {"input": texts, "model": self.model_name}

                                # Add encoding_format if supported by fallback model
                                if self.model_capabilities["supports_encoding_format"]:
                                    payload["encoding_format"] = self.model_capabilities.get(
                                        "default_encoding_format", "float"
                                    )

                                # Add truncate if supported by fallback model
                                if self.model_capabilities["supports_truncate"]:
                                    payload["truncate"] = self.model_capabilities.get("default_truncate", "END")

                                logger.debug(
                                    f"Rebuilt payload for fallback model '{self.model_family}': {list(payload.keys())}"
                                )

                                logger.info(f"Retrying request with fallback model: {self.model_name}")

                                fallback_response = self._session.post(url, json=payload, timeout=self.request_timeout)

                                if fallback_response.status_code == 200:
                                    result = fallback_response.json()
                                    embeddings = self._parse_embeddings_response(result)
                                    logger.info(f"Successfully got embeddings using fallback model '{self.model_name}'")
                                    self.model_selection_reason = reason or f"fallback_status_{response.status_code}"
                                    return embeddings

                                last_fallback_error = (
                                    f"Fallback model '{self.model_name}' failed with status "
                                    f"{fallback_response.status_code}. Response: {fallback_response.text}"
                                )
                                logger.error(last_fallback_error)

                                if attempt < self.max_retries - 1:
                                    logger.warning(
                                        "Continuing retries with fallback model '%s' despite previous failure (attempt %s of %s).",
                                        self.model_name,
                                        attempt + 2,
                                        self.max_retries,
                                    )
                                    time.sleep(self.retry_delay)
                                    continue

                                raise Exception(last_fallback_error)

                            else:
                                response_snippet = response.text[:500]
                                logger.info(
                                    "Received status %s without fallback reason; raising to avoid masking upstream errors. Response: %s",
                                    response.status_code,
                                    response_snippet,
                                )
                                raise Exception(
                                    f"Embedding request failed with status {response.status_code} and did not provide a fallback reason. Response: {response_snippet}"
                                )
                        else:
                            response_snippet = response.text[:500]
                            logger.error(
                                "Embedding request failed with status %s while already on fallback model '%s': %s",
                                response.status_code,
                                self.model_name,
                                response_snippet,
                            )
                            raise Exception(
                                f"Embedding request failed with status {response.status_code} while using fallback model {self.model_name}. Response: {response_snippet}"
                            )

                    if handled:
                        continue

                    if response.status_code in (400, 404, 422, 501):
                        error_message = response.text
                        logger.error(
                            f"Embedding request failed with status {response.status_code} (no fallback reason): {error_message}"
                        )
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        raise Exception(
                            f"Embedding request failed with status {response.status_code}. Response: {error_message}"
                        )

                    logger.warning(
                        "Embedding request returned transient status %s: %s",
                        response.status_code,
                        response.text,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        break

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise

        if last_fallback_error:
            raise Exception(last_fallback_error)

        raise Exception(f"Failed to get embeddings after {self.max_retries} attempts")

    def _build_probe_payload(self, model_name: str) -> dict:
        """
        Build a capability-aware probe payload for model availability checking.

        Args:
            model_name: The model name to build payload for

        Returns:
            dict: Capability-aware probe payload
        """
        payload = {"input": ["probe"], "model": model_name}

        # Get model family for the model being probed
        model_family = self._detect_model_family(model_name)
        capabilities = self.MODEL_CAPABILITIES.get(model_family, self.MODEL_CAPABILITIES["default"])

        # Add encoding_format if supported by model
        if capabilities["supports_encoding_format"]:
            payload["encoding_format"] = capabilities.get("default_encoding_format", "float")
            logger.debug(f"Using encoding_format='{payload['encoding_format']}' for {model_family} probe")

        # Add input_type if supported by model (use 'query' as default for probes)
        if capabilities["supports_input_type"]:
            payload["input_type"] = "query"
            logger.debug(f"Using input_type='query' for {model_family} probe")

        # Add truncate if supported by model
        if capabilities["supports_truncate"]:
            payload["truncate"] = capabilities.get("default_truncate", "END")
            logger.debug(f"Using truncate='{payload['truncate']}' for {model_family} probe")

        logger.debug(f"Built probe payload for {model_family}: {list(payload.keys())}")
        return payload

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} documents")

        # Process in batches to avoid API limits
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(
                f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}"
            )

            # Use "passage" as input_type for documents when enabled
            input_type = "passage" if self.input_type_enabled else None
            batch_embeddings = self._make_request(batch, input_type=input_type)
            all_embeddings.extend(batch_embeddings)

            # Small delay between batches to be respectful to the API
            if i + self.batch_size < len(texts):
                time.sleep(0.1)

        logger.info(f"Successfully embedded {len(all_embeddings)} documents")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        logger.debug("Embedding query text")
        # Use "query" as input_type for queries when enabled
        input_type = "query" if self.input_type_enabled else None
        embeddings = self._make_request([text], input_type=input_type)
        if not embeddings:
            raise Exception("Empty embedding response for query")
        return embeddings[0]

    def test_connection(self) -> bool:
        """
        Test the connection to NVIDIA API

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            logger.info("Testing NVIDIA API connection...")
            test_embedding = self.embed_query("test")
            logger.info(f"Connection successful! Embedding dimension: {len(test_embedding)}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of embeddings from this model

        Returns:
            Embedding dimension, or None if it cannot be determined
        """
        # Return cached dimension if available
        if self._dimension is not None:
            logger.debug(f"Using cached embedding dimension: {self._dimension}")
            return self._dimension

        try:
            test_embedding = self.embed_query("test")
            dimension = len(test_embedding)
            # Cache the dimension for future use
            self._dimension = dimension
            logger.debug(f"Determined and cached embedding dimension: {dimension}")
            return dimension
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {str(e)}")
            return None


def main():
    """Test the NVIDIA embeddings"""
    from dotenv import load_dotenv

    load_dotenv()

    # Unit-like assertions to demonstrate non-embed names are not altered
    print("Testing model name normalization...")
    test_embeddings = NVIDIAEmbeddings(probe_on_init=False)

    # Test that non-embedding model names are not altered
    non_embed_names = [
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3_1-8b-instruct",
        "microsoft/DialoGPT-medium",
        "gpt-3.5-turbo",
    ]

    for name in non_embed_names:
        normalized = test_embeddings._normalize_model_name(name)
        assert normalized == name, f"Non-embed model name '{name}' was altered to '{normalized}'"
        print(f"✅ Non-embed name preserved: '{name}' -> '{normalized}'")

    # Test that embedding model names are still normalized
    embed_names_tests = [
        ("nvidia/llama-3_2-nemoretriever-1b-vlm-embed-v1", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"),
        ("nvidia/llama-3-2-nemoretriever-1b-vlm-embed-v1", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"),
        ("nvidia/llama3.2-nemoretriever-1b-vlm-embed-v1", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"),
        ("llama-3_2-nemoretriever-1b-vlm-embed-v1", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"),
        ("llama-3-2-nemoretriever-1b-vlm-embed-v1", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"),
        ("nvidia/nv-embed-v1", "nvidia/nv-embed-v1"),  # Should remain unchanged
        ("nvidia/llama3.2-embed", "nvidia/llama-3.2-embed"),  # Should be normalized (has nvidia prefix)
    ]

    for original, expected in embed_names_tests:
        normalized = test_embeddings._normalize_model_name(original)
        assert (
            normalized == expected
        ), f"Embed model name '{original}' normalized to '{normalized}', expected '{expected}'"
        print(f"✅ Embed name normalized correctly: '{original}' -> '{normalized}'")

    # Additional edge case tests to verify tight normalization
    print("\nTesting edge cases for tight normalization...")

    # Test that non-NVIDIA models with embedding indicators are NOT normalized
    edge_case_tests = [
        ("acme/llama32-vision-encoder", "acme/llama32-vision-encoder"),  # Should remain unchanged
        ("openai/llama-3_2-text-embed", "openai/llama-3_2-text-embed"),  # Should remain unchanged
        ("acme/embed-v1", "acme/embed-v1"),  # Should remain unchanged
        ("llama32-encoder", "llama32-encoder"),  # Should remain unchanged (no nvidia prefix)
        ("random/llama3.2-model", "random/llama3.2-model"),  # Should remain unchanged
    ]

    for original, expected in edge_case_tests:
        normalized = test_embeddings._normalize_model_name(original)
        assert normalized == expected, f"Edge case '{original}' was altered to '{normalized}', expected '{expected}'"
        print(f"✅ Edge case preserved: '{original}' -> '{normalized}'")

    # Initialize embeddings with probing
    embeddings = NVIDIAEmbeddings()

    # Sanity check: print the resolved model name
    print(f"\nResolved model name: {embeddings.model_name}")

    # Test connection
    if embeddings.test_connection():
        print("✅ NVIDIA API connection successful!")

        # Test embedding
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on algorithms.",
        ]

        print(f"\nTesting embedding of {len(test_texts)} documents...")
        doc_embeddings = embeddings.embed_documents(test_texts)
        print(f"✅ Document embeddings successful! Shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")

        print("\nTesting query embedding...")
        query_embedding = embeddings.embed_query("What is artificial intelligence?")
        print(f"✅ Query embedding successful! Dimension: {len(query_embedding)}")

    else:
        print("❌ NVIDIA API connection failed!")


if __name__ == "__main__":
    main()
