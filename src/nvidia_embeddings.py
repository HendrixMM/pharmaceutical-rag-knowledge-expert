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

import os
import json
import logging
import re
import requests
import time
from typing import List, Optional, Dict, Any
from langchain.embeddings.base import Embeddings

# Set up logging
logger = logging.getLogger(__name__)

class NVIDIAEmbeddings(Embeddings):
    """NVIDIA LLaMA 3.2 NemoRetriever embedding model integration"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 10,
        probe_on_init: Optional[bool] = None,
        fallback_model_name: Optional[str] = None
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
        provided_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME") or "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
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

        # Track fallback status to avoid infinite loops
        self._already_fallback = False

        # Cache for embedding dimension to avoid duplicate network calls
        self._dimension: Optional[int] = None

        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Traycer-RAG/1.0 (+https://traycer.ai)",
        }

        # Probe and set model with fallback (if enabled)
        if self.probe_on_init:
            self._probe_and_set_model()
            logger.info(f"Initialized NVIDIA embeddings with final model: {self.model_name}")
        else:
            logger.info(f"Initialized NVIDIA embeddings with model (no probe): {self.model_name}")
            logger.info("Model probing disabled - first embedding request will validate model availability")

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name by converting recognized model-name patterns only

        Args:
            model_name: Raw model name that might contain variants

        Returns:
            Normalized model name
        """
        # Only apply normalization to recognized LLaMA model name patterns
        # Pattern: "llama-X_Y" where X and Y are digits
        if re.match(r'.*llama-\d+_\d+.*', model_name, re.IGNORECASE):
            # Convert underscore to dot in version numbers for LLaMA models
            # e.g., "llama-3_2" -> "llama-3.2"
            normalized = re.sub(r'(llama-\d+)_(\d+)', r'\1.\2', model_name, flags=re.IGNORECASE)
            return normalized

        # Handle compact variants such as "llama3.2" or "llama32"
        compact_pattern = re.compile(r'(llama)[\-_]?(\d)(?:[\.\-_]?(\d))?', re.IGNORECASE)
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
                payload = {
                    "input": ["probe"],
                    "model": self.model_name,
                    "encoding_format": "float"
                }

                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    logger.info(f"Successfully using preferred model: {preferred}")
                    self.model_selection_reason = "preferred"
                    return
                elif response.status_code in (401, 403):  # Authentication/Authorization errors
                    logger.error(f"Authentication error during probe (status {response.status_code}): {response.text}")
                    raise Exception(f"Authentication failed with status {response.status_code}. Check your NVIDIA API key.")
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
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

                    logger.warning(
                        f"Model probe received {response.status_code} with non-fallback error."
                    )
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
        logger.warning(f"Model probe failed after {self.max_retries} attempts with transient errors. Keeping model: {preferred}")
        try:
            url = f"{self.base_url}/embeddings"
            payload = {
                "input": ["probe"],
                "model": fallback,
                "encoding_format": "float"
            }
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
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

    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """
        Make request to NVIDIA API for embeddings

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        url = f"{self.base_url}/embeddings"

        payload = {
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float"
        }
        last_fallback_error: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making embedding request with model '{self.model_name}' (attempt {attempt + 1}/{self.max_retries})")

                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    if not isinstance(result, dict) or "data" not in result:
                        raise Exception(f"Unexpected embeddings response format: {result}")
                    try:
                        embeddings = [item["embedding"] for item in result["data"]]
                    except Exception as parsing_error:
                        logger.error("Malformed embedding payload: %s", parsing_error)
                        raise

                    # Cache embedding dimension on first successful request
                    if self._dimension is None and embeddings:
                        self._dimension = len(embeddings[0])
                        logger.debug(f"Cached embedding dimension: {self._dimension}")

                    logger.debug(f"Successfully got embeddings for {len(texts)} texts using model '{self.model_name}'")
                    return embeddings

                elif response.status_code in (401, 403):  # Authentication/Authorization errors
                    logger.error(f"Authentication error (status {response.status_code}): {response.text}")
                    raise Exception(f"Authentication failed with status {response.status_code}. Check your NVIDIA API key.")
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                else:
                    handled = False
                    if response.status_code in (400, 404, 422, 501) and not self._already_fallback:
                        reason = self._extract_model_unavailable_reason(response)
                        if reason:
                            logger.warning(f"Fallback triggered for request: {reason}")
                            logger.warning(f"Falling back to {self._fallback_model}")
                            self.model_name = self._fallback_model
                            self._dimension = None
                            logger.debug("Embedding dimension cache cleared after request-triggered fallback")
                            self._already_fallback = True
                            self.model_selection_reason = reason

                            payload["model"] = self.model_name

                            logger.info(f"Retrying request with fallback model: {self.model_name}")

                            fallback_response = requests.post(
                                url,
                                headers=self.headers,
                                json=payload,
                                timeout=60
                            )

                            if fallback_response.status_code == 200:
                                result = fallback_response.json()
                                if not isinstance(result, dict) or "data" not in result:
                                    raise Exception(f"Unexpected embeddings response format: {result}")
                                try:
                                    embeddings = [item["embedding"] for item in result["data"]]
                                except Exception as parsing_error:
                                    logger.error("Malformed embedding payload (fallback): %s", parsing_error)
                                    raise

                                if self._dimension is None and embeddings:
                                    self._dimension = len(embeddings[0])
                                    logger.debug(f"Cached embedding dimension: {self._dimension}")

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

                        handled = True

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
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            batch_embeddings = self._make_request(batch)
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
        embeddings = self._make_request([text])
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
    
    # Initialize embeddings
    embeddings = NVIDIAEmbeddings()
    
    # Test connection
    if embeddings.test_connection():
        print("✅ NVIDIA API connection successful!")
        
        # Test embedding
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on algorithms."
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
