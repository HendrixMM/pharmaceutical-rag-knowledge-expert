"""
Configuration validator for NeMo/NIM-native pipeline (Phase 0-1).

Provides fast, actionable validation of API key presence, model availability,
endpoint format/connectivity, and production environment flags.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from dataclasses import field

import requests

try:
    # Import model registries and helpers
    from src.nemo_retriever_client import NeMoRetrieverClient
except Exception:  # pragma: no cover - defensive import fallback
    NeMoRetrieverClient = None  # type: ignore


@dataclass
class ConfigValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _combine(results: list[ConfigValidationResult]) -> ConfigValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    for r in results:
        errors.extend(r.errors)
        warnings.extend(r.warnings)
    return ConfigValidationResult(valid=(len(errors) == 0), errors=errors, warnings=warnings)


def validate_nvidia_api_key(api_key: str | None) -> ConfigValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if not api_key:
        errors.append("NVIDIA_API_KEY is missing. Set it in your environment or .env file.")
        return ConfigValidationResult(False, errors, warnings)

    # Check for placeholder values
    placeholder_patterns = [
        r"your[_\-].*key",
        r"your[_\-].*api",
        r"example",
        r"placeholder",
        r"replace[_\-]?this",
        r"changeme",
        r"xxx+",
    ]
    api_key_lower = api_key.lower()
    for pattern in placeholder_patterns:
        if re.search(pattern, api_key_lower):
            errors.append(
                f"NVIDIA_API_KEY appears to be a placeholder value: '{api_key}'. "
                f"Please replace it with a real API key from https://build.nvidia.com"
            )
            return ConfigValidationResult(False, errors, warnings)

    if len(api_key.strip()) < 20:
        errors.append("NVIDIA_API_KEY appears too short (must be at least 20 characters).")
    if not re.match(r"^[A-Za-z0-9-_.]+$", api_key):
        warnings.append("NVIDIA_API_KEY contains unusual characters; ensure the key is correct.")
    return ConfigValidationResult(len(errors) == 0, errors, warnings)


def _validate_model_common(service: str, model: str | None) -> ConfigValidationResult:
    if not model:
        return ConfigValidationResult(True, [], [])
    errors: list[str] = []
    ok = True
    if NeMoRetrieverClient is not None and model:
        try:
            reg = getattr(
                NeMoRetrieverClient,
                "EMBEDDING_MODELS" if service == "embedding" else "RERANKING_MODELS",
                {},
            )
            key = model.split("/")[-1]
            ok = key in reg or any((isinstance(info, dict) and info.get("full_name") == model) for info in reg.values())
        except Exception:
            ok = True
    if not ok:
        suggestions = []
        if NeMoRetrieverClient is not None:
            reg = getattr(NeMoRetrieverClient, "EMBEDDING_MODELS" if service == "embedding" else "RERANKING_MODELS", {})
            suggestions = sorted(reg.keys())
        errors.append(
            f"Unsupported {service} model: '{model}'. Valid options include: {', '.join(suggestions) if suggestions else 'see docs'}"
        )
    return ConfigValidationResult(len(errors) == 0, errors, [])


def validate_embedding_model(model: str | None) -> ConfigValidationResult:
    return _validate_model_common("embedding", model)


def validate_reranking_model(model: str | None) -> ConfigValidationResult:
    return _validate_model_common("reranking", model)


def validate_endpoint_url(url: str | None) -> ConfigValidationResult:
    if not url:
        return ConfigValidationResult(True, [], [])

    errors: list[str] = []
    warnings: list[str] = []
    app_env = os.getenv("APP_ENV", "").strip().lower()

    if not re.match(r"^https?://", url):
        errors.append(f"Endpoint '{url}' must start with http:// or https://")
        return ConfigValidationResult(False, errors, warnings)
    if app_env == "production" and not url.startswith("https://"):
        errors.append(f"Endpoint '{url}' must use HTTPS in production.")

    if "ai.api.nvidia.com" in url and "/retrieval/" not in url:
        warnings.append("Endpoint host looks like NVIDIA API but path doesn't include /retrieval/.")

    # Connectivity test (non-fatal)
    try:
        resp = requests.head(url, timeout=3, allow_redirects=True)
        if resp.status_code == 405:
            resp = requests.get(url, timeout=3, allow_redirects=True)
        if resp.status_code >= 500:
            warnings.append(f"Endpoint '{url}' responded with {resp.status_code}; service may be unavailable.")
    except requests.RequestException as e:
        warnings.append(f"Endpoint '{url}' not reachable: {e}")

    return ConfigValidationResult(len(errors) == 0, errors, warnings)


def validate_production_environment() -> ConfigValidationResult:
    warnings: list[str] = []
    expected = {
        "ENABLE_NEMO_EXTRACTION": "true",
        "NEMO_EXTRACTION_STRATEGY": "nemo",
        "NEMO_EXTRACTION_STRICT": "true",
        "APP_ENV": "production",
    }
    for k, v in expected.items():
        actual = os.getenv(k, "").strip().lower()
        if actual != v:
            warnings.append(f"{k} should be '{v}' for production; current='{actual or '(unset)'}'.")
    return ConfigValidationResult(True, [], warnings)


def validate_complete_configuration() -> ConfigValidationResult:
    api_key = os.getenv("NVIDIA_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    rerank_model = os.getenv("RERANK_MODEL")
    custom_embedding_endpoint = os.getenv("NEMO_EMBEDDING_ENDPOINT")
    custom_reranking_endpoint = os.getenv("NEMO_RERANKING_ENDPOINT")
    custom_extraction_endpoint = os.getenv("NEMO_EXTRACTION_ENDPOINT")

    results: list[ConfigValidationResult] = []
    results.append(validate_nvidia_api_key(api_key))
    results.append(validate_embedding_model(embedding_model))
    results.append(validate_reranking_model(rerank_model))
    results.append(validate_endpoint_url(custom_embedding_endpoint))
    results.append(validate_endpoint_url(custom_reranking_endpoint))
    results.append(validate_endpoint_url(custom_extraction_endpoint))
    results.append(validate_production_environment())

    return _combine(results)


__all__ = [
    "ConfigValidationResult",
    "validate_nvidia_api_key",
    "validate_embedding_model",
    "validate_reranking_model",
    "validate_endpoint_url",
    "validate_production_environment",
    "validate_complete_configuration",
]
