import json
from unittest.mock import Mock

from src.nvidia_embeddings import NVIDIAEmbeddings


def _embedding_instance():
    # Bypass __init__ to avoid network calls or API key requirements
    return NVIDIAEmbeddings.__new__(NVIDIAEmbeddings)


def test_extract_reason_detects_disabled_message():
    embeddings = _embedding_instance()
    response = Mock()
    response.json.return_value = {"error": {"message": "This model has been disabled temporarily."}}

    reason = embeddings._extract_model_unavailable_reason(response)
    assert reason == "This model has been disabled temporarily."


def test_extract_reason_detects_access_denied_phrase():
    embeddings = _embedding_instance()
    response = Mock()
    response.json.return_value = {"error": {"message": "Access denied for model due to entitlement."}}

    reason = embeddings._extract_model_unavailable_reason(response)
    assert "access denied" in reason.lower()


def test_extract_reason_uses_error_code_when_present():
    embeddings = _embedding_instance()
    response = Mock()
    response.json.return_value = {"error": {"code": "MODEL_DISABLED", "message": ""}}

    reason = embeddings._extract_model_unavailable_reason(response)
    assert reason == "MODEL_DISABLED"


def test_extract_reason_returns_none_without_indicators():
    embeddings = _embedding_instance()
    response = Mock()
    response.json.return_value = {"error": {"message": "Rate limit exceeded.", "code": "rate_limit"}}

    reason = embeddings._extract_model_unavailable_reason(response)
    assert reason is None


def test_extract_reason_handles_invalid_json():
    embeddings = _embedding_instance()
    response = Mock()
    response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

    reason = embeddings._extract_model_unavailable_reason(response)
    assert reason is None
