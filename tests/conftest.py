from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True, scope="session")
def load_dotenv_if_available():
    """Load environment variables from .env if python-dotenv is installed.

    Keeps tests flexible for local runs without committing secrets.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except Exception:
        return

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
        # Also export to os.environ for subprocesses
        # load_dotenv already injects into os.environ by default.
        # No extra work needed here.


# Pharmaceutical Benchmarking Test Fixtures


@pytest.fixture
def mock_enhanced_nemo_client():
    """Mock EnhancedNeMoClient for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.success = True
    mock_response.data = {
        "choices": [
            {"message": {"content": "This is a test response with interaction mechanism and clinical significance."}}
        ]
    }
    mock_response.endpoint_type = "cloud"
    mock_response.cost_tier = "infrastructure"
    mock_response.error = None
    mock_client.create_chat_completion.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_pharmaceutical_classifier():
    """Mock PharmaceuticalQueryClassifier for testing."""
    mock_classifier = Mock()
    mock_context = Mock()
    mock_context.domain = Mock()
    mock_context.domain.value = "DRUG_INTERACTIONS"
    mock_context.safety_urgency = "routine"
    mock_context.research_priority = "medium"
    mock_context.drug_names = ["aspirin", "warfarin"]
    mock_classifier.classify_query.return_value = mock_context
    return mock_classifier


@pytest.fixture
def mock_cost_analyzer():
    """Mock PharmaceuticalCostAnalyzer for testing."""
    mock_analyzer = Mock()
    mock_analyzer.record_pharmaceutical_query.return_value = None
    mock_analyzer.get_metrics.return_value = {"total_queries": 10, "total_cost": 125.0, "average_cost_per_query": 12.5}
    return mock_analyzer


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark JSON structure for testing."""
    return {
        "metadata": {
            "version": "1",
            "category": "drug_interactions",
            "created_date": "2025-09-30",
            "total_queries": 5,
            "description": "Test benchmark dataset",
            "baselines": {
                "cloud": {
                    "average_latency_ms": 450.0,
                    "success_rate": 0.98,
                    "average_cost_per_query": 12.5,
                    "average_accuracy": 0.85,
                    "notes": "Test cloud baseline",
                },
                "self_hosted": {
                    "average_latency_ms": 850.0,
                    "success_rate": 0.95,
                    "average_cost_per_query": 0.0,
                    "average_accuracy": 0.82,
                    "notes": "Test self-hosted baseline",
                },
                "regression_thresholds": {
                    "accuracy_drop_percent": 5,
                    "cost_increase_percent": 20,
                    "latency_increase_percent": 50,
                },
            },
        },
        "queries": [
            {
                "id": "test_001",
                "query": "Test query about drug interactions",
                "expected_type": "comparison",
                "expected_content": ["interaction mechanism", "clinical significance"],
                "evaluation_criteria": {"accuracy_weight": 0.4, "completeness_weight": 0.3, "relevance_weight": 0.3},
                "tags": ["test", "interaction"],
            }
        ],
    }


@pytest.fixture
def sample_baseline_metadata():
    """Sample baseline metadata structure for testing."""
    return {
        "cloud": {
            "average_latency_ms": 450.0,
            "success_rate": 0.98,
            "average_cost_per_query": 12.5,
            "average_accuracy": 0.85,
            "notes": "Based on NVIDIA Build cloud endpoints",
        },
        "self_hosted": {
            "average_latency_ms": 850.0,
            "success_rate": 0.95,
            "average_cost_per_query": 0.0,
            "average_accuracy": 0.82,
            "notes": "Based on local NIM containers",
        },
        "regression_thresholds": {
            "accuracy_drop_percent": 5,
            "cost_increase_percent": 20,
            "latency_increase_percent": 50,
        },
    }


@pytest.fixture
def sample_client_response_openai_format():
    """Sample OpenAI-style response format."""
    return {"choices": [{"message": {"content": "This is a test response with drug interaction details."}}]}


@pytest.fixture
def sample_client_response_custom_format():
    """Sample custom response format."""
    return {"text": "This is a test response in custom format.", "metadata": {"model": "test-model", "tokens": 50}}


@pytest.fixture
def benchmarks_directory():
    """Path to benchmarks directory."""
    return Path(__file__).parent.parent / "benchmarks"


@pytest.fixture
def test_fixtures_directory():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"
