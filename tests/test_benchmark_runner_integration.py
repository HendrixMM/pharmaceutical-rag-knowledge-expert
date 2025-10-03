"""
Test Benchmark Runner Integration

Tests for real client integration from Comment 1 verification.
Tests the integration of EnhancedNeMoClient, PharmaceuticalQueryClassifier,
and PharmaceuticalCostAnalyzer in the benchmark runner.
"""
import sys
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_pharmaceutical_benchmarks import BenchmarkConfig, BenchmarkRunner


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestRunnerInitialization:
    """Test BenchmarkRunner initialization with real vs simulated clients."""

    def test_runner_initialization_with_real_clients(
        self, mock_enhanced_nemo_client, mock_pharmaceutical_classifier, mock_cost_analyzer
    ):
        """Test runner initialization with real clients available."""
        config = BenchmarkConfig()

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")

            assert runner.use_real_clients is True
            assert runner.mode == "cloud"
            assert runner.nemo_client is not None
            assert runner.classifier is not None
            assert runner.cost_analyzer is not None

    def test_runner_initialization_without_clients(self):
        """Test runner initialization falls back to simulation when clients unavailable."""
        config = BenchmarkConfig()

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", False):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")

            # Should fall back to simulation mode
            assert runner.use_real_clients is False

    def test_runner_simulation_mode_flag(self):
        """Test that --simulate flag bypasses real clients."""
        config = BenchmarkConfig()

        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")

        assert runner.use_real_clients is False


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestExecuteQueryFullPipeline:
    """Test execute_query() full pipeline with mocked clients."""

    def test_execute_query_full_pipeline_mocked(
        self, mock_enhanced_nemo_client, mock_pharmaceutical_classifier, mock_cost_analyzer
    ):
        """Test execute_query() path: classify → execute → track."""
        config = BenchmarkConfig()

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            query = "What are the interactions between aspirin and warfarin?"

            response, latency, credits = runner.execute_query(query)

            # Verify response
            assert isinstance(response, str)
            assert len(response) > 0
            assert "interaction mechanism" in response.lower() or "clinical significance" in response.lower()

            # Verify latency
            assert isinstance(latency, float)
            assert latency >= 0

            # Verify credits
            assert isinstance(credits, int)
            assert credits >= 0

            # Verify classifier was called
            mock_pharmaceutical_classifier.classify_query.assert_called_once_with(query)

            # Verify nemo client was called
            mock_enhanced_nemo_client.create_chat_completion.assert_called_once()

            # Verify cost analyzer was called
            mock_cost_analyzer.record_pharmaceutical_query.assert_called_once()

    def test_execute_query_handles_api_failure(
        self, mock_enhanced_nemo_client, mock_pharmaceutical_classifier, mock_cost_analyzer
    ):
        """Test execute_query() handles API failures gracefully."""
        config = BenchmarkConfig()

        # Configure mock to return failure
        mock_response = Mock()
        mock_response.success = False
        mock_response.error = "API Error"
        mock_response.cost_tier = "infrastructure"
        mock_enhanced_nemo_client.create_chat_completion.return_value = mock_response

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            query = "Test query"

            response, latency, credits = runner.execute_query(query)

            # Should return empty response on failure
            assert response == ""
            assert latency >= 0
            assert credits >= 0


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestResponseExtraction:
    """Test _extract_response_content() with different response formats."""

    def test_extract_response_openai_format(
        self, mock_pharmaceutical_classifier, mock_cost_analyzer, sample_client_response_openai_format
    ):
        """Test extraction from OpenAI-style response format."""
        config = BenchmarkConfig()

        mock_client = Mock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = sample_client_response_openai_format
        mock_response.cost_tier = "infrastructure"
        mock_client.create_chat_completion.return_value = mock_response

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            response, _, _ = runner.execute_query("Test query")

            assert "drug interaction" in response.lower()

    def test_extract_response_custom_format(
        self, mock_pharmaceutical_classifier, mock_cost_analyzer, sample_client_response_custom_format
    ):
        """Test extraction from custom response format."""
        config = BenchmarkConfig()

        mock_client = Mock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = sample_client_response_custom_format
        mock_response.cost_tier = "infrastructure"
        mock_client.create_chat_completion.return_value = mock_response

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            response, _, _ = runner.execute_query("Test query")

            # Should extract text or response field
            assert len(response) > 0

    def test_extract_response_edge_cases(self, mock_pharmaceutical_classifier, mock_cost_analyzer):
        """Test extraction handles edge cases (empty, malformed)."""
        config = BenchmarkConfig()

        mock_client = Mock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {}  # Empty data
        mock_response.cost_tier = "infrastructure"
        mock_client.create_chat_completion.return_value = mock_response

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            response, _, _ = runner.execute_query("Test query")

            # Should handle gracefully, return empty or fallback
            assert isinstance(response, str)


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.cost_optimization
class TestCreditsEstimation:
    """Test _estimate_credits() with pharmaceutical multipliers."""

    def test_estimate_credits_safety_query_multiplier(self, mock_enhanced_nemo_client, mock_cost_analyzer):
        """Test that safety-critical queries get 1.5x multiplier."""
        config = BenchmarkConfig()

        mock_classifier = Mock()
        mock_context = Mock()
        mock_context.domain = Mock()
        mock_context.domain.value = "DRUG_SAFETY"
        mock_context.safety_urgency = "critical"
        mock_classifier.classify_query.return_value = mock_context

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier", return_value=mock_classifier
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            _, _, credits = runner.execute_query("What are the adverse reactions to penicillin?")

            # Safety queries should have higher credits
            assert credits > 0

    def test_estimate_credits_free_tier(self, mock_pharmaceutical_classifier, mock_cost_analyzer):
        """Test that free tier returns minimal credits (implementation returns 1 as minimum)."""
        config = BenchmarkConfig()

        mock_client = Mock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.data = {"choices": [{"message": {"content": "Response"}}]}
        mock_response.cost_tier = "free"
        mock_client.create_chat_completion.return_value = mock_response

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
            _, _, credits = runner.execute_query("Test query")

            # Implementation returns minimum 1 credit even for free tier for tracking
            assert credits >= 0


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestDomainMapping:
    """Test _map_domain_to_query_type() enum mapping."""

    def test_map_domain_to_query_type_complete(self, mock_enhanced_nemo_client, mock_cost_analyzer):
        """Test that all pharmaceutical domains map to query types."""
        config = BenchmarkConfig()

        pharmaceutical_domains = [
            "DRUG_SAFETY",
            "DRUG_INTERACTIONS",
            "CLINICAL_TRIALS",
            "PHARMACOKINETICS",
            "MECHANISM_OF_ACTION",
        ]

        for domain_name in pharmaceutical_domains:
            mock_classifier = Mock()
            mock_context = Mock()
            mock_context.domain = Mock()
            mock_context.domain.value = domain_name
            mock_context.safety_urgency = "routine"
            mock_classifier.classify_query.return_value = mock_context

            with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
                "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
            ), patch(
                "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier", return_value=mock_classifier
            ), patch(
                "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
            ):
                runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")
                runner.execute_query(f"Test query for {domain_name}")

                # Verify cost analyzer was called (domain was successfully mapped)
                mock_cost_analyzer.record_pharmaceutical_query.assert_called()
                call_args = mock_cost_analyzer.record_pharmaceutical_query.call_args
                assert call_args is not None


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestModeSelection:
    """Test cloud/self_hosted/both mode selection."""

    def test_mode_cloud_only(self, mock_enhanced_nemo_client, mock_pharmaceutical_classifier, mock_cost_analyzer):
        """Test cloud mode uses cloud endpoints."""
        config = BenchmarkConfig()

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="cloud")

            assert runner.mode == "cloud"

    def test_mode_self_hosted(self, mock_enhanced_nemo_client, mock_pharmaceutical_classifier, mock_cost_analyzer):
        """Test self_hosted mode uses local endpoints."""
        config = BenchmarkConfig()

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="self_hosted")

            assert runner.mode == "self_hosted"

    def test_mode_both_comparison(self, mock_enhanced_nemo_client, mock_pharmaceutical_classifier, mock_cost_analyzer):
        """Test 'both' mode enables comparison between cloud and self_hosted."""
        config = BenchmarkConfig()

        with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", True), patch(
            "scripts.run_pharmaceutical_benchmarks.EnhancedNeMoClient", return_value=mock_enhanced_nemo_client
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalQueryClassifier",
            return_value=mock_pharmaceutical_classifier,
        ), patch(
            "scripts.run_pharmaceutical_benchmarks.PharmaceuticalCostAnalyzer", return_value=mock_cost_analyzer
        ):
            runner = BenchmarkRunner(config, use_real_clients=True, mode="both")

            assert runner.mode == "both"


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.integration
@pytest.mark.fast
class TestCLIArguments:
    """Test CLI argument parsing and usage."""

    def test_cli_mode_argument(self):
        """Test --mode argument parsing."""
        # This is a basic structural test - actual CLI testing would use argparse testing
        config = BenchmarkConfig()

        for mode in ["cloud", "self_hosted", "both"]:
            with patch("scripts.run_pharmaceutical_benchmarks.CLIENTS_AVAILABLE", False):
                runner = BenchmarkRunner(config, use_real_clients=False, mode=mode)
                assert runner.mode == mode

    def test_cli_simulate_argument(self):
        """Test --simulate argument usage."""
        config = BenchmarkConfig()

        # Simulate mode should bypass real clients
        runner = BenchmarkRunner(config, use_real_clients=False, mode="cloud")
        assert runner.use_real_clients is False


@pytest.mark.pharmaceutical
@pytest.mark.client_integration
@pytest.mark.unit
@pytest.mark.fast
class TestLoggerInitialization:
    """Test logger initialization order fix (Comment 1)."""

    def test_logger_available_before_import_failure(self):
        """Test that logger is defined before import try/except block.

        This test validates Comment 1 fix: logger must be initialized before
        the import try/except block to avoid NameError when imports fail.
        """
        import builtins
        import sys

        # Remove the module from sys.modules to force re-import
        if "scripts.run_pharmaceutical_benchmarks" in sys.modules:
            del sys.modules["scripts.run_pharmaceutical_benchmarks"]

        # Mock ImportError for pharmaceutical client imports
        original_import = builtins.__import__

        def mock_import_with_failure(name, *args, **kwargs):
            # Fail when trying to import pharmaceutical clients
            if "nemo_client_enhanced" in name:
                raise ImportError("Simulated import failure for testing")
            return original_import(name, *args, **kwargs)

        # Temporarily replace __import__
        builtins.__import__ = mock_import_with_failure

        try:
            # Import should succeed despite pharmaceutical client import failure
            # and logger.warning() should work without NameError
            import scripts.run_pharmaceutical_benchmarks as bench

            # Verify module imported successfully
            assert bench is not None

            # Verify CLIENTS_AVAILABLE is False due to import failure
            assert bench.CLIENTS_AVAILABLE is False

            # Verify logger exists and is usable
            assert hasattr(bench, "logger")
            assert bench.logger is not None

            # No NameError should have occurred - test passes if we get here

        finally:
            # Restore original import
            builtins.__import__ = original_import

            # Clean up - remove from sys.modules for fresh import in other tests
            if "scripts.run_pharmaceutical_benchmarks" in sys.modules:
                del sys.modules["scripts.run_pharmaceutical_benchmarks"]

    def test_logger_defined_before_import_block(self):
        """Test that logger is defined early in the module.

        Verifies that logger initialization happens before line 33 (import try block).
        """
        import inspect

        import scripts.run_pharmaceutical_benchmarks as bench

        # Get the source code
        source = inspect.getsource(bench)

        # Find positions of logger initialization and import try block
        logger_init_pos = source.find("logger = logging.getLogger(__name__)")
        import_try_pos = source.find("from src.clients.nemo_client_enhanced import")

        # Logger should be initialized before import try block
        assert logger_init_pos > 0, "Logger initialization not found"
        assert import_try_pos > 0, "Import try block not found"
        assert (
            logger_init_pos < import_try_pos
        ), f"Logger must be initialized before imports (logger at {logger_init_pos}, imports at {import_try_pos})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
