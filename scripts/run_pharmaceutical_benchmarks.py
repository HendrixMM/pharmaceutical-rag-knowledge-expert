#!/usr/bin/env python3
"""
Pharmaceutical Benchmark Runner

Executes pharmaceutical benchmarks and tracks performance metrics.
Integrates with PharmaceuticalCostAnalyzer and EnhancedNeMoClient.

Usage:
    python scripts/run_pharmaceutical_benchmarks.py
    python scripts/run_pharmaceutical_benchmarks.py --category drug_interactions
    python scripts/run_pharmaceutical_benchmarks.py --version 1
    python scripts/run_pharmaceutical_benchmarks.py --save-results
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Configure logging BEFORE imports that might fail
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional runtime control of log level for regression visibility
_env_log_level = os.environ.get("BENCHMARK_LOG_LEVEL")
if _env_log_level:
    try:
        logging.getLogger().setLevel(getattr(logging, _env_log_level.upper(), logging.INFO))
    except Exception:
        pass

# Import pharmaceutical components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.clients.nemo_client_enhanced import ClientResponse, EndpointType, EnhancedNeMoClient
    from src.enhanced_config import EnhancedRAGConfig
    from src.monitoring.pharmaceutical_benchmark_tracker import PharmaceuticalBenchmarkTracker
    from src.monitoring.pharmaceutical_cost_analyzer import PharmaceuticalCostAnalyzer, PharmaceuticalQueryType
    from src.pharmaceutical.query_classifier import (
        PharmaceuticalContext,
        PharmaceuticalDomain,
        PharmaceuticalQueryClassifier,
    )

    CLIENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pharmaceutical clients not available: {e}")
    EnhancedNeMoClient = None
    PharmaceuticalQueryClassifier = None
    PharmaceuticalCostAnalyzer = None
    PharmaceuticalBenchmarkTracker = None
    EnhancedRAGConfig = None
    ClientResponse = None
    EndpointType = None
    PharmaceuticalContext = None
    PharmaceuticalDomain = None
    PharmaceuticalQueryType = None
    CLIENTS_AVAILABLE = False


class BenchmarkConfig:
    """Configuration loader for benchmarks."""

    def __init__(self, config_path: str = "config/benchmarks.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self.get_default_config()

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "datasets": {"base_path": "benchmarks", "categories": {}},
            "evaluation": {
                "default_weights": {"accuracy_weight": 0.4, "completeness_weight": 0.3, "relevance_weight": 0.3}
            },
            "execution": {"batch_size": 10, "timeout_seconds": 30},
            "storage": {"save_results": True, "results_directory": "results/benchmark_runs"},
        }


class BenchmarkLoader:
    """Loads benchmark datasets."""

    def __init__(self, benchmarks_dir: str = "benchmarks"):
        self.benchmarks_dir = Path(benchmarks_dir)

    def load_benchmark(self, category: str, version: int = 1) -> Dict[str, Any]:
        """Load a specific benchmark dataset."""
        filename = f"{category}_v{version}.json"
        filepath = self.benchmarks_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark file not found: {filepath}")

        with open(filepath) as f:
            benchmark = json.load(f)

        logger.info(f"Loaded {category} v{version}: {len(benchmark.get('queries', []))} queries")
        return benchmark

    def list_available_benchmarks(self) -> List[Tuple[str, int]]:
        """List all available benchmark files."""
        benchmarks = []
        for file in self.benchmarks_dir.glob("*_v*.json"):
            # Parse filename: category_vN.json
            parts = file.stem.split("_v")
            if len(parts) == 2:
                category = parts[0]
                version = int(parts[1])
                benchmarks.append((category, version))

        return sorted(benchmarks)


class QueryEvaluator:
    """Evaluates query responses against expected content."""

    @staticmethod
    def calculate_score(response: str, expected_content: List[str], weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate accuracy, completeness, and relevance scores."""
        if not response:
            return {"accuracy": 0.0, "completeness": 0.0, "relevance": 0.0, "overall": 0.0}

        # Simple keyword-based scoring (production would use semantic similarity)
        response_lower = response.lower()
        keywords_found = sum(1 for keyword in expected_content if keyword.lower() in response_lower)
        total_keywords = len(expected_content)

        # Basic scoring logic
        accuracy = keywords_found / total_keywords if total_keywords > 0 else 0.0
        completeness = min(1.0, keywords_found / max(1, total_keywords * 0.7))
        relevance = min(1.0, len(response) / 500)  # Penalize too short responses

        # Calculate weighted overall score
        overall = (
            accuracy * weights.get("accuracy_weight", 0.4)
            + completeness * weights.get("completeness_weight", 0.3)
            + relevance * weights.get("relevance_weight", 0.3)
        )

        return {
            "accuracy": round(accuracy, 3),
            "completeness": round(completeness, 3),
            "relevance": round(relevance, 3),
            "overall": round(overall, 3),
        }


class BenchmarkRunner:
    """Executes pharmaceutical benchmarks."""

    def __init__(self, config: BenchmarkConfig, use_real_clients: bool = True, mode: str = "cloud"):
        """
        Initialize benchmark runner with pharmaceutical clients.

        Args:
            config: Benchmark configuration
            use_real_clients: If True, use real NeMo/pharmaceutical clients; if False, use simulation
            mode: Execution mode - "cloud", "self_hosted", or "both"
        """
        self.config = config
        self.loader = BenchmarkLoader(config.config.get("datasets", {}).get("base_path", "benchmarks"))
        self.evaluator = QueryEvaluator()
        self.results = []
        self.use_real_clients = use_real_clients and CLIENTS_AVAILABLE
        self.mode = mode

        # Initialize pharmaceutical clients
        self.nemo_client = None
        self.classifier = None
        self.cost_analyzer = None
        self.benchmark_tracker = None

        # Always initialize classifier for validation purposes (even in simulation mode)
        try:
            if CLIENTS_AVAILABLE:
                from src.pharmaceutical.query_classifier import PharmaceuticalQueryClassifier

                self.classifier = PharmaceuticalQueryClassifier()
        except Exception as e:
            logger.warning(f"Could not initialize classifier: {e}")

        if self.use_real_clients:
            try:
                self._initialize_clients()
                logger.info(f"Initialized pharmaceutical clients in {mode} mode")
            except Exception as e:
                logger.error(f"Failed to initialize clients: {e}")
                logger.warning("Falling back to simulation mode")
                self.use_real_clients = False
        else:
            if not CLIENTS_AVAILABLE:
                logger.warning("Pharmaceutical clients not available - using simulation mode")
            else:
                logger.info("Using simulation mode (use_real_clients=False)")

    def _initialize_clients(self) -> None:
        """Initialize EnhancedNeMoClient, PharmaceuticalQueryClassifier, and PharmaceuticalCostAnalyzer."""
        # Initialize configuration from environment to honor cloud-first toggles
        try:
            rag_config = EnhancedRAGConfig.from_env()
        except Exception:
            # Fallback to defaults if env-based construction fails
            rag_config = EnhancedRAGConfig()

        # Initialize EnhancedNeMoClient
        self.nemo_client = EnhancedNeMoClient(config=rag_config)

        # Initialize PharmaceuticalQueryClassifier
        self.classifier = PharmaceuticalQueryClassifier()

        # Initialize PharmaceuticalCostAnalyzer
        self.cost_analyzer = PharmaceuticalCostAnalyzer(config=rag_config)

        # Optional: create a research project for budget tracking when configured
        self._project_id = None
        try:
            project_id = getattr(rag_config, "pharma_project_id", None) or os.getenv("PHARMA_PROJECT_ID")
            budget_limit = float(
                getattr(rag_config, "research_project_budget_limit_usd", 0.0)
                or os.getenv("PHARMA_BUDGET_LIMIT_USD")
                or 0.0
            )
            if project_id and budget_limit and budget_limit > 0:
                self.cost_analyzer.create_research_project(
                    project_id=project_id,
                    project_name=f"benchmark:{project_id}",
                    monthly_budget_usd=budget_limit,
                    priority_level=3,
                )
                self._project_id = project_id
        except Exception as e:
            logger.debug(f"Budget project setup skipped: {e}")

        # Initialize PharmaceuticalBenchmarkTracker
        try:
            self.benchmark_tracker = PharmaceuticalBenchmarkTracker(
                cost_analyzer=self.cost_analyzer, baseline_path=None  # Can be configured with benchmark metadata later
            )
            logger.info("Initialized PharmaceuticalBenchmarkTracker")
            # Route analyzer forwarding to the configured research project if available
            try:
                if self._project_id:
                    self.benchmark_tracker.project_id = self._project_id
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not initialize benchmark tracker: {e}")
            self.benchmark_tracker = None

        logger.info("Successfully initialized all pharmaceutical clients")

    def execute_query(self, query: str, timeout: int = 30) -> Tuple[str, float, int]:
        """
        Execute a query against the RAG system using real pharmaceutical clients.

        Returns: (response, latency_ms, credits_used)

        Integration:
        1. Use PharmaceuticalQueryClassifier to detect pharmaceutical queries
        2. Use EnhancedNeMoClient to execute the query
        3. Use PharmaceuticalCostAnalyzer to track credits
        """
        if not self.use_real_clients:
            # Simulation mode
            return self._execute_query_simulated(query)

        start_time = time.time()

        try:
            # 1. Classify query using PharmaceuticalQueryClassifier
            pharma_context = self.classifier.classify_query(query)
            logger.debug(f"Query classified as: {pharma_context.domain.value}")

            # 2. Execute using EnhancedNeMoClient.create_chat_completion()
            # Map mode to endpoint type for forced execution
            force_endpoint = None
            if self.mode == "cloud":
                force_endpoint = EndpointType.CLOUD
            elif self.mode == "self_hosted":
                force_endpoint = EndpointType.SELF_HOSTED
            # mode="both" is handled separately in execute_query_both()

            messages = [{"role": "user", "content": query}]
            response_obj = self.nemo_client.create_chat_completion(
                messages=messages, temperature=0.7, max_tokens=500, force_endpoint=force_endpoint
            )

            # 3. Extract response data
            if response_obj.success:
                # Extract content from response
                response_text = self._extract_response_content(response_obj)
                endpoint_type = response_obj.endpoint_type
                cost_tier = response_obj.cost_tier or "infrastructure"
            else:
                logger.error(f"Query execution failed: {response_obj.error}")
                response_text = ""
                endpoint_type = None
                cost_tier = "infrastructure"

            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to ms

            # 4. Estimate credits and track with PharmaceuticalCostAnalyzer
            # Prefer real token usage from response if available
            usage = None
            try:
                if hasattr(response_obj, "data") and isinstance(response_obj.data, dict):
                    usage = response_obj.data.get("usage")
            except Exception:
                usage = None

            estimated_tokens = 0
            if isinstance(usage, dict):
                # Prefer total_tokens; fall back to prompt+completion if present
                total_tokens = usage.get("total_tokens")
                if isinstance(total_tokens, (int, float)):
                    estimated_tokens = int(total_tokens)
                else:
                    pt = usage.get("prompt_tokens") or 0
                    ct = usage.get("completion_tokens") or 0
                    try:
                        estimated_tokens = int(pt) + int(ct)
                    except Exception:
                        estimated_tokens = 0

            # If usage not available, fall back to heuristic
            if estimated_tokens <= 0:
                estimated_tokens = len(query.split()) + len(response_text.split())

            credits = self._estimate_credits(
                query=query, response=response_text, pharma_context=pharma_context, cost_tier=cost_tier
            )

            # Track query with cost analyzer
            query_id = f"bench_{uuid.uuid4().hex}"

            # Build tags for audit and correlation
            tags = ["benchmark"]
            # Include run context if available
            run_id = None
            try:
                run_id = (getattr(self, "_current_run", {}) or {}).get("run_id")
                category = (getattr(self, "_current_run", {}) or {}).get("category")
            except Exception:
                run_id = None
                category = None
            if run_id:
                tags.append(f"run:{run_id}")
            if category:
                tags.append(category)
            if endpoint_type:
                try:
                    etag = endpoint_type.value if hasattr(endpoint_type, "value") else str(endpoint_type)
                except Exception:
                    etag = str(endpoint_type)
                tags.append(f"endpoint:{etag}")
            tags.append(f"mode:{self.mode}")

            # Build idempotency tag
            idem_raw = f"{run_id or ''}|{etag}|{query}"
            idem = hashlib.sha256(idem_raw.encode("utf-8")).hexdigest()[:16]
            tags.append(f"idem:{idem}")

            # Guard cost analyzer to avoid double logging in dual-endpoint mode
            should_record_cost = True
            try:
                if self.benchmark_tracker and getattr(self.benchmark_tracker, "cost_forwarding_enabled", False):
                    # Tracker will forward costs; skip direct analyzer call
                    should_record_cost = False
            except Exception:
                should_record_cost = True

            if should_record_cost:
                self.cost_analyzer.record_pharmaceutical_query(
                    query_id=query_id,
                    query_text=query,
                    cost_tier=cost_tier,
                    estimated_tokens=estimated_tokens,
                    project_id=(self._project_id or "benchmark_run"),
                    tags=tags,
                )

            logger.debug(f"Query executed: {len(response_text)} chars, {latency:.2f}ms, {credits} credits")

            return response_text, latency, credits

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            latency = (time.time() - start_time) * 1000
            # Return empty response on error
            return "", latency, 0

    def execute_query_with_endpoint(self, query: str, timeout: int, endpoint: EndpointType) -> Tuple[str, float, int]:
        """
        Execute a query against a specific endpoint (cloud or self-hosted).

        Args:
            query: Query string to execute
            timeout: Timeout in seconds
            endpoint: EndpointType.CLOUD or EndpointType.SELF_HOSTED

        Returns:
            (response, latency_ms, credits_used)
        """
        if not self.use_real_clients:
            # Simulation mode
            return self._execute_query_simulated(query)

        start_time = time.time()

        try:
            # 1. Classify query
            pharma_context = self.classifier.classify_query(query)
            logger.debug(f"Query classified as: {pharma_context.domain.value}")

            # 2. Execute with forced endpoint
            messages = [{"role": "user", "content": query}]
            response_obj = self.nemo_client.create_chat_completion(
                messages=messages, temperature=0.7, max_tokens=500, force_endpoint=endpoint
            )

            # 3. Extract response data
            if response_obj.success:
                response_text = self._extract_response_content(response_obj)
                cost_tier = response_obj.cost_tier or "infrastructure"
            else:
                logger.error(f"Query execution failed on {endpoint.value}: {response_obj.error}")
                response_text = ""
                cost_tier = "infrastructure"
                try:
                    # Detect unsupported self-hosted chat and set skip flag for subsequent queries
                    if (
                        endpoint == EndpointType.SELF_HOSTED
                        and isinstance(getattr(response_obj, "error", None), str)
                        and "NeMo chat is not supported" in response_obj.error
                        and getattr(self, "_skip_unsupported_self_hosted", False)
                    ):
                        self._skip_unsupported_self_hosted_detected = True
                except Exception:
                    pass

            # Calculate latency
            latency = (time.time() - start_time) * 1000

            # 4. Extract usage, compute estimated tokens
            usage = None
            try:
                if hasattr(response_obj, "data") and isinstance(response_obj.data, dict):
                    usage = response_obj.data.get("usage")
            except Exception:
                usage = None

            estimated_tokens = 0
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
                if isinstance(total_tokens, (int, float)):
                    estimated_tokens = int(total_tokens)
                else:
                    pt = usage.get("prompt_tokens") or 0
                    ct = usage.get("completion_tokens") or 0
                    try:
                        estimated_tokens = int(pt) + int(ct)
                    except Exception:
                        estimated_tokens = 0
            if estimated_tokens <= 0:
                estimated_tokens = len(query.split()) + len(response_text.split())

            # Cache last tokens per-endpoint for dual-mode reporting
            try:
                if endpoint == EndpointType.CLOUD:
                    self._last_estimated_tokens_cloud = estimated_tokens
                elif endpoint == EndpointType.SELF_HOSTED:
                    self._last_estimated_tokens_self_hosted = estimated_tokens
            except Exception:
                pass

            # 5. Estimate credits
            credits = self._estimate_credits(
                query=query, response=response_text, pharma_context=pharma_context, cost_tier=cost_tier
            )

            # Track query
            query_id = f"bench_{endpoint.value}_{int(time.time() * 1000)}"

            # Tags for endpoint-specific analyzer forwarding
            tags = ["benchmark", f"endpoint:{endpoint.value}", f"mode:{self.mode}"]
            try:
                run_id = (getattr(self, "_current_run", {}) or {}).get("run_id")
            except Exception:
                run_id = None
            if run_id:
                tags.append(f"run:{run_id}")
            idem_raw = f"{run_id or ''}|{endpoint.value}|{query}"
            idem = hashlib.sha256(idem_raw.encode("utf-8")).hexdigest()[:16]
            tags.append(f"idem:{idem}")

            # Guard cost analyzer to avoid double logging in dual-endpoint mode
            record_cost = True
            try:
                if self.benchmark_tracker and getattr(self.benchmark_tracker, "cost_forwarding_enabled", False):
                    # In dual-endpoint runs the tracker forwards costs; skip direct call
                    record_cost = False
            except Exception:
                record_cost = True

            if record_cost:
                self.cost_analyzer.record_pharmaceutical_query(
                    query_id=query_id,
                    query_text=query,
                    cost_tier=cost_tier,
                    estimated_tokens=estimated_tokens,
                    project_id=(self._project_id or f"benchmark_run_{endpoint.value}"),
                    tags=tags,
                )

            logger.debug(
                f"Query executed on {endpoint.value}: {len(response_text)} chars, {latency:.2f}ms, {credits} credits"
            )

            return response_text, latency, credits

        except Exception as e:
            logger.error(f"Error executing query on {endpoint.value}: {e}")
            latency = (time.time() - start_time) * 1000
            return "", latency, 0

    def _validate_classifier(
        self, pharma_context: PharmaceuticalContext, expected_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate classifier output against expected classification.

        Args:
            pharma_context: Actual classification from PharmaceuticalQueryClassifier
            expected_classification: Expected classification from benchmark data

        Returns:
            Dictionary containing:
                - domain_correct: bool
                - safety_urgency_correct: bool
                - research_priority_correct: bool
                - drug_names_correct: bool (optional, only if expected_drug_names provided)
                - overall_correct: bool
                - mismatches: list of mismatch descriptions
        """
        validation = {
            "domain_correct": False,
            "safety_urgency_correct": False,
            "research_priority_correct": False,
            "drug_names_correct": None,  # None if not validated
            "overall_correct": False,
            "mismatches": [],
        }

        # Validate domain
        expected_domain = expected_classification.get("domain", "").lower()
        actual_domain = pharma_context.domain.value.lower() if pharma_context.domain else ""

        if expected_domain and actual_domain:
            validation["domain_correct"] = expected_domain == actual_domain
            if not validation["domain_correct"]:
                validation["mismatches"].append(f"Domain mismatch: expected '{expected_domain}', got '{actual_domain}'")

        # Validate safety_urgency (use enum name, case-insensitive)
        expected_safety = expected_classification.get("safety_urgency", "").upper()
        actual_safety = pharma_context.safety_urgency.name.upper() if pharma_context.safety_urgency else ""

        if expected_safety and actual_safety:
            validation["safety_urgency_correct"] = expected_safety == actual_safety
            if not validation["safety_urgency_correct"]:
                validation["mismatches"].append(
                    f"Safety urgency mismatch: expected '{expected_safety}', got '{actual_safety}'"
                )

        # Validate research_priority (use enum name, case-insensitive)
        expected_priority = expected_classification.get("research_priority", "").upper()
        actual_priority = pharma_context.research_priority.name.upper() if pharma_context.research_priority else ""

        if expected_priority and actual_priority:
            validation["research_priority_correct"] = expected_priority == actual_priority
            if not validation["research_priority_correct"]:
                validation["mismatches"].append(
                    f"Research priority mismatch: expected '{expected_priority}', got '{actual_priority}'"
                )

        # Validate drug_names (optional, fuzzy matching with 50% overlap threshold)
        expected_drugs = expected_classification.get("drug_names", [])
        if expected_drugs:
            actual_drugs = pharma_context.drug_names or []

            # Normalize for case-insensitive comparison
            expected_drugs_normalized = {drug.lower() for drug in expected_drugs}
            actual_drugs_normalized = {drug.lower() for drug in actual_drugs}

            # Calculate overlap
            if expected_drugs_normalized:
                overlap = len(expected_drugs_normalized & actual_drugs_normalized)
                overlap_ratio = overlap / len(expected_drugs_normalized)

                validation["drug_names_correct"] = overlap_ratio >= 0.5
                if not validation["drug_names_correct"]:
                    validation["mismatches"].append(
                        f"Drug names mismatch: expected {list(expected_drugs_normalized)}, "
                        f"got {list(actual_drugs_normalized)} (overlap: {overlap_ratio:.1%})"
                    )

        # Calculate overall correctness
        # Domain, safety_urgency, and research_priority are required
        # drug_names is optional (only counts if provided)
        required_checks = [
            validation["domain_correct"],
            validation["safety_urgency_correct"],
            validation["research_priority_correct"],
        ]

        # Add drug_names check if it was validated
        if validation["drug_names_correct"] is not None:
            required_checks.append(validation["drug_names_correct"])

        validation["overall_correct"] = all(required_checks)

        return validation

    def execute_query_both(self, query: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute query against both cloud and self-hosted endpoints for comparison.

        Args:
            query: Query string to execute
            timeout: Timeout in seconds

        Returns:
            Dictionary with cloud results, self_hosted results, and comparison metrics
        """
        logger.info("Executing query on both cloud and self-hosted endpoints")

        # Guard: if EndpointType is unavailable (imports failed) or real clients disabled,
        # avoid accessing EndpointType and fall back to simulated responses.
        try:
            endpoint_type_missing = EndpointType is None
        except Exception:
            endpoint_type_missing = True

        if endpoint_type_missing or not self.use_real_clients:
            # Simulate both endpoints independently
            cloud_response, cloud_latency, cloud_credits = self._execute_query_simulated(query)
            sh_response, sh_latency, sh_credits = self._execute_query_simulated(query)

            # Heuristic token estimation based on text length (words proxy)
            cloud_tokens = len((query or "").split()) + len((cloud_response or "").split())
            sh_tokens = len((query or "").split()) + len((sh_response or "").split())

            return {
                "cloud": {
                    "response": cloud_response,
                    "latency_ms": cloud_latency,
                    "credits_used": cloud_credits,
                    "succeeded": bool(cloud_response),
                    "estimated_tokens": int(cloud_tokens),
                },
                "self_hosted": {
                    "response": sh_response,
                    "latency_ms": sh_latency,
                    "credits_used": sh_credits,
                    "succeeded": bool(sh_response),
                    "estimated_tokens": int(sh_tokens),
                },
                "comparison": {
                    "latency_diff_ms": sh_latency - cloud_latency,
                    "latency_ratio": sh_latency / cloud_latency if cloud_latency > 0 else float("inf"),
                    "cost_diff": cloud_credits - sh_credits,
                    "both_succeeded": bool(cloud_response and sh_response),
                    "cloud_faster": cloud_latency < sh_latency,
                    "self_hosted_cheaper": sh_credits < cloud_credits,
                },
            }

        # Execute with real clients for each endpoint (in parallel for ROI)
        results_map: Dict[str, Tuple[str, float, int]] = {}

        def _run(endpoint: EndpointType) -> Tuple[str, float, int, str]:
            resp, lat, cred = self.execute_query_with_endpoint(query, timeout, endpoint)
            return resp, lat, cred, endpoint.value

        # If self-hosted chat unsupported was detected and skipping is enabled, avoid scheduling it
        allow_sh = not getattr(self, "_skip_unsupported_self_hosted_detected", False)
        if not getattr(self, "_skip_unsupported_self_hosted", True):
            allow_sh = True

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_run, EndpointType.CLOUD), ex.submit(_run, EndpointType.SELF_HOSTED)]
            if not allow_sh:
                futures = [ex.submit(_run, EndpointType.CLOUD)]
            for fut in as_completed(futures):
                try:
                    resp, lat, cred, label = fut.result()
                    results_map[label] = (resp, lat, cred)
                except Exception as e:
                    logger.warning(f"Parallel execution error: {e}")
        # Fallback defaults if any endpoint missing
        cloud_response, cloud_latency, cloud_credits = results_map.get("cloud", ("", 0.0, 0))
        if allow_sh:
            sh_response, sh_latency, sh_credits = results_map.get("self_hosted", ("", 0.0, 0))
        else:
            sh_response, sh_latency, sh_credits = "", 0.0, 0

        # Estimate tokens heuristically to avoid cross-query races under concurrency
        cloud_tokens = len((query or "").split()) + len((cloud_response or "").split())
        sh_tokens = len((query or "").split()) + len((sh_response or "").split())

        # Build comparison
        return {
            "cloud": {
                "response": cloud_response,
                "latency_ms": cloud_latency,
                "credits_used": cloud_credits,
                "succeeded": bool(cloud_response),
                "estimated_tokens": int(cloud_tokens),
            },
            "self_hosted": {
                "response": sh_response,
                "latency_ms": sh_latency,
                "credits_used": sh_credits,
                "succeeded": bool(sh_response),
                "estimated_tokens": int(sh_tokens),
            },
            "comparison": {
                "latency_diff_ms": sh_latency - cloud_latency,
                "latency_ratio": sh_latency / cloud_latency if cloud_latency > 0 else float("inf"),
                "cost_diff": cloud_credits - sh_credits,
                "both_succeeded": bool(cloud_response and sh_response),
                "cloud_faster": cloud_latency < sh_latency,
                "self_hosted_cheaper": sh_credits < cloud_credits,
            },
        }

    def _execute_query_simulated(self, query: str) -> Tuple[str, float, int]:
        """Execute query in simulation mode (fallback when clients unavailable)."""
        start_time = time.time()

        # Simulate query execution
        response = f"Simulated response for: {query[:100]}..."
        credits = 10  # Simulated credits

        latency = (time.time() - start_time) * 1000  # Convert to ms

        return response, latency, credits

    def _extract_response_content(self, response_obj: Any) -> str:
        """Extract text content from EnhancedNeMoClient response."""
        try:
            # Handle different response formats
            if hasattr(response_obj, "data") and response_obj.data:
                data = response_obj.data
                # Try different content extraction methods
                if isinstance(data, dict):
                    # Check for common content fields
                    for key in ["content", "text", "message", "response"]:
                        if key in data:
                            content = data[key]
                            if isinstance(content, str):
                                return content
                            elif isinstance(content, dict) and "content" in content:
                                return str(content["content"])
                    # If data has choices (OpenAI format)
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"]
                # Fallback to string representation
                return str(data)
            return ""
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            return ""

    def _estimate_credits(self, query: str, response: str, pharma_context: Any, cost_tier: str) -> int:
        """Estimate credits used for a query."""
        # Rough token estimation (4 chars per token)
        total_tokens = (len(query) + len(response)) // 4

        if cost_tier == "free_tier":
            # Free tier doesn't consume actual credits, but count opportunity cost
            return 0

        # Estimate credits based on tokens
        # Typical pricing: ~0.02 credits per 1000 tokens
        credits = max(1, int(total_tokens * 0.02 / 1000))

        # Adjust for pharmaceutical query type (higher value queries might use more resources)
        if pharma_context and hasattr(pharma_context, "safety_urgency"):
            # Safety-critical queries might need more comprehensive responses
            from src.pharmaceutical.query_classifier import SafetyUrgency

            if pharma_context.safety_urgency in [SafetyUrgency.CRITICAL, SafetyUrgency.HIGH]:
                credits = int(credits * 1.5)

        return credits

    def _map_domain_to_query_type(self, domain: Any) -> Any:
        """Map PharmaceuticalDomain to PharmaceuticalQueryType."""
        # Map domain enum to query type enum
        domain_to_type_mapping = {
            "drug_safety": PharmaceuticalQueryType.DRUG_SAFETY,
            "drug_interactions": PharmaceuticalQueryType.DRUG_INTERACTIONS,
            "clinical_trials": PharmaceuticalQueryType.CLINICAL_TRIALS,
            "pharmacokinetics": PharmaceuticalQueryType.PHARMACOKINETICS,
            "mechanism_of_action": PharmaceuticalQueryType.MECHANISM_OF_ACTION,
            "dosage_guidelines": PharmaceuticalQueryType.DOSAGE_GUIDELINES,
            "adverse_reactions": PharmaceuticalQueryType.DRUG_SAFETY,  # Map to safety
            "general_research": PharmaceuticalQueryType.GENERAL_RESEARCH,
        }

        domain_str = domain.value if hasattr(domain, "value") else str(domain)
        return domain_to_type_mapping.get(domain_str, PharmaceuticalQueryType.GENERAL_RESEARCH)

    def run_benchmark(
        self,
        category: str,
        version: int = 1,
        batch_size: Optional[int] = None,
        concurrency: int = 1,
        skip_classifier_validation: bool = False,
        enforce_budget: bool = False,
        budget_stop_utilization: float = 0.95,
        max_queries: Optional[int] = None,
        adaptive_concurrency: bool = False,
        adapt_latency_high_ms: float = 6500.0,
        adapt_latency_low_ms: float = 2500.0,
        adapt_step: int = 1,
    ) -> Dict[str, Any]:
        """Run a complete benchmark for a category."""
        logger.info(f"Running benchmark: {category} v{version}")

        # Load benchmark
        benchmark = self.loader.load_benchmark(category, version)
        queries = benchmark.get("queries", [])
        # Optional sampling for faster high-ROI runs
        if isinstance(max_queries, int) and max_queries > 0:
            queries = list(queries)[:max_queries]
        metadata = benchmark.get("metadata", {})

        # Get configuration
        batch_size = batch_size or self.config.config.get("execution", {}).get("batch_size", 10)
        timeout = self.config.config.get("execution", {}).get("timeout_seconds", 30)
        default_weights = self.config.config.get("evaluation", {}).get("default_weights", {})

        # Generate a run_id for correlation and set tracker forwarding policy
        run_id = f"{category}_v{version}_{int(time.time()*1000)}"
        # Make run context available to other methods
        self._current_run = {"run_id": run_id, "category": category}
        if self.benchmark_tracker:
            try:
                self.benchmark_tracker.current_run_id = run_id
                # Forward costs only in dual-endpoint mode to avoid duplicate analyzer records
                self.benchmark_tracker.cost_forwarding_enabled = self.mode == "both"
            except Exception:
                pass

        # Execute queries
        run_start = time.time()
        query_results = []
        total_credits = 0
        total_latency = 0
        # For mode="both", track separate totals
        cloud_total_credits = 0
        cloud_total_latency = 0
        self_hosted_total_credits = 0
        self_hosted_total_latency = 0

        # Concurrency guard
        concurrency = max(1, int(concurrency or 1))

        # If concurrency>1, execute in adaptive windows to control rate/latency
        if concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            exec_results: Dict[int, Any] = {}
            allowed_max = max(1, int(concurrency))
            cur_conc = min(allowed_max, len(queries))
            i = 0
            # thresholds from env if provided
            try:
                adapt_latency_high_ms = float(os.getenv("BENCHMARK_ADAPT_LATENCY_HIGH_MS", adapt_latency_high_ms))
                adapt_latency_low_ms = float(os.getenv("BENCHMARK_ADAPT_LATENCY_LOW_MS", adapt_latency_low_ms))
                adapt_step = int(os.getenv("BENCHMARK_ADAPT_STEP", str(adapt_step)))
            except Exception:
                pass
            while i < len(queries):
                window_end = min(len(queries), i + cur_conc)
                window_idx = list(range(i, window_end))
                if not window_idx:
                    break
                with ThreadPoolExecutor(max_workers=cur_conc) as pool:
                    future_map = {}
                    for idx in window_idx:
                        query_text = queries[idx].get("query", "")
                        if self.mode == "both":
                            fut = pool.submit(self.execute_query_both, query_text, timeout)
                        else:
                            fut = pool.submit(self.execute_query, query_text, timeout)
                        future_map[fut] = idx
                        # Stagger launches slightly to reduce burst rate
                        try:
                            stagger_ms = int(os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50"))
                        except Exception:
                            stagger_ms = 50
                        if stagger_ms > 0:
                            time.sleep(stagger_ms / 1000.0)
                    # Collect results
                    latencies_ms = []
                    failures = 0
                    for fut in as_completed(future_map):
                        idx = future_map[fut]
                        try:
                            result = fut.result()
                            exec_results[idx] = result
                            # measure cloud latency primarily
                            if self.mode == "both" and isinstance(result, dict):
                                lat = float(result.get("cloud", {}).get("latency_ms", 0.0) or 0.0)
                            else:
                                # single-mode returns tuple (resp, latency, credits)
                                try:
                                    lat = float(result[1]) if isinstance(result, tuple) and len(result) >= 2 else 0.0
                                except Exception:
                                    lat = 0.0
                            latencies_ms.append(lat)
                        except Exception as e:
                            failures += 1
                            exec_results[idx] = None
                            logger.error(f"Error executing query {idx} in parallel window: {e}")
                # Simple adaptation heuristic
                if adaptive_concurrency:
                    try:
                        avg_lat = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
                        if failures > 0 or avg_lat > adapt_latency_high_ms:
                            cur_conc = max(1, cur_conc - max(1, adapt_step))
                            logger.info(
                                f"Adaptive concurrency: lowering to {cur_conc} (avg_lat={avg_lat:.2f}ms, failures={failures})"
                            )
                        elif avg_lat > 0 and avg_lat < adapt_latency_low_ms and cur_conc < allowed_max:
                            cur_conc = min(allowed_max, cur_conc + max(1, adapt_step))
                            logger.info(f"Adaptive concurrency: increasing to {cur_conc} (avg_lat={avg_lat:.2f}ms)")
                    except Exception:
                        pass
                i = window_end

        for i, query_data in enumerate(queries):
            query_id = query_data.get("id", f"query_{i}")
            query_text = query_data.get("query", "")
            expected_content = query_data.get("expected_content", [])
            eval_criteria = query_data.get("evaluation_criteria", {})
            expected_classification = query_data.get("expected_classification", None)

            # Merge weights (query-specific overrides default)
            weights = {**default_weights, **eval_criteria}

            # Optional budget enforcement: stop early if utilization threshold exceeded
            if enforce_budget and self.cost_analyzer and getattr(self, "_project_id", None):
                try:
                    proj = self.cost_analyzer.projects.get(self._project_id)
                    if proj:
                        status = proj.get_budget_status()
                        util = float(status.get("budget_utilization", 0.0) or 0.0)
                        if util >= float(budget_stop_utilization or 0.95):
                            logger.warning(
                                f"Budget utilization {util:.2f} >= threshold {budget_stop_utilization:.2f}; stopping further queries."
                            )
                            break
                except Exception as e:
                    logger.debug(f"Budget check skipped: {e}")

            logger.info(f"Executing query {i+1}/{len(queries)}: {query_id}")

            # Classify query and validate if expected_classification is provided
            pharma_context = None
            classifier_validation = None
            actual_classification = None

            try:
                pharma_context = self.classifier.classify_query(query_text)

                # Store actual classification
                actual_classification = {
                    "domain": pharma_context.domain.value if pharma_context.domain else None,
                    "safety_urgency": pharma_context.safety_urgency.name if pharma_context.safety_urgency else None,
                    "research_priority": pharma_context.research_priority.name
                    if pharma_context.research_priority
                    else None,
                    "drug_names": pharma_context.drug_names if pharma_context.drug_names else [],
                }

                # Validate if expected_classification exists and not skipped
                if (not skip_classifier_validation) and expected_classification and pharma_context:
                    classifier_validation = self._validate_classifier(pharma_context, expected_classification)
                    logger.debug(f"Classifier validation: {classifier_validation['overall_correct']}")
            except Exception as e:
                logger.error(f"Error classifying query {query_id}: {e}")

            try:
                if self.mode == "both":
                    # Execute against both endpoints (use pre-executed result when concurrency>1)
                    if concurrency > 1:
                        dual_result = exec_results.get(i)
                        if dual_result is None:
                            dual_result = {
                                "cloud": {
                                    "response": "",
                                    "latency_ms": 0.0,
                                    "credits_used": 0,
                                    "succeeded": False,
                                    "estimated_tokens": 0,
                                },
                                "self_hosted": {
                                    "response": "",
                                    "latency_ms": 0.0,
                                    "credits_used": 0,
                                    "succeeded": False,
                                    "estimated_tokens": 0,
                                },
                                "comparison": {
                                    "latency_diff_ms": 0.0,
                                    "latency_ratio": 0.0,
                                    "cost_diff": 0.0,
                                    "both_succeeded": False,
                                    "cloud_faster": False,
                                    "self_hosted_cheaper": False,
                                },
                            }
                    else:
                        dual_result = self.execute_query_both(query_text, timeout)

                    # Evaluate both responses
                    cloud_scores = self.evaluator.calculate_score(
                        dual_result["cloud"]["response"], expected_content, weights
                    )
                    sh_scores = self.evaluator.calculate_score(
                        dual_result["self_hosted"]["response"], expected_content, weights
                    )

                    # Store dual result
                    result = {
                        "query_id": query_id,
                        "query": query_text,
                        "expected_content": expected_content,
                        "mode": "both",
                        "cloud": {
                            "response": dual_result["cloud"]["response"],
                            "scores": cloud_scores,
                            "latency_ms": round(dual_result["cloud"]["latency_ms"], 2),
                            "credits_used": dual_result["cloud"]["credits_used"],
                            "succeeded": dual_result["cloud"]["succeeded"],
                            "estimated_tokens": int(dual_result["cloud"].get("estimated_tokens", 0) or 0),
                        },
                        "self_hosted": {
                            "response": dual_result["self_hosted"]["response"],
                            "scores": sh_scores,
                            "latency_ms": round(dual_result["self_hosted"]["latency_ms"], 2),
                            "credits_used": dual_result["self_hosted"]["credits_used"],
                            "succeeded": dual_result["self_hosted"]["succeeded"],
                            "estimated_tokens": int(dual_result["self_hosted"].get("estimated_tokens", 0) or 0),
                        },
                        "comparison": dual_result["comparison"],
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Add classifier validation if available
                    if actual_classification:
                        result["actual_classification"] = actual_classification
                    if classifier_validation:
                        result["classifier_validation"] = classifier_validation

                    query_results.append(result)

                    # Update separate totals - only accumulate for successful queries
                    if dual_result["cloud"]["succeeded"]:
                        cloud_total_credits += dual_result["cloud"]["credits_used"]
                        cloud_total_latency += dual_result["cloud"]["latency_ms"]
                    if dual_result["self_hosted"]["succeeded"]:
                        self_hosted_total_credits += dual_result["self_hosted"]["credits_used"]
                        self_hosted_total_latency += dual_result["self_hosted"]["latency_ms"]

                    # Track with benchmark tracker if available
                    if self.benchmark_tracker:
                        try:
                            # Track cloud endpoint (both success and failure)
                            if dual_result["cloud"]["succeeded"]:
                                self.benchmark_tracker.track_query_result(
                                    category=category,
                                    query_type=pharma_context.domain.value if pharma_context else "unknown",
                                    accuracy=cloud_scores["accuracy"],
                                    cost=dual_result["cloud"]["credits_used"],
                                    latency_ms=dual_result["cloud"]["latency_ms"],
                                    success=True,
                                    endpoint="cloud",
                                    run_id=run_id,
                                    estimated_tokens=int(dual_result["cloud"].get("estimated_tokens", 0) or 0),
                                )
                            else:
                                self.benchmark_tracker.track_query_result(
                                    category=category,
                                    query_type=pharma_context.domain.value if pharma_context else "unknown",
                                    accuracy=0.0,
                                    cost=0.0,
                                    latency_ms=0.0,
                                    success=False,
                                    endpoint="cloud",
                                    run_id=run_id,
                                    estimated_tokens=0,
                                )

                            # Track self-hosted endpoint (both success and failure)
                            if dual_result["self_hosted"]["succeeded"]:
                                self.benchmark_tracker.track_query_result(
                                    category=category,
                                    query_type=pharma_context.domain.value if pharma_context else "unknown",
                                    accuracy=sh_scores["accuracy"],
                                    cost=dual_result["self_hosted"]["credits_used"],
                                    latency_ms=dual_result["self_hosted"]["latency_ms"],
                                    success=True,
                                    endpoint="self_hosted",
                                    run_id=run_id,
                                    estimated_tokens=int(dual_result["self_hosted"].get("estimated_tokens", 0) or 0),
                                )
                            else:
                                self.benchmark_tracker.track_query_result(
                                    category=category,
                                    query_type=pharma_context.domain.value if pharma_context else "unknown",
                                    accuracy=0.0,
                                    cost=0.0,
                                    latency_ms=0.0,
                                    success=False,
                                    endpoint="self_hosted",
                                    run_id=run_id,
                                    estimated_tokens=0,
                                )
                        except Exception as e:
                            logger.warning(f"Error tracking query with benchmark tracker: {e}")

                else:
                    # Single-mode execution (cloud or self_hosted)
                    if concurrency > 1:
                        resp_tuple = exec_results.get(i)
                        if resp_tuple is None:
                            response, latency, credits = "", 0.0, 0
                        else:
                            response, latency, credits = resp_tuple
                    else:
                        response, latency, credits = self.execute_query(query_text, timeout)

                    # Evaluate response
                    scores = self.evaluator.calculate_score(response, expected_content, weights)

                    # Store result
                    result = {
                        "query_id": query_id,
                        "query": query_text,
                        "response": response,
                        "expected_content": expected_content,
                        "scores": scores,
                        "latency_ms": round(latency, 2),
                        "credits_used": credits,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Add classifier validation if available
                    if actual_classification:
                        result["actual_classification"] = actual_classification
                    if classifier_validation:
                        result["classifier_validation"] = classifier_validation

                    query_results.append(result)
                    total_credits += credits
                    total_latency += latency

                    # Track with benchmark tracker if available
                    if self.benchmark_tracker:
                        try:
                            self.benchmark_tracker.track_query_result(
                                category=category,
                                query_type=pharma_context.domain.value if pharma_context else "unknown",
                                accuracy=scores["accuracy"],
                                cost=credits,
                                latency_ms=latency,
                                success=bool(response),
                                endpoint=self.mode,
                                run_id=run_id,
                            )
                        except Exception as e:
                            logger.warning(f"Error tracking query with benchmark tracker: {e}")

            except Exception as e:
                logger.error(f"Error executing query {query_id}: {e}")
                query_results.append(
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Calculate aggregate metrics
        # Calculate classifier validation metrics (shared across both modes)
        queries_with_validation = [r for r in query_results if r.get("classifier_validation")]
        classifier_metrics = None
        if queries_with_validation:
            total_validated = len(queries_with_validation)
            domain_correct = sum(1 for r in queries_with_validation if r["classifier_validation"]["domain_correct"])
            safety_correct = sum(
                1 for r in queries_with_validation if r["classifier_validation"]["safety_urgency_correct"]
            )
            priority_correct = sum(
                1 for r in queries_with_validation if r["classifier_validation"]["research_priority_correct"]
            )
            overall_correct = sum(1 for r in queries_with_validation if r["classifier_validation"]["overall_correct"])

            # Drug names validation (only count queries where it was validated)
            drug_validations = [
                r for r in queries_with_validation if r["classifier_validation"]["drug_names_correct"] is not None
            ]
            drug_correct = (
                sum(1 for r in drug_validations if r["classifier_validation"]["drug_names_correct"])
                if drug_validations
                else 0
            )
            drug_accuracy = drug_correct / len(drug_validations) if drug_validations else None

            classifier_metrics = {
                "total_validated_queries": total_validated,
                "domain_accuracy": round(domain_correct / total_validated, 3),
                "safety_urgency_accuracy": round(safety_correct / total_validated, 3),
                "research_priority_accuracy": round(priority_correct / total_validated, 3),
                "overall_accuracy": round(overall_correct / total_validated, 3),
                "drug_names_accuracy": round(drug_accuracy, 3) if drug_accuracy is not None else None,
            }

        if self.mode == "both":
            # Calculate separate metrics for cloud and self_hosted
            cloud_successful = [
                r for r in query_results if r.get("mode") == "both" and r.get("cloud", {}).get("succeeded")
            ]
            sh_successful = [
                r for r in query_results if r.get("mode") == "both" and r.get("self_hosted", {}).get("succeeded")
            ]

            # Cloud metrics
            cloud_avg_accuracy = (
                sum(r["cloud"]["scores"]["accuracy"] for r in cloud_successful) / len(cloud_successful)
                if cloud_successful
                else 0
            )
            cloud_avg_overall = (
                sum(r["cloud"]["scores"]["overall"] for r in cloud_successful) / len(cloud_successful)
                if cloud_successful
                else 0
            )
            cloud_avg_latency = cloud_total_latency / len(cloud_successful) if cloud_successful else 0
            cloud_avg_credits = cloud_total_credits / len(cloud_successful) if cloud_successful else 0
            cloud_avg_tokens = (
                sum(int(r["cloud"].get("estimated_tokens", 0) or 0) for r in cloud_successful) / len(cloud_successful)
                if cloud_successful
                else 0
            )

            # Percentiles
            def _pct(values: List[float], p: float) -> float:
                if not values:
                    return 0.0
                vals = sorted(values)
                k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
                return float(vals[k])

            cloud_latencies = [
                float(r["cloud"]["latency_ms"])
                for r in cloud_successful
                if isinstance(r.get("cloud", {}).get("latency_ms"), (int, float))
            ]
            cloud_tokens = [float(int(r["cloud"].get("estimated_tokens", 0) or 0)) for r in cloud_successful]
            cloud_p90_latency = _pct(cloud_latencies, 90.0)
            cloud_p95_latency = _pct(cloud_latencies, 95.0)
            cloud_p90_tokens = _pct(cloud_tokens, 90.0)
            cloud_p95_tokens = _pct(cloud_tokens, 95.0)

            # Self-hosted metrics
            sh_avg_accuracy = (
                sum(r["self_hosted"]["scores"]["accuracy"] for r in sh_successful) / len(sh_successful)
                if sh_successful
                else 0
            )
            sh_avg_overall = (
                sum(r["self_hosted"]["scores"]["overall"] for r in sh_successful) / len(sh_successful)
                if sh_successful
                else 0
            )
            sh_avg_latency = self_hosted_total_latency / len(sh_successful) if sh_successful else 0
            sh_avg_credits = self_hosted_total_credits / len(sh_successful) if sh_successful else 0
            sh_avg_tokens = (
                sum(int(r["self_hosted"].get("estimated_tokens", 0) or 0) for r in sh_successful) / len(sh_successful)
                if sh_successful
                else 0
            )
            sh_latencies = [
                float(r["self_hosted"]["latency_ms"])
                for r in sh_successful
                if isinstance(r.get("self_hosted", {}).get("latency_ms"), (int, float))
            ]
            sh_tokens = [float(int(r["self_hosted"].get("estimated_tokens", 0) or 0)) for r in sh_successful]
            sh_p90_latency = _pct(sh_latencies, 90.0)
            sh_p95_latency = _pct(sh_latencies, 95.0)
            sh_p90_tokens = _pct(sh_tokens, 90.0)
            sh_p95_tokens = _pct(sh_tokens, 95.0)

            # Calculate failed_queries: only count queries where BOTH endpoints failed
            failed_queries = sum(
                1
                for r in query_results
                if r.get("mode") == "both"
                and not r.get("cloud", {}).get("succeeded")
                and not r.get("self_hosted", {}).get("succeeded")
            )

            benchmark_result = {
                "metadata": {
                    "category": category,
                    "version": version,
                    "mode": "both",
                    "run_timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "concurrency": concurrency,
                    "duration_ms": int((time.time() - run_start) * 1000),
                    "total_queries": len(queries),
                    "cloud_successful_queries": len(cloud_successful),
                    "self_hosted_successful_queries": len(sh_successful),
                    "failed_queries": failed_queries,
                },
                "metrics": {
                    "cloud": {
                        "average_accuracy": round(cloud_avg_accuracy, 3),
                        "average_overall_score": round(cloud_avg_overall, 3),
                        "average_latency_ms": round(cloud_avg_latency, 2),
                        "average_credits_per_query": round(cloud_avg_credits, 2),
                        "total_credits": cloud_total_credits,
                        "average_tokens": int(cloud_avg_tokens),
                        "p90_latency_ms": round(cloud_p90_latency, 2),
                        "p95_latency_ms": round(cloud_p95_latency, 2),
                        "p90_tokens": int(cloud_p90_tokens),
                        "p95_tokens": int(cloud_p95_tokens),
                    },
                    "self_hosted": {
                        "average_accuracy": round(sh_avg_accuracy, 3),
                        "average_overall_score": round(sh_avg_overall, 3),
                        "average_latency_ms": round(sh_avg_latency, 2),
                        "average_credits_per_query": round(sh_avg_credits, 2),
                        "total_credits": self_hosted_total_credits,
                        "average_tokens": int(sh_avg_tokens),
                        "p90_latency_ms": round(sh_p90_latency, 2),
                        "p95_latency_ms": round(sh_p95_latency, 2),
                        "p90_tokens": int(sh_p90_tokens),
                        "p95_tokens": int(sh_p95_tokens),
                    },
                    "comparison": {
                        "accuracy_diff": round(cloud_avg_accuracy - sh_avg_accuracy, 3),
                        "latency_diff_ms": round(cloud_avg_latency - sh_avg_latency, 2),
                        "cost_diff": round(cloud_avg_credits - sh_avg_credits, 2),
                        "cloud_faster": cloud_avg_latency < sh_avg_latency,
                        "self_hosted_cheaper": sh_avg_credits < cloud_avg_credits,
                    },
                },
                "query_results": query_results,
            }

            # Add classifier validation metrics if available
            if classifier_metrics:
                benchmark_result["metrics"]["classifier_validation"] = classifier_metrics
        else:
            # Single-mode metrics (original behavior)
            successful_results = [r for r in query_results if "scores" in r]
            avg_accuracy = (
                sum(r["scores"]["accuracy"] for r in successful_results) / len(successful_results)
                if successful_results
                else 0
            )
            avg_overall = (
                sum(r["scores"]["overall"] for r in successful_results) / len(successful_results)
                if successful_results
                else 0
            )
            avg_latency = total_latency / len(successful_results) if successful_results else 0
            avg_credits = total_credits / len(successful_results) if successful_results else 0
            # Percentiles
            latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in successful_results]
            tokens_list = [float(int(r.get("estimated_tokens", 0) or 0)) for r in successful_results]

            def _pct(values: List[float], p: float) -> float:
                if not values:
                    return 0.0
                vals = sorted(values)
                k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
                return float(vals[k])

            p90_latency = _pct(latencies, 90.0)
            p95_latency = _pct(latencies, 95.0)
            p90_tokens = _pct(tokens_list, 90.0)
            p95_tokens = _pct(tokens_list, 95.0)

            benchmark_result = {
                "metadata": {
                    "category": category,
                    "version": version,
                    "mode": self.mode,
                    "run_timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "concurrency": concurrency,
                    "duration_ms": int((time.time() - run_start) * 1000),
                    "total_queries": len(queries),
                    "successful_queries": len(successful_results),
                    "failed_queries": len(queries) - len(successful_results),
                },
                "metrics": {
                    "average_accuracy": round(avg_accuracy, 3),
                    "average_overall_score": round(avg_overall, 3),
                    "average_latency_ms": round(avg_latency, 2),
                    "average_credits_per_query": round(avg_credits, 2),
                    "total_credits": total_credits,
                    "p90_latency_ms": round(p90_latency, 2),
                    "p95_latency_ms": round(p95_latency, 2),
                    "p90_tokens": int(p90_tokens),
                    "p95_tokens": int(p95_tokens),
                },
                "query_results": query_results,
            }

            # Add classifier validation metrics if available
            if classifier_metrics:
                benchmark_result["metrics"]["classifier_validation"] = classifier_metrics

        # Compare against dataset baselines if present
        baselines = metadata.get("baselines") if isinstance(metadata, dict) else None
        if baselines:
            try:
                # Validate baselines before comparison
                self._validate_baselines(baselines)
                comparison_report = self.compare_against_baselines(benchmark_result, baselines)
                benchmark_result["regression_analysis"] = comparison_report

                # Emit concise regression logs for visibility
                mode = comparison_report.get("mode", benchmark_result["metadata"].get("mode"))
                if mode == "both":
                    cloud = comparison_report.get("cloud", {})
                    sh = comparison_report.get("self_hosted", {})
                    if cloud.get("has_regressions"):
                        logger.warning(
                            f"Cloud baseline regressions detected for {category}: {cloud.get('regressions')}"
                        )
                        if os.environ.get("BENCHMARK_REGRESSION_VERBOSE", "").lower() in {"1", "true", "yes"}:
                            c = cloud.get("comparison", {})
                            logger.info(
                                f"Cloud deltas: accuracy={c.get('accuracy_change_pct', 0):+.2f}%, "
                                f"cost={c.get('cost_change_pct', 0):+.2f}%, latency={c.get('latency_change_pct', 0):+.2f}%"
                            )
                        if os.environ.get("BENCHMARK_REGRESSION_JSON", "").lower() in {"1", "true", "yes"}:
                            logger.info(
                                json.dumps(
                                    {
                                        "event": "regression",
                                        "category": category,
                                        "mode": "cloud",
                                        "flags": cloud.get("regressions"),
                                        "comparison": cloud.get("comparison"),
                                    }
                                )
                            )
                    if sh.get("has_regressions"):
                        logger.warning(
                            f"Self-hosted baseline regressions detected for {category}: {sh.get('regressions')}"
                        )
                        if os.environ.get("BENCHMARK_REGRESSION_VERBOSE", "").lower() in {"1", "true", "yes"}:
                            c = sh.get("comparison", {})
                            logger.info(
                                f"Self-hosted deltas: accuracy={c.get('accuracy_change_pct', 0):+.2f}%, "
                                f"cost={c.get('cost_change_pct', 0):+.2f}%, latency={c.get('latency_change_pct', 0):+.2f}%"
                            )
                        if os.environ.get("BENCHMARK_REGRESSION_JSON", "").lower() in {"1", "true", "yes"}:
                            logger.info(
                                json.dumps(
                                    {
                                        "event": "regression",
                                        "category": category,
                                        "mode": "self_hosted",
                                        "flags": sh.get("regressions"),
                                        "comparison": sh.get("comparison"),
                                    }
                                )
                            )
                else:
                    if comparison_report.get("has_regressions"):
                        logger.warning(
                            f"Baseline regressions detected for {category} ({mode}): {comparison_report.get('regressions')}"
                        )
                        if os.environ.get("BENCHMARK_REGRESSION_VERBOSE", "").lower() in {"1", "true", "yes"}:
                            c = comparison_report.get("comparison", {})
                            logger.info(
                                f"Deltas: accuracy={c.get('accuracy_change_pct', 0):+.2f}%, "
                                f"cost={c.get('cost_change_pct', 0):+.2f}%, latency={c.get('latency_change_pct', 0):+.2f}%"
                            )
                        if os.environ.get("BENCHMARK_REGRESSION_JSON", "").lower() in {"1", "true", "yes"}:
                            logger.info(
                                json.dumps(
                                    {
                                        "event": "regression",
                                        "category": category,
                                        "mode": mode,
                                        "flags": comparison_report.get("regressions"),
                                        "comparison": comparison_report.get("comparison"),
                                    }
                                )
                            )
            except Exception as e:
                logger.warning(f"Baseline comparison failed: {e}")

        # Track benchmark run with tracker (provide baselines when available)
        if self.benchmark_tracker:
            try:
                # If dataset baselines are available, provide them to the tracker
                if baselines:
                    # Tracker expects mapping by category; it will normalize keys internally
                    self.benchmark_tracker.baseline_metrics = {category: baselines}
                self.benchmark_tracker.track_benchmark_run(benchmark_result)
            except Exception as e:
                logger.warning(f"Error tracking benchmark run: {e}")

        self.results.append(benchmark_result)

        # Log completion with mode-specific metrics
        if self.mode == "both":
            logger.info(
                f"Benchmark complete: {category} v{version} - Cloud Avg: {cloud_avg_overall:.3f}, Self-Hosted Avg: {sh_avg_overall:.3f}"
            )
        else:
            logger.info(f"Benchmark complete: {category} v{version} - Avg Score: {avg_overall:.3f}")

        # Clear run context
        try:
            self._current_run = None
            if self.benchmark_tracker:
                self.benchmark_tracker.current_run_id = None
        except Exception:
            pass

        return benchmark_result

    def compare_against_baselines(self, benchmark_result: Dict[str, Any], baselines: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare benchmark results against baseline metadata.

        Args:
            benchmark_result: Current benchmark results (with cloud/self_hosted metrics if mode="both")
            baselines: Baseline metadata from benchmark JSON

        Returns:
            Comparison report with regression flags for each mode
        """
        from scripts.pharmaceutical_benchmark_report import ComparisonReportGenerator

        mode = benchmark_result["metadata"].get("mode", "cloud")

        # Helper function to check classifier validation regressions
        def _check_classifier_regression(current_metrics, baselines):
            """Check for classifier validation regressions."""
            classifier_flags = []

            # Get current classifier metrics
            classifier_validation = current_metrics.get("classifier_validation")
            if not classifier_validation:
                return classifier_flags

            # Get baseline classifier accuracy (default 0.95 if not specified)
            baseline_classifier_accuracy = baselines.get("classifier_validation", {}).get("overall_accuracy", 0.95)
            current_classifier_accuracy = classifier_validation.get("overall_accuracy", 0.0)

            # Regression threshold: 5% drop in classifier accuracy
            threshold = 0.05
            accuracy_drop = baseline_classifier_accuracy - current_classifier_accuracy

            if accuracy_drop > threshold:
                classifier_flags.append(
                    {
                        "type": "classifier_validation_regression",
                        "message": f"Classifier accuracy dropped by {accuracy_drop:.1%} (from {baseline_classifier_accuracy:.1%} to {current_classifier_accuracy:.1%})",
                        "baseline": baseline_classifier_accuracy,
                        "current": current_classifier_accuracy,
                        "threshold": threshold,
                    }
                )

            return classifier_flags

        if mode == "both":
            # Compare both cloud and self-hosted against their respective baselines
            cloud_metrics = benchmark_result["metrics"]["cloud"]
            sh_metrics = benchmark_result["metrics"]["self_hosted"]

            # Compare cloud against cloud baseline
            cloud_flags = ComparisonReportGenerator._check_regressions(
                baseline_accuracy=baselines["cloud"]["average_accuracy"],
                current_accuracy=cloud_metrics["average_accuracy"],
                baseline_cost=baselines["cloud"]["average_cost_per_query"],
                current_cost=cloud_metrics["average_credits_per_query"],
                baseline_latency=baselines["cloud"]["average_latency_ms"],
                current_latency=cloud_metrics["average_latency_ms"],
            )

            # Compare self_hosted against self_hosted baseline
            sh_flags = ComparisonReportGenerator._check_regressions(
                baseline_accuracy=baselines["self_hosted"]["average_accuracy"],
                current_accuracy=sh_metrics["average_accuracy"],
                baseline_cost=baselines["self_hosted"]["average_cost_per_query"],
                current_cost=sh_metrics["average_credits_per_query"],
                baseline_latency=baselines["self_hosted"]["average_latency_ms"],
                current_latency=sh_metrics["average_latency_ms"],
            )

            # Check classifier validation regressions (shared across modes)
            classifier_flags = _check_classifier_regression(benchmark_result["metrics"], baselines)

            # Add classifier flags to both cloud and self_hosted
            cloud_flags.extend(classifier_flags)
            sh_flags.extend(classifier_flags)

            return {
                "mode": "both",
                "cloud": {
                    "regressions": cloud_flags,
                    "has_regressions": len(cloud_flags) > 0,
                    "comparison": {
                        "accuracy_change_pct": (
                            (cloud_metrics["average_accuracy"] - baselines["cloud"]["average_accuracy"])
                            / baselines["cloud"]["average_accuracy"]
                            * 100
                        )
                        if baselines["cloud"]["average_accuracy"] > 0
                        else 0,
                        "cost_change_pct": (
                            (cloud_metrics["average_credits_per_query"] - baselines["cloud"]["average_cost_per_query"])
                            / baselines["cloud"]["average_cost_per_query"]
                            * 100
                        )
                        if baselines["cloud"]["average_cost_per_query"] > 0
                        else 0,
                        "latency_change_pct": (
                            (cloud_metrics["average_latency_ms"] - baselines["cloud"]["average_latency_ms"])
                            / baselines["cloud"]["average_latency_ms"]
                            * 100
                        )
                        if baselines["cloud"]["average_latency_ms"] > 0
                        else 0,
                    },
                },
                "self_hosted": {
                    "regressions": sh_flags,
                    "has_regressions": len(sh_flags) > 0,
                    "comparison": {
                        "accuracy_change_pct": (
                            (sh_metrics["average_accuracy"] - baselines["self_hosted"]["average_accuracy"])
                            / baselines["self_hosted"]["average_accuracy"]
                            * 100
                        )
                        if baselines["self_hosted"]["average_accuracy"] > 0
                        else 0,
                        "cost_change_pct": (
                            (
                                sh_metrics["average_credits_per_query"]
                                - baselines["self_hosted"]["average_cost_per_query"]
                            )
                            / baselines["self_hosted"]["average_cost_per_query"]
                            * 100
                        )
                        if baselines["self_hosted"]["average_cost_per_query"] > 0
                        else 0,
                        "latency_change_pct": (
                            (sh_metrics["average_latency_ms"] - baselines["self_hosted"]["average_latency_ms"])
                            / baselines["self_hosted"]["average_latency_ms"]
                            * 100
                        )
                        if baselines["self_hosted"]["average_latency_ms"] > 0
                        else 0,
                    },
                },
                "overall_has_regressions": len(cloud_flags) > 0 or len(sh_flags) > 0,
            }
        else:
            # Single mode comparison
            metrics = benchmark_result["metrics"]
            baseline_key = "cloud" if mode == "cloud" else "self_hosted"

            flags = ComparisonReportGenerator._check_regressions(
                baseline_accuracy=baselines[baseline_key]["average_accuracy"],
                current_accuracy=metrics["average_accuracy"],
                baseline_cost=baselines[baseline_key]["average_cost_per_query"],
                current_cost=metrics["average_credits_per_query"],
                baseline_latency=baselines[baseline_key]["average_latency_ms"],
                current_latency=metrics["average_latency_ms"],
            )

            # Check classifier validation regressions
            classifier_flags = _check_classifier_regression(metrics, baselines)
            flags.extend(classifier_flags)

            return {
                "mode": mode,
                "regressions": flags,
                "has_regressions": len(flags) > 0,
                "comparison": {
                    "accuracy_change_pct": (
                        (metrics["average_accuracy"] - baselines[baseline_key]["average_accuracy"])
                        / baselines[baseline_key]["average_accuracy"]
                        * 100
                    )
                    if baselines[baseline_key]["average_accuracy"] > 0
                    else 0,
                    "cost_change_pct": (
                        (metrics["average_credits_per_query"] - baselines[baseline_key]["average_cost_per_query"])
                        / baselines[baseline_key]["average_cost_per_query"]
                        * 100
                    )
                    if baselines[baseline_key]["average_cost_per_query"] > 0
                    else 0,
                    "latency_change_pct": (
                        (metrics["average_latency_ms"] - baselines[baseline_key]["average_latency_ms"])
                        / baselines[baseline_key]["average_latency_ms"]
                        * 100
                    )
                    if baselines[baseline_key]["average_latency_ms"] > 0
                    else 0,
                },
            }

    def save_results(self, output_dir: Optional[str] = None) -> Path:
        """Save benchmark results to file."""
        output_dir = Path(
            output_dir or self.config.config.get("storage", {}).get("results_directory", "results/benchmark_runs")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def _validate_baselines(self, baselines: Dict[str, Any]) -> None:
        """Best-practice validation for baseline structure (non-fatal)."""
        try:
            if not isinstance(baselines, dict):
                logger.warning("Baselines malformed: expected dict")
                return
            for endpoint in ("cloud", "self_hosted"):
                if endpoint not in baselines:
                    # Allow single-mode datasets without both endpoints
                    continue
                b = baselines.get(endpoint, {})
                for key in ("average_accuracy", "average_latency_ms"):
                    if not isinstance(b.get(key, 0), (int, float)):
                        logger.warning(f"Baseline '{endpoint}.{key}' missing or non-numeric")
                # cost key can be cost_per_query or credits per query
                cost_val = b.get("average_cost_per_query", b.get("average_credits_per_query", 0))
                if not isinstance(cost_val, (int, float)):
                    logger.warning(f"Baseline '{endpoint}.average_cost_per_query' missing or non-numeric")
        except Exception as e:
            logger.debug(f"Baseline validation error (non-fatal): {e}")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Run pharmaceutical benchmarks")
    parser.add_argument("--category", help="Benchmark category to run (if not specified, runs all)")
    parser.add_argument("--version", type=int, default=1, help="Benchmark version to run")
    parser.add_argument("--config", default="config/benchmarks.yaml", help="Path to configuration file")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument(
        "--mode",
        choices=["cloud", "self_hosted", "both"],
        default="cloud",
        help="Execution mode: cloud (NVIDIA Build), self_hosted (local NIM), or both for comparison",
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Use simulation mode instead of real API calls (for testing)"
    )
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved effective settings (no execution) and exit"
    )
    parser.add_argument(
        "--inspect-preset",
        action="store_true",
        help="Print a per-category Markdown table of effective settings and exit",
    )
    parser.add_argument("--preset", help="Apply tuned settings from config/benchmark_presets.yaml (preset name)")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Max concurrent queries per category (high-ROI speedup)"
    )
    parser.add_argument(
        "--auto-concurrency",
        action="store_true",
        help="Automatically choose a sensible concurrency based on dataset size",
    )
    parser.add_argument(
        "--launch-stagger-ms",
        type=int,
        default=50,
        help="Stagger (ms) between launching concurrent requests to reduce burst rate",
    )
    parser.add_argument(
        "--skip-classifier-validation",
        action="store_true",
        help="Skip classifier-vs-expected validation to speed up runs",
    )
    parser.add_argument(
        "--summary-output", help="When running all categories, also write a consolidated summary JSON to this path"
    )
    parser.add_argument(
        "--min-cloud-score", type=float, help="Fail if any category cloud average overall score is below this threshold"
    )
    parser.add_argument(
        "--max-cloud-latency-ms", type=float, help="Fail if any category cloud p95 latency exceeds this threshold (ms)"
    )
    parser.add_argument(
        "--adaptive-concurrency",
        action="store_true",
        help="Dynamically adjust concurrency per window based on latency/failures",
    )
    parser.add_argument(
        "--adapt-latency-high-ms",
        type=float,
        default=6500.0,
        help="If average cloud latency per window exceeds this, reduce concurrency",
    )
    parser.add_argument(
        "--adapt-latency-low-ms",
        type=float,
        default=2500.0,
        help="If average cloud latency per window is below this, raise concurrency",
    )
    parser.add_argument("--adapt-step", type=int, default=1, help="Concurrency step size when adapting")
    parser.add_argument(
        "--enforce-budget",
        action="store_true",
        help="Stop early if configured research project budget utilization exceeds threshold",
    )
    parser.add_argument(
        "--budget-stop-utilization",
        type=float,
        default=0.95,
        help="Utilization threshold [0-1] at which to stop additional queries when --enforce-budget is set",
    )
    parser.add_argument(
        "--max-queries", type=int, help="Process at most N queries from the dataset (high-ROI sampling)"
    )
    parser.add_argument(
        "--fail-on-regressions", action="store_true", help="Return non-zero exit code when regressions are detected"
    )
    parser.add_argument(
        "--preflight", action="store_true", help="Run a per-category latency health check to auto-tune concurrency"
    )
    parser.add_argument(
        "--preflight-only", action="store_true", help="Run preflight and exit without executing benchmarks"
    )
    parser.add_argument(
        "--preflight-then-run",
        action="store_true",
        help="Convenience flag to run preflight before executing benchmarks",
    )
    parser.add_argument(
        "--preflight-min-concurrency",
        type=int,
        help="Fail if any category recommended concurrency is below this threshold during preflight",
    )
    parser.add_argument(
        "--fail-on-preflight",
        action="store_true",
        help="Return non-zero exit code if preflight fails the min-concurrency gate",
    )
    parser.add_argument(
        "--preflight-map-input", help="Path to a prior preflight JSON; apply its recommended concurrency"
    )
    parser.add_argument(
        "--skip-unsupported-self-hosted",
        action="store_true",
        default=True,
        help="Skip self-hosted endpoint if detected unsupported (e.g., NeMo chat not supported)",
    )
    parser.add_argument(
        "--preflight-output", help="Write preflight results to this path (JSON). Also writes CSV next to it."
    )
    parser.add_argument(
        "--preflight-sample-count",
        type=int,
        default=1,
        help="Number of sample queries per category to measure in preflight (cache warm-up)",
    )
    parser.add_argument("--warm-cache-count", type=int, help="Alias for --preflight-sample-count (for cache warm-up)")

    args = parser.parse_args()

    # Load configuration
    config = BenchmarkConfig(args.config)

    # List presets and exit if requested
    if args.list_presets:
        try:
            preset_path = Path("config/benchmark_presets.yaml")
            if not preset_path.exists():
                print("No preset file found at config/benchmark_presets.yaml")
                return 0
            with open(preset_path) as pf:
                presets_yaml = yaml.safe_load(pf) or {}
            presets = list((presets_yaml.get("presets") or {}).keys())
            if presets:
                print("Available presets:")
                for name in presets:
                    print(f" - {name}")
            else:
                print("No presets defined")
        except Exception as e:
            print(f"Failed to list presets: {e}")
        return 0

    # Initialize runner with mode selection
    use_real_clients = not args.simulate
    runner = BenchmarkRunner(config, use_real_clients=use_real_clients, mode=args.mode)
    # Optional runtime behaviors
    try:
        runner._skip_unsupported_self_hosted = bool(getattr(args, "skip_unsupported_self_hosted", True))
    except Exception:
        runner._skip_unsupported_self_hosted = True

    try:
        # Apply alias: preflight-then-run implies preflight=True
        if getattr(args, "preflight_then_run", False):
            try:
                args.preflight = True
                # Ensure we don't exit early
                args.preflight_only = False
            except Exception:
                pass

        # Load preset file if requested
        preset = None
        if args.preset:
            preset_path = Path("config/benchmark_presets.yaml")
            if preset_path.exists():
                try:
                    with open(preset_path) as pf:
                        presets_yaml = yaml.safe_load(pf) or {}
                    preset = (presets_yaml.get("presets", {}) or {}).get(args.preset)
                    if not preset:
                        logger.warning(f"Preset '{args.preset}' not found in {preset_path}")
                except Exception as e:
                    logger.warning(f"Failed to load preset '{args.preset}': {e}")
            else:
                logger.warning(f"Preset file not found: {preset_path}")
        # If inspect-preset requested, compute and print per-category effective settings then exit
        if args.inspect_preset:
            try:
                available = runner.loader.list_available_benchmarks()
                if not available:
                    print("No benchmarks found to inspect")
                    return 0
                lines = []
                lines.append(
                    "| Category | Version | Concurrency | Max Qs | Adaptive | Lat High (ms) | Lat Low (ms) | Step | Skip Classifier | Stagger (ms) |"
                )
                lines.append("|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|")
                for category, version in available:
                    # Base effective values from CLI
                    eff_conc = args.concurrency
                    if args.auto_concurrency:
                        try:
                            bench = runner.loader.load_benchmark(category, version)
                            qn = int(len(bench.get("queries", [])))
                        except Exception:
                            qn = 10
                        if qn <= 10:
                            eff_conc = 2
                        elif qn <= 25:
                            eff_conc = 4
                        elif qn <= 100:
                            eff_conc = 6
                        else:
                            eff_conc = 8
                    adapt = args.adaptive_concurrency
                    ah = args.adapt_latency_high_ms
                    al = args.adapt_latency_low_ms
                    astep = args.adapt_step
                    scv = args.skip_classifier_validation
                    mqs = args.max_queries
                    # Apply preset overrides
                    if preset:
                        g = preset.get("global") or {}
                        eff_conc = g.get("concurrency", eff_conc)
                        scv = g.get("skip_classifier_validation", scv)
                        adapt = g.get("adaptive_concurrency", adapt)
                        ah = g.get("adapt_latency_high_ms", ah)
                        al = g.get("adapt_latency_low_ms", al)
                        astep = g.get("adapt_step", astep)
                        mqs = g.get("max_queries", mqs)
                        co = (preset.get("categories") or {}).get(category, {})
                        if co:
                            eff_conc = co.get("concurrency", eff_conc)
                            mqs = co.get("max_queries", mqs)
                            ah = co.get("adapt_latency_high_ms", ah)
                            al = co.get("adapt_latency_low_ms", al)
                    stagger = int(os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50"))
                    lines.append(
                        f"| {category} | {version} | {eff_conc} | {mqs if mqs is not None else ''} | {str(bool(adapt))} | {int(ah)} | {int(al)} | {int(astep)} | {str(bool(scv))} | {stagger} |"
                    )
                print("\n".join(lines))
            except Exception as e:
                print(f"Failed to inspect preset: {e}")
            return 0

        if args.category:
            # Run specific category
            logger.info(f"Running benchmark for category: {args.category}")
            # Resolve effective concurrency
            eff_conc = args.concurrency
            if args.auto_concurrency:
                try:
                    # Quick peek at dataset size for heuristic
                    bench = runner.loader.load_benchmark(args.category, args.version)
                    qn = int(len(bench.get("queries", [])))
                except Exception:
                    qn = 10
                if qn <= 10:
                    eff_conc = 2
                elif qn <= 25:
                    eff_conc = 4
                elif qn <= 100:
                    eff_conc = 6
                else:
                    eff_conc = 8

            # Apply preset overrides if present
            adapt = args.adaptive_concurrency
            ah = args.adapt_latency_high_ms
            al = args.adapt_latency_low_ms
            astep = args.adapt_step
            scv = args.skip_classifier_validation
            mqs = args.max_queries
            enf = args.enforce_budget
            bstop = args.budget_stop_utilization
            if preset:
                g = preset.get("global") or {}
                os.environ["BENCHMARK_LAUNCH_STAGGER_MS"] = str(
                    g.get("launch_stagger_ms", os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50"))
                )
                eff_conc = g.get("concurrency", eff_conc)
                scv = g.get("skip_classifier_validation", scv)
                adapt = g.get("adaptive_concurrency", adapt)
                ah = g.get("adapt_latency_high_ms", ah)
                al = g.get("adapt_latency_low_ms", al)
                astep = g.get("adapt_step", astep)
                mqs = g.get("max_queries", mqs)
                enf = g.get("enforce_budget", enf)
                bstop = g.get("budget_stop_utilization", bstop)
                # Gating thresholds from preset (global)
                try:
                    gating = g.get("gating", {}) or {}
                    if "min_cloud_score" in gating and getattr(args, "min_cloud_score", None) is None:
                        args.min_cloud_score = float(gating.get("min_cloud_score"))
                    if "max_cloud_latency_ms" in gating and getattr(args, "max_cloud_latency_ms", None) is None:
                        args.max_cloud_latency_ms = float(gating.get("max_cloud_latency_ms"))
                except Exception:
                    pass
                # Category override
                co = (preset.get("categories") or {}).get(args.category, {})
                if co:
                    eff_conc = co.get("concurrency", eff_conc)
                    mqs = co.get("max_queries", mqs)
                    ah = co.get("adapt_latency_high_ms", ah)
                    al = co.get("adapt_latency_low_ms", al)

            # If dry-run, print effective settings and exit
            if args.dry_run:
                eff = {
                    "mode": args.mode,
                    "category": args.category,
                    "version": args.version,
                    "preset": args.preset,
                    "effective": {
                        "concurrency": eff_conc,
                        "skip_classifier_validation": scv,
                        "adaptive_concurrency": adapt,
                        "adapt_latency_high_ms": ah,
                        "adapt_latency_low_ms": al,
                        "adapt_step": astep,
                        "max_queries": mqs,
                        "enforce_budget": enf,
                        "budget_stop_utilization": bstop,
                        "launch_stagger_ms": int(os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50")),
                    },
                }
                print(json.dumps(eff, indent=2))
                return 0

            result = runner.run_benchmark(
                args.category,
                args.version,
                concurrency=eff_conc,
                skip_classifier_validation=scv,
                enforce_budget=enf,
                budget_stop_utilization=bstop,
                max_queries=mqs,
                adaptive_concurrency=adapt,
                adapt_latency_high_ms=ah,
                adapt_latency_low_ms=al,
                adapt_step=astep,
            )
            print(f"\nResults for {args.category} v{args.version}:")

            # Handle dual-mode (cloud + self_hosted) vs single-mode output
            if result["metadata"].get("mode") == "both":
                # Print dual-mode metrics with comparison
                print(f"  Cloud:")
                print(f"    Average Score: {result['metrics']['cloud']['average_overall_score']:.3f}")
                print(f"    Average Latency: {result['metrics']['cloud']['average_latency_ms']:.2f} ms")
                print(f"    Average Credits: {result['metrics']['cloud']['average_credits_per_query']:.2f}")
                if "average_tokens" in result["metrics"]["cloud"]:
                    print(f"    Average Tokens: {result['metrics']['cloud']['average_tokens']}")
                print(f"  Self-Hosted:")
                print(f"    Average Score: {result['metrics']['self_hosted']['average_overall_score']:.3f}")
                print(f"    Average Latency: {result['metrics']['self_hosted']['average_latency_ms']:.2f} ms")
                print(f"    Average Credits: {result['metrics']['self_hosted']['average_credits_per_query']:.2f}")
                if "average_tokens" in result["metrics"]["self_hosted"]:
                    print(f"    Average Tokens: {result['metrics']['self_hosted']['average_tokens']}")
                print(f"  Comparison:")
                print(f"    Accuracy Diff: {result['metrics']['comparison']['accuracy_diff']:+.3f}")
                print(f"    Latency Diff: {result['metrics']['comparison']['latency_diff_ms']:+.2f} ms")
                print(f"    Cost Diff: {result['metrics']['comparison']['cost_diff']:+.2f} credits")
            else:
                # Print single-mode metrics (original behavior)
                print(f"  Average Score: {result['metrics']['average_overall_score']:.3f}")
                print(f"  Average Latency: {result['metrics']['average_latency_ms']:.2f} ms")
                print(f"  Average Credits: {result['metrics']['average_credits_per_query']:.2f}")
                if "average_tokens" in result["metrics"]:
                    print(f"  Average Tokens: {result['metrics']['average_tokens']}")
        else:
            # Run all available benchmarks
            logger.info("Running all available benchmarks")
            available = runner.loader.list_available_benchmarks()
            logger.info(f"Found {len(available)} benchmarks")

            consolidated: List[Dict[str, Any]] = []

            # Optional preflight to auto-tune per-category concurrency based on measured cloud latency
            preflight_map: Dict[str, int] = {}
            if args.preflight:
                print("\nPreflight: measuring per-category cloud latency...")

                # Simple recommender mapping (ms -> concurrency)
                def _recommend_conc(lat_ms: float) -> int:
                    if lat_ms <= 2000:
                        return 8
                    if lat_ms <= 3500:
                        return 6
                    if lat_ms <= 5500:
                        return 4
                    if lat_ms <= 8000:
                        return 2
                    return 1

                pf_lines = [
                    "| Category | Version | Cloud Latency (ms) | Recommended Concurrency |",
                    "|---|---:|---:|---:|",
                ]
                pf_records = []
                sample_count = max(1, int(args.preflight_sample_count or 1))
                if args.warm_cache_count is not None:
                    try:
                        sample_count = max(sample_count, int(args.warm_cache_count))
                    except Exception:
                        pass
                for category, version in available:
                    try:
                        bench = runner.loader.load_benchmark(category, version)
                        q = bench.get("queries") or []
                        if not q:
                            pf_lines.append(f"| {category} | {version} | 0 | 1 |")
                            pf_records.append(
                                {
                                    "category": category,
                                    "version": version,
                                    "avg_cloud_latency_ms": 0.0,
                                    "recommended_concurrency": 1,
                                }
                            )
                            preflight_map[category] = 1
                            continue
                        lats = []
                        for i in range(min(sample_count, len(q))):
                            sample_q = q[i].get("query") or ""
                            result = runner.execute_query_both(
                                sample_q, timeout=runner.config.config.get("execution", {}).get("timeout_seconds", 30)
                            )
                            cloud_lat = float(result.get("cloud", {}).get("latency_ms", 0.0) or 0.0)
                            if cloud_lat > 0:
                                lats.append(cloud_lat)
                        avg_lat = sum(lats) / len(lats) if lats else 0.0
                        rec = _recommend_conc(avg_lat)
                        pf_lines.append(f"| {category} | {version} | {avg_lat:.2f} | {rec} |")
                        pf_records.append(
                            {
                                "category": category,
                                "version": version,
                                "avg_cloud_latency_ms": round(avg_lat, 2),
                                "recommended_concurrency": rec,
                                "samples": len(lats),
                            }
                        )
                        preflight_map[category] = rec
                    except Exception as e:
                        logger.warning(f"Preflight failed for {category}: {e}")
                        pf_lines.append(f"| {category} | {version} | NA | 2 |")
                        pf_records.append(
                            {
                                "category": category,
                                "version": version,
                                "avg_cloud_latency_ms": None,
                                "recommended_concurrency": 2,
                                "samples": 0,
                                "error": str(e),
                            }
                        )
                        preflight_map[category] = 2

                print("\n".join(pf_lines))
                # Evaluate preflight gating (min recommended concurrency)
                if args.preflight_min_concurrency is not None:
                    try:
                        minc = int(args.preflight_min_concurrency)
                        offenders = [cat for cat, rec in preflight_map.items() if int(rec) < minc]
                        if offenders:
                            print(
                                f"Preflight gate FAIL: recommended concurrency below {minc} for: {', '.join(offenders)}"
                            )
                            if args.fail_on_preflight:
                                return 4
                    except Exception as e:
                        logger.warning(f"Preflight gate evaluation failed: {e}")
                # Persist preflight output if requested
                if args.preflight_output:
                    try:
                        outp = Path(args.preflight_output)
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        with open(outp, "w") as jf:
                            json.dump({"timestamp": datetime.now().isoformat(), "records": pf_records}, jf, indent=2)
                        # Also write CSV next to it
                        csvp = outp.with_suffix(".csv")
                        import csv

                        with open(csvp, "w", newline="") as cf:
                            writer = csv.writer(cf)
                            writer.writerow(
                                ["category", "version", "avg_cloud_latency_ms", "recommended_concurrency", "samples"]
                            )
                            for r in pf_records:
                                writer.writerow(
                                    [
                                        r.get("category"),
                                        r.get("version"),
                                        r.get("avg_cloud_latency_ms"),
                                        r.get("recommended_concurrency"),
                                        r.get("samples"),
                                    ]
                                )
                        print(f"Preflight saved to: {outp} and {csvp}")
                    except Exception as e:
                        logger.warning(f"Failed to persist preflight output: {e}")
                if args.preflight_only:
                    return 0
            # Load prior preflight map if provided
            if args.preflight_map_input:
                try:
                    with open(args.preflight_map_input) as jf:
                        pdata = json.load(jf)
                    for rec in pdata.get("records", []):
                        cat = rec.get("category")
                        rec_c = rec.get("recommended_concurrency")
                        if cat and rec_c:
                            preflight_map[cat] = int(rec_c)
                except Exception as e:
                    logger.warning(f"Failed to load preflight map input: {e}")
            for category, version in available:
                # Resolve effective concurrency for this dataset
                eff_conc = args.concurrency
                if args.auto_concurrency:
                    try:
                        bench = runner.loader.load_benchmark(category, version)
                        qn = int(len(bench.get("queries", [])))
                    except Exception:
                        qn = 10
                    if qn <= 10:
                        eff_conc = 2
                    elif qn <= 25:
                        eff_conc = 4
                    elif qn <= 100:
                        eff_conc = 6
                    else:
                        eff_conc = 8

                # Apply preflight recommendation if present (take the min with auto heuristic)
                if args.preflight and category in preflight_map:
                    try:
                        eff_conc = min(int(eff_conc), int(preflight_map[category]))
                    except Exception:
                        pass

                # Apply preset overrides per category
                adapt = args.adaptive_concurrency
                ah = args.adapt_latency_high_ms
                al = args.adapt_latency_low_ms
                astep = args.adapt_step
                scv = args.skip_classifier_validation
                mqs = args.max_queries
                enf = args.enforce_budget
                bstop = args.budget_stop_utilization
                if preset:
                    g = preset.get("global") or {}
                    os.environ["BENCHMARK_LAUNCH_STAGGER_MS"] = str(
                        g.get("launch_stagger_ms", os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50"))
                    )
                    eff_conc = g.get("concurrency", eff_conc)
                    scv = g.get("skip_classifier_validation", scv)
                    adapt = g.get("adaptive_concurrency", adapt)
                    ah = g.get("adapt_latency_high_ms", ah)
                    al = g.get("adapt_latency_low_ms", al)
                    astep = g.get("adapt_step", astep)
                    mqs = g.get("max_queries", mqs)
                    enf = g.get("enforce_budget", enf)
                    bstop = g.get("budget_stop_utilization", bstop)
                    # Gating thresholds from preset (global)
                    try:
                        gating = g.get("gating", {}) or {}
                        if "min_cloud_score" in gating and getattr(args, "min_cloud_score", None) is None:
                            args.min_cloud_score = float(gating.get("min_cloud_score"))
                        if "max_cloud_latency_ms" in gating and getattr(args, "max_cloud_latency_ms", None) is None:
                            args.max_cloud_latency_ms = float(gating.get("max_cloud_latency_ms"))
                    except Exception:
                        pass
                    co = (preset.get("categories") or {}).get(category, {})
                    if co:
                        eff_conc = co.get("concurrency", eff_conc)
                        mqs = co.get("max_queries", mqs)
                        ah = co.get("adapt_latency_high_ms", ah)
                        al = co.get("adapt_latency_low_ms", al)

                r = runner.run_benchmark(
                    category,
                    version,
                    concurrency=eff_conc,
                    skip_classifier_validation=scv,
                    enforce_budget=enf,
                    budget_stop_utilization=bstop,
                    max_queries=mqs,
                    adaptive_concurrency=adapt,
                    adapt_latency_high_ms=ah,
                    adapt_latency_low_ms=al,
                    adapt_step=astep,
                )
                try:
                    consolidated.append(
                        {
                            "category": category,
                            "version": version,
                            "mode": r.get("metadata", {}).get("mode"),
                            "metrics": r.get("metrics", {}),
                        }
                    )
                except Exception:
                    pass

        # Save results if requested
        if args.save_results:
            output_path = runner.save_results(args.output)
            print(f"\nResults saved to: {output_path}")

        # Persist effective run settings
        try:
            out_dir = Path(args.output or "results/benchmark_runs")
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rs = {
                "timestamp": datetime.now().isoformat(),
                "preset": args.preset,
                "mode": args.mode,
                "auto_concurrency": args.auto_concurrency,
                "launch_stagger_ms": int(os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50")),
                "adaptive": {
                    "enabled": args.adaptive_concurrency,
                    "adapt_latency_high_ms": args.adapt_latency_high_ms,
                    "adapt_latency_low_ms": args.adapt_latency_low_ms,
                    "adapt_step": args.adapt_step,
                },
                "budget": {
                    "enforce_budget": args.enforce_budget,
                    "budget_stop_utilization": args.budget_stop_utilization,
                },
                "sampling": {"max_queries": args.max_queries},
                "category": args.category,
            }
            with open(out_dir / f"run_settings_{timestamp}.json", "w") as rf:
                json.dump(rs, rf, indent=2)
        except Exception as e:
            logger.debug(f"Failed to persist run settings: {e}")

        # Persist effective run settings
        try:
            out_dir = Path(args.output or "results/benchmark_runs")
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rs = {
                "timestamp": datetime.now().isoformat(),
                "preset": args.preset,
                "mode": args.mode,
                "auto_concurrency": args.auto_concurrency,
                "launch_stagger_ms": int(os.getenv("BENCHMARK_LAUNCH_STAGGER_MS", "50")),
            }
            # Add adaptive/budget/sampling knobs if present
            try:
                rs.update(
                    {
                        "adaptive": {
                            "enabled": args.adaptive_concurrency,
                            "adapt_latency_high_ms": args.adapt_latency_high_ms,
                            "adapt_latency_low_ms": args.adapt_latency_low_ms,
                            "adapt_step": args.adapt_step,
                        },
                        "budget": {
                            "enforce_budget": args.enforce_budget,
                            "budget_stop_utilization": args.budget_stop_utilization,
                        },
                        "sampling": {"max_queries": args.max_queries},
                        "category": args.category,
                    }
                )
            except Exception:
                pass
            with open(out_dir / f"run_settings_{timestamp}.json", "w") as rf:
                json.dump(rs, rf, indent=2)
        except Exception as e:
            logger.debug(f"Failed to persist run settings: {e}")

        # Export benchmark tracker metrics if available
        if hasattr(runner, "benchmark_tracker") and runner.benchmark_tracker:
            try:
                # Create tracker output directory
                tracker_output_dir = Path(args.output or "results/benchmark_runs") / "monitoring"
                tracker_output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tracker_output_file = tracker_output_dir / f"benchmark_tracker_{timestamp}.json"

                runner.benchmark_tracker.export_metrics(str(tracker_output_file))
                print(f"Benchmark tracker metrics saved to: {tracker_output_file}")

                # Export cost summary from analyzer if available
                try:
                    if hasattr(runner, "cost_analyzer") and runner.cost_analyzer:
                        cost_summary = runner.cost_analyzer.get_cost_analysis(days_back=1, force_refresh=True)
                        cost_file = tracker_output_dir / f"cost_summary_{timestamp}.json"
                        with open(cost_file, "w") as cf:
                            json.dump(cost_summary, cf, indent=2)
                        print(f"Cost summary saved to: {cost_file}")
                except Exception as e:
                    logger.debug(f"Cost summary export skipped: {e}")

                # Print regression summary if any found
                if runner.benchmark_tracker.has_regressions():
                    print("\n⚠️  REGRESSIONS DETECTED:")
                    for regression in runner.benchmark_tracker.get_regression_summary():
                        endpoint_info = (
                            f" ({regression.get('endpoint', 'unknown')})" if "endpoint" in regression else ""
                        )
                        print(f"  - {regression['type']}{endpoint_info}: {regression.get('category', 'unknown')}")
                        print(
                            f"    Baseline: {regression.get('baseline', 'N/A')}, Current: {regression.get('current', 'N/A')}"
                        )
                else:
                    print("\n✅ No regressions detected")
            except Exception as e:
                logger.warning(f"Error exporting benchmark tracker metrics: {e}")

        # Optionally write consolidated summary when running all categories
        if not args.category and (args.save_results or args.summary_output):
            try:
                out_dir = Path(args.output or "results/benchmark_runs")
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target = (
                    Path(args.summary_output)
                    if args.summary_output
                    else (out_dir / f"consolidated_summary_{timestamp}.json")
                )

                # Build summary from runner.results (list of per-category results)
                categories: List[Dict[str, Any]] = []
                for item in runner.results:
                    meta = item.get("metadata", {})
                    categories.append(
                        {
                            "category": meta.get("category"),
                            "version": meta.get("version"),
                            "mode": meta.get("mode"),
                            "metrics": item.get("metrics", {}),
                        }
                    )

                # Simple overall aggregation (cloud/self hosted averages across categories)
                def _avg(vals: List[float]) -> float:
                    return round(sum(vals) / len(vals), 3) if vals else 0.0

                cloud_scores = []
                sh_scores = []
                for c in categories:
                    m = c.get("metrics", {})
                    if c.get("mode") == "both":
                        cloud_scores.append(m.get("cloud", {}).get("average_overall_score", 0.0) or 0.0)
                        sh_scores.append(m.get("self_hosted", {}).get("average_overall_score", 0.0) or 0.0)
                    else:
                        cloud_scores.append(m.get("average_overall_score", 0.0) or 0.0)

                # Build consolidated summary JSON
                consolidated_summary = {
                    "timestamp": datetime.now().isoformat(),
                    "total_categories": len(categories),
                    "categories": categories,
                    "overall": {
                        "avg_cloud_overall_score": _avg(cloud_scores),
                        "avg_self_hosted_overall_score": _avg(sh_scores) if sh_scores else None,
                    },
                }

                # Attach gating evaluation if thresholds provided
                try:
                    min_score = getattr(args, "min_cloud_score", None)
                    max_p95_latency = getattr(args, "max_cloud_latency_ms", None)
                    score_violations: List[str] = []
                    latency_violations: List[str] = []
                    if min_score is not None or max_p95_latency is not None:
                        for c in categories:
                            m = c.get("metrics", {})
                            mode_c = c.get("mode")
                            if mode_c == "both":
                                cs = float(m.get("cloud", {}).get("average_overall_score", 0.0) or 0.0)
                                cp95 = float(m.get("cloud", {}).get("p95_latency_ms", 0.0) or 0.0)
                            else:
                                cs = float(m.get("average_overall_score", 0.0) or 0.0)
                                cp95 = float(m.get("p95_latency_ms", 0.0) or 0.0)
                            if (min_score is not None) and cs < float(min_score):
                                score_violations.append(c.get("category", "unknown"))
                            if (max_p95_latency is not None) and cp95 > float(max_p95_latency):
                                latency_violations.append(c.get("category", "unknown"))
                        consolidated_summary["gating"] = {
                            "rules": {"min_cloud_score": min_score, "max_cloud_latency_ms": max_p95_latency},
                            "pass": (not score_violations and not latency_violations),
                            "score_violations": score_violations,
                            "latency_violations": latency_violations,
                        }
                except Exception:
                    pass

                with open(target, "w") as f:
                    json.dump(consolidated_summary, f, indent=2)
                print(f"Consolidated summary saved to: {target}")

                # Also emit a Markdown report for quick review
                md_path = target.with_suffix(".md")
                try:
                    lines = [
                        f"# Consolidated Pharmaceutical Benchmark Summary ({consolidated_summary['timestamp']})",
                        "",
                        f"Total Categories: {consolidated_summary['total_categories']}",
                        "",
                    ]
                    # Add a PASS/FAIL badge near the top when gating info is present
                    try:
                        g = consolidated_summary.get("gating") or {}
                        if g:
                            status = "✅ PASS" if g.get("pass") else "❌ FAIL"
                            lines.append(f"Status: {status}")
                            lines.append("")
                    except Exception:
                        pass
                    # Table header depends on mode presence
                    lines.append(
                        "| Category | Version | Mode | Cloud Score | Cloud Latency (ms) | Cloud P95 Lat (ms) | Cloud P95 Tokens | Self Score | Self Latency (ms) |"
                    )
                    lines.append("|---|---:|:---:|---:|---:|---:|---:|---:|---:|")
                    for c in categories:
                        cat = c.get("category", "")
                        ver = c.get("version", "")
                        mode = c.get("mode", "")
                        m = c.get("metrics", {})
                        if mode == "both":
                            cs = m.get("cloud", {}).get("average_overall_score", 0)
                            cl = m.get("cloud", {}).get("average_latency_ms", 0)
                            cp95l = m.get("cloud", {}).get("p95_latency_ms", 0)
                            cp95t = m.get("cloud", {}).get("p95_tokens", 0)
                            ss = m.get("self_hosted", {}).get("average_overall_score", 0)
                            sl = m.get("self_hosted", {}).get("average_latency_ms", 0)
                        else:
                            cs = m.get("average_overall_score", 0)
                            cl = m.get("average_latency_ms", 0)
                            cp95l = m.get("p95_latency_ms", 0)
                            cp95t = m.get("p95_tokens", 0)
                            ss = ""
                            sl = ""
                        lines.append(
                            f"| {cat} | {ver} | {mode} | {cs:.3f} | {cl:.2f} | {cp95l:.2f} | {cp95t} | {ss if ss=='' else f'{ss:.3f}'} | {sl if sl=='' else f'{sl:.2f}'} |"
                        )
                    lines.append("")
                    ov = consolidated_summary.get("overall", {})
                    lines.append("## Overall Averages")
                    lines.append("")
                    lines.append(f"- Avg Cloud Overall Score: {ov.get('avg_cloud_overall_score',0)}")
                    if ov.get("avg_self_hosted_overall_score") is not None:
                        lines.append(f"- Avg Self-Hosted Overall Score: {ov.get('avg_self_hosted_overall_score',0)}")
                    # Gating Outcome
                    try:
                        g = consolidated_summary.get("gating") or {}
                        if g:
                            lines.append("")
                            lines.append("## Gating Outcome")
                            lines.append("")
                            lines.append(
                                f"- Rules: min_cloud_score={g.get('rules',{}).get('min_cloud_score')}, max_cloud_latency_ms={g.get('rules',{}).get('max_cloud_latency_ms')}"
                            )
                            lines.append(f"- Result: {'PASS' if g.get('pass') else 'FAIL'}")
                            if g.get("score_violations"):
                                lines.append(f"- Score violations: {', '.join(g.get('score_violations'))}")
                            if g.get("latency_violations"):
                                lines.append(f"- Latency violations: {', '.join(g.get('latency_violations'))}")
                    except Exception:
                        pass
                    # Top offenders
                    try:
                        # Build helper lists
                        def _get(m, path, default=0.0):
                            cur = m
                            for k in path:
                                if not isinstance(cur, dict):
                                    return default
                                cur = cur.get(k)
                            return cur if isinstance(cur, (int, float)) else default

                        # Score low = bad, Latency high = bad
                        cloud_scores = [
                            (c.get("category"), _get(c, ["metrics", "cloud", "average_overall_score"], 0.0))
                            for c in categories
                            if c.get("mode") == "both"
                        ]
                        cloud_p95 = [
                            (c.get("category"), _get(c, ["metrics", "cloud", "p95_latency_ms"], 0.0))
                            for c in categories
                            if c.get("mode") == "both"
                        ]
                        worst_scores = sorted(cloud_scores, key=lambda x: x[1])[:3]
                        worst_latency = sorted(cloud_p95, key=lambda x: x[1], reverse=True)[:3]
                        lines.append("")
                        lines.append("## Top Offenders")
                        lines.append("")
                        lines.append("- Lowest Cloud Scores:")
                        for name, val in worst_scores:
                            lines.append(f"  - {name}: {val:.3f}")
                        lines.append("- Highest Cloud P95 Latency:")
                        for name, val in worst_latency:
                            lines.append(f"  - {name}: {val:.2f} ms")
                    except Exception:
                        pass
                    with open(md_path, "w") as mf:
                        mf.write("\n".join(lines))
                    print(f"Markdown summary saved to: {md_path}")
                except Exception as e:
                    logger.warning(f"Failed to write Markdown summary: {e}")
            except Exception as e:
                logger.warning(f"Failed to write consolidated summary: {e}")

        logger.info("Benchmark execution completed successfully")
        # CI gating: regressions and thresholds
        exit_code = 0
        try:
            if (
                getattr(args, "fail_on_regressions", False)
                and hasattr(runner, "benchmark_tracker")
                and runner.benchmark_tracker
                and runner.benchmark_tracker.has_regressions()
            ):
                exit_code = max(exit_code, 2)
            # Threshold gating
            min_score = getattr(args, "min_cloud_score", None)
            max_p95_latency = getattr(args, "max_cloud_latency_ms", None)
            if min_score is not None or max_p95_latency is not None:

                def _check_run(res: Dict[str, Any]) -> bool:
                    try:
                        if res.get("metadata", {}).get("mode") == "both":
                            cloud = res.get("metrics", {}).get("cloud", {})
                            score_ok = (
                                True
                                if min_score is None
                                else (float(cloud.get("average_overall_score", 0.0) or 0.0) >= float(min_score))
                            )
                            lat_ok = (
                                True
                                if max_p95_latency is None
                                else (float(cloud.get("p95_latency_ms", 0.0) or 0.0) <= float(max_p95_latency))
                            )
                            return score_ok and lat_ok
                        else:
                            m = res.get("metrics", {})
                            score_ok = (
                                True
                                if min_score is None
                                else (float(m.get("average_overall_score", 0.0) or 0.0) >= float(min_score))
                            )
                            lat_ok = (
                                True
                                if max_p95_latency is None
                                else (float(m.get("p95_latency_ms", 0.0) or 0.0) <= float(max_p95_latency))
                            )
                            return score_ok and lat_ok
                    except Exception:
                        return False

                if args.category:
                    ok = _check_run(result)
                    if not ok:
                        exit_code = max(exit_code, 3)
                else:
                    # All categories must pass
                    for res in runner.results:
                        if not _check_run(res):
                            exit_code = max(exit_code, 3)
                            break
        except Exception as e:
            logger.debug(f"Gating checks failed: {e}")
        return exit_code

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
