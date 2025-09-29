"""
NVIDIA Model Validation and Access Diagnostic Utility

A comprehensive tool to validate API access across different NVIDIA endpoints
and provide detailed diagnostics about what models are accessible with your API key.

Features:
- Tests multiple NVIDIA endpoint formats
- Validates specific model access
- Provides detailed error analysis
- Suggests optimal configuration
- Generates access reports

Usage:
  python scripts/model_validation_utility.py
  python scripts/model_validation_utility.py --verbose
  python scripts/model_validation_utility.py --test-all-endpoints

Prerequisites:
  - .env with a valid NVIDIA_API_KEY
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import requests

# Ensure local src is importable
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

@dataclass
class EndpointTest:
    """Test configuration for an endpoint."""
    name: str
    base_url: str
    endpoint_path: str
    method: str = "POST"
    headers_template: Dict[str, str] = None
    test_payload: Dict[str, Any] = None
    expected_auth_header: str = "Bearer"

@dataclass
class ModelTest:
    """Test configuration for a specific model."""
    name: str
    full_name: str
    endpoint_type: str  # "embedding" or "chat"
    test_input: Any
    expected_response_fields: List[str]
    model_type: str  # "embedding", "llm", "reranking"

@dataclass
class TestResult:
    """Result of an endpoint/model test."""
    success: bool
    status_code: Optional[int]
    response_time_ms: Optional[int]
    error_message: Optional[str]
    response_data: Optional[Dict[str, Any]]
    endpoint_url: str
    model_name: Optional[str] = None

class NVIDIAModelValidator:
    """Comprehensive NVIDIA model access validator."""

    # Known NVIDIA endpoint configurations
    ENDPOINTS = [
        EndpointTest(
            name="NVIDIA Build (integrate.api.nvidia.com)",
            base_url="https://integrate.api.nvidia.com/v1",
            endpoint_path="/embeddings",
            headers_template={"Content-Type": "application/json", "accept": "application/json"},
            test_payload={
                "model": "nvidia/nv-embed-v1",
                "input": ["Test pharmaceutical research query"],
                "input_type": "query",
                "encoding_format": "float"
            }
        ),
        EndpointTest(
            name="NVIDIA AI (ai.api.nvidia.com)",
            base_url="https://ai.api.nvidia.com/v1",
            endpoint_path="/retrieval/nvidia/embeddings",
            headers_template={"Content-Type": "application/json"},
            test_payload={
                "input": ["Test pharmaceutical research query"],
                "model": "nvidia/nv-embedqa-e5-v5",
                "encoding_format": "float"
            }
        ),
        EndpointTest(
            name="NVIDIA API (api.nvidia.com)",
            base_url="https://api.nvidia.com/v1",
            endpoint_path="/embeddings",
            headers_template={"Content-Type": "application/json"},
            test_payload={
                "model": "nvidia/nv-embed-v1",
                "input": ["Test pharmaceutical research query"]
            }
        ),
    ]

    # Models to test for access
    MODELS_TO_TEST = [
        ModelTest(
            name="nv-embed-v1",
            full_name="nvidia/nv-embed-v1",
            endpoint_type="embedding",
            model_type="embedding",
            test_input=["Test pharmaceutical research applications"],
            expected_response_fields=["data", "usage"]
        ),
        ModelTest(
            name="nv-embedqa-e5-v5",
            full_name="nvidia/nv-embedqa-e5-v5",
            endpoint_type="embedding",
            model_type="embedding",
            test_input=["Test pharmaceutical research applications"],
            expected_response_fields=["data", "usage"]
        ),
        ModelTest(
            name="llama-3.1-8b-instruct",
            full_name="meta/llama-3.1-8b-instruct",
            endpoint_type="chat",
            model_type="llm",
            test_input=[{"role": "user", "content": "What are drug interactions?"}],
            expected_response_fields=["choices", "usage"]
        ),
    ]

    def __init__(self, api_key: str, verbose: bool = False):
        self.api_key = api_key
        self.verbose = verbose
        self.results = []

    def test_endpoint_access(self, endpoint: EndpointTest) -> TestResult:
        """Test basic access to an endpoint."""
        if self.verbose:
            print(f"  Testing endpoint: {endpoint.name}")

        url = f"{endpoint.base_url}{endpoint.endpoint_path}"
        headers = {
            "Authorization": f"{endpoint.expected_auth_header} {self.api_key}",
            **endpoint.headers_template
        }

        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=endpoint.test_payload, timeout=30)
            end_time = time.time()

            return TestResult(
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time_ms=int((end_time - start_time) * 1000),
                error_message=None if response.status_code == 200 else response.text,
                response_data=response.json() if response.status_code == 200 else None,
                endpoint_url=url
            )
        except Exception as e:
            return TestResult(
                success=False,
                status_code=None,
                response_time_ms=None,
                error_message=str(e),
                response_data=None,
                endpoint_url=url
            )

    def test_model_on_endpoint(self, model: ModelTest, endpoint_base: str, endpoint_path: str) -> TestResult:
        """Test a specific model on a specific endpoint."""
        if self.verbose:
            print(f"    Testing {model.name} on {endpoint_base}")

        url = f"{endpoint_base}{endpoint_path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        if model.endpoint_type == "embedding":
            payload = {
                "model": model.full_name,
                "input": model.test_input,
                "input_type": "query",
                "encoding_format": "float"
            }
        else:  # chat
            payload = {
                "model": model.full_name,
                "messages": model.test_input,
                "max_tokens": 50,
                "temperature": 0.1
            }

        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            end_time = time.time()

            return TestResult(
                success=response.status_code == 200,
                status_code=response.status_code,
                response_time_ms=int((end_time - start_time) * 1000),
                error_message=None if response.status_code == 200 else response.text,
                response_data=response.json() if response.status_code == 200 else None,
                endpoint_url=url,
                model_name=model.full_name
            )
        except Exception as e:
            return TestResult(
                success=False,
                status_code=None,
                response_time_ms=None,
                error_message=str(e),
                response_data=None,
                endpoint_url=url,
                model_name=model.full_name
            )

    def discover_working_endpoints(self) -> List[Tuple[str, TestResult]]:
        """Discover which endpoints are accessible."""
        working_endpoints = []

        print("🔍 Discovering accessible endpoints...")
        for endpoint in self.ENDPOINTS:
            result = self.test_endpoint_access(endpoint)
            if result.success:
                working_endpoints.append((endpoint.name, result))
                print(f"  ✅ {endpoint.name}: ACCESSIBLE")
            else:
                print(f"  ❌ {endpoint.name}: {result.status_code} - {result.error_message[:100] if result.error_message else 'Connection failed'}")

        return working_endpoints

    def test_models_on_known_endpoints(self) -> Dict[str, List[TestResult]]:
        """Test specific models on known working endpoint patterns."""
        model_results = {}

        print("\n🧪 Testing model access...")

        # Test embedding models
        embedding_endpoints = [
            ("NVIDIA Build", "https://integrate.api.nvidia.com/v1", "/embeddings"),
            ("NVIDIA AI", "https://ai.api.nvidia.com/v1", "/retrieval/nvidia/embeddings"),
        ]

        chat_endpoints = [
            ("NVIDIA Build", "https://integrate.api.nvidia.com/v1", "/chat/completions"),
            ("NVIDIA AI", "https://ai.api.nvidia.com/v1", "/chat/completions"),
        ]

        for model in self.MODELS_TO_TEST:
            model_results[model.name] = []
            print(f"\n  Testing {model.name} ({model.full_name}):")

            if model.endpoint_type == "embedding":
                endpoints_to_test = embedding_endpoints
            else:
                endpoints_to_test = chat_endpoints

            for endpoint_name, base_url, path in endpoints_to_test:
                result = self.test_model_on_endpoint(model, base_url, path)
                model_results[model.name].append(result)

                if result.success:
                    print(f"    ✅ {endpoint_name}: ACCESSIBLE ({result.response_time_ms}ms)")
                else:
                    print(f"    ❌ {endpoint_name}: {result.status_code} - {result.error_message[:50] if result.error_message else 'Failed'}")

        return model_results

    def analyze_api_key_tier(self, model_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Analyze what tier/type of API key this appears to be."""
        analysis = {
            "accessible_models": [],
            "accessible_endpoints": set(),
            "failed_models": [],
            "api_key_type": "unknown",
            "recommendations": []
        }

        for model_name, results in model_results.items():
            has_access = any(r.success for r in results)
            if has_access:
                analysis["accessible_models"].append(model_name)
                for result in results:
                    if result.success:
                        domain = result.endpoint_url.split('/')[2]  # Extract domain
                        analysis["accessible_endpoints"].add(domain)
            else:
                analysis["failed_models"].append(model_name)

        # Determine API key type
        if "integrate.api.nvidia.com" in analysis["accessible_endpoints"]:
            analysis["api_key_type"] = "NVIDIA Build Platform"
        elif "ai.api.nvidia.com" in analysis["accessible_endpoints"]:
            analysis["api_key_type"] = "NVIDIA AI/NeMo Services"
        elif not analysis["accessible_models"]:
            analysis["api_key_type"] = "Invalid or Restricted"

        # Generate recommendations
        if analysis["accessible_models"]:
            analysis["recommendations"].append("✅ Your API key provides access to some NVIDIA models")
            if "nv-embed-v1" in analysis["accessible_models"]:
                analysis["recommendations"].append("💡 You can use nvidia/nv-embed-v1 as a fallback embedding model")
            if "llama-3.1-8b-instruct" in analysis["accessible_models"]:
                analysis["recommendations"].append("💡 You can use meta/llama-3.1-8b-instruct for LLM tasks")
        else:
            analysis["recommendations"].append("❌ No model access detected")
            analysis["recommendations"].append("🔧 Check API key validity and permissions")
            analysis["recommendations"].append("📧 Contact NVIDIA support for access issues")

        return analysis

    def generate_configuration_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Generate .env configuration recommendations."""
        config_recommendations = []

        if analysis["accessible_models"]:
            config_recommendations.append("# Add these to your .env for accessible models:")
            config_recommendations.append("")

            if "nv-embed-v1" in analysis["accessible_models"]:
                config_recommendations.append("# Enable NVIDIA Build fallback")
                config_recommendations.append("ENABLE_NVIDIA_BUILD_FALLBACK=true")
                config_recommendations.append("NVIDIA_BUILD_EMBEDDING_MODEL=nvidia/nv-embed-v1")

            if "llama-3.1-8b-instruct" in analysis["accessible_models"]:
                config_recommendations.append("NVIDIA_BUILD_LLM_MODEL=meta/llama-3.1-8b-instruct")

            config_recommendations.append("")
            config_recommendations.append("# Your current NeMo models remain optimal for pharmaceutical research")

        return "\n".join(config_recommendations)

def main():
    parser = argparse.ArgumentParser(description="NVIDIA Model Validation Utility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--test-all-endpoints", action="store_true", help="Test all known endpoints")
    args = parser.parse_args()

    print("NVIDIA Model Validation and Access Diagnostic Utility")
    print("=" * 60)

    # Get API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        return 1

    print(f"API Key: {api_key[:10]}... (Type: {'nvapi-' if api_key.startswith('nvapi-') else 'Unknown'})")

    # Initialize validator
    validator = NVIDIAModelValidator(api_key, verbose=args.verbose)

    # Run tests
    if args.test_all_endpoints:
        working_endpoints = validator.discover_working_endpoints()

    model_results = validator.test_models_on_known_endpoints()

    # Analyze results
    analysis = validator.analyze_api_key_tier(model_results)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"API Key Type: {analysis['api_key_type']}")
    print(f"Accessible Models: {', '.join(analysis['accessible_models']) if analysis['accessible_models'] else 'None'}")
    print(f"Working Endpoints: {', '.join(analysis['accessible_endpoints']) if analysis['accessible_endpoints'] else 'None'}")

    print("\n📋 RECOMMENDATIONS:")
    for rec in analysis["recommendations"]:
        print(f"  {rec}")

    # Configuration recommendations
    config_rec = validator.generate_configuration_recommendations(analysis)
    if config_rec:
        print(f"\n⚙️  CONFIGURATION RECOMMENDATIONS:")
        print(config_rec)

    print(f"\n{'=' * 60}")
    print("Validation Complete")

    return 0 if analysis["accessible_models"] else 1

if __name__ == "__main__":
    sys.exit(main())