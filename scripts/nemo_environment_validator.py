#!/usr/bin/env python3
"""
NeMo Retriever Environment Validation Script

Validates all prerequisites for migrating to NVIDIA NeMo Retriever:
- NVIDIA API keys and authentication
- GPU availability and CUDA support
- Python environment and dependencies
- Network connectivity to NeMo NIMs
- Existing system compatibility

Usage: python nemo_environment_validator.py
"""

import os
import sys
import subprocess
import platform
import importlib
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    """Stores validation results for environment checks."""
    component: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: Optional[Dict[str, Any]] = None
    fix_suggestion: Optional[str] = None

class NeMoEnvironmentValidator:
    """Comprehensive environment validator for NeMo Retriever migration."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")

        # NeMo NIM endpoints (cloud-hosted by default)
        self.nemo_endpoints = {
            "embedding": "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings",
            "reranking": "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
            "extraction": "https://ai.api.nvidia.com/v1/retrieval/nvidia/extraction"
        }

        # Required dependencies for NeMo integration
        self.required_packages = {
            "langchain": ">=0.3,<0.4",
            "langchain-nvidia-ai-endpoints": ">=0.3.0,<0.4.0",
            "langchain-community": ">=0.3,<0.4",
            "langchain-core": ">=0.3,<0.4",
            "faiss-cpu": ">=1.7.4,<1.8.0",
            "numpy": ">=1.21.0,<2.0.0",
            "requests": ">=2.31.0,<3.0.0",
            "tqdm": ">=4.65.0,<5.0.0",
            "pydantic": ">=2.0.0"
        }

        # Optional but recommended packages
        self.optional_packages = {
            "cupy": "GPU acceleration for cuVS",
            "cudf": "GPU DataFrame operations",
            "cuvs": "NVIDIA cuVS for vector search"
        }

    def validate_python_environment(self) -> None:
        """Validate Python version and virtual environment setup."""
        python_version = sys.version_info

        if python_version < (3, 8):
            self.results.append(ValidationResult(
                component="Python Version",
                status="FAIL",
                message=f"Python {python_version.major}.{python_version.minor} detected, need 3.8+",
                fix_suggestion="Upgrade to Python 3.8 or higher"
            ))
        elif python_version >= (3, 12):
            self.results.append(ValidationResult(
                component="Python Version",
                status="WARN",
                message=f"Python {python_version.major}.{python_version.minor} detected, some packages may have compatibility issues",
                fix_suggestion="Consider using Python 3.9-3.11 for best compatibility"
            ))
        else:
            self.results.append(ValidationResult(
                component="Python Version",
                status="PASS",
                message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro} is compatible"
            ))

        # Check if in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

        if in_venv:
            venv_path = os.environ.get('VIRTUAL_ENV', sys.prefix)
            self.results.append(ValidationResult(
                component="Virtual Environment",
                status="PASS",
                message=f"Virtual environment active: {venv_path}"
            ))
        else:
            self.results.append(ValidationResult(
                component="Virtual Environment",
                status="WARN",
                message="No virtual environment detected",
                fix_suggestion="Consider using a virtual environment to avoid dependency conflicts"
            ))

    def validate_nvidia_api_access(self) -> None:
        """Validate NVIDIA API key and endpoint connectivity."""
        if not self.nvidia_api_key:
            self.results.append(ValidationResult(
                component="NVIDIA API Key",
                status="FAIL",
                message="NVIDIA_API_KEY environment variable not found",
                fix_suggestion="Set NVIDIA_API_KEY environment variable with your API key"
            ))
            return

        # Test API key validity with a simple embedding request
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }

        test_payload = {
            "input": ["test connection"],
            "model": "nvidia/nv-embedqa-e5-v5"
        }

        try:
            response = requests.post(
                self.nemo_endpoints["embedding"],
                headers=headers,
                json=test_payload,
                timeout=10
            )

            if response.status_code == 200:
                self.results.append(ValidationResult(
                    component="NVIDIA API Access",
                    status="PASS",
                    message="NVIDIA API key is valid and endpoints are accessible",
                    details={"response_time_ms": response.elapsed.total_seconds() * 1000}
                ))
            elif response.status_code == 401:
                self.results.append(ValidationResult(
                    component="NVIDIA API Access",
                    status="FAIL",
                    message="NVIDIA API key is invalid or expired",
                    fix_suggestion="Check your API key or generate a new one from https://build.nvidia.com (NVIDIA Build platform)"
                ))
            elif response.status_code == 429:
                self.results.append(ValidationResult(
                    component="NVIDIA API Access",
                    status="WARN",
                    message="NVIDIA API rate limit reached",
                    fix_suggestion="Wait and retry, or check your API quota"
                ))
            else:
                self.results.append(ValidationResult(
                    component="NVIDIA API Access",
                    status="FAIL",
                    message=f"API request failed with status {response.status_code}: {response.text}",
                    fix_suggestion="Check API endpoint availability and your network connection"
                ))

        except requests.exceptions.RequestException as e:
            self.results.append(ValidationResult(
                component="NVIDIA API Access",
                status="FAIL",
                message=f"Failed to connect to NVIDIA API: {str(e)}",
                fix_suggestion="Check network connectivity and firewall settings"
            ))

    def validate_gpu_environment(self) -> None:
        """Validate GPU availability and CUDA support."""
        try:
            # Check if NVIDIA GPU is available
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_info.append({
                            "name": parts[0],
                            "memory_mb": int(parts[1]),
                            "driver_version": parts[2]
                        })

                total_memory = sum(gpu["memory_mb"] for gpu in gpu_info)

                if total_memory >= 8000:  # 8GB minimum recommended
                    self.results.append(ValidationResult(
                        component="GPU Hardware",
                        status="PASS",
                        message=f"Found {len(gpu_info)} GPU(s) with {total_memory}MB total memory",
                        details={"gpus": gpu_info}
                    ))
                else:
                    self.results.append(ValidationResult(
                        component="GPU Hardware",
                        status="WARN",
                        message=f"GPU memory ({total_memory}MB) may be insufficient for large models",
                        details={"gpus": gpu_info},
                        fix_suggestion="Consider using smaller batch sizes or cloud GPUs"
                    ))

            else:
                self.results.append(ValidationResult(
                    component="GPU Hardware",
                    status="WARN",
                    message="No NVIDIA GPU detected or nvidia-smi not available",
                    fix_suggestion="GPU acceleration will not be available, will fall back to CPU"
                ))

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.results.append(ValidationResult(
                component="GPU Hardware",
                status="WARN",
                message=f"Could not check GPU status: {str(e)}",
                fix_suggestion="Install NVIDIA drivers and nvidia-smi if GPU acceleration is needed"
            ))

    def validate_existing_dependencies(self) -> None:
        """Validate current Python dependencies and compatibility."""
        missing_packages = []
        outdated_packages = []

        for package, version_spec in self.required_packages.items():
            try:
                module = importlib.import_module(package.replace('-', '_'))
                current_version = getattr(module, '__version__', 'unknown')

                # Simple version check (production code should use packaging.version)
                if current_version != 'unknown':
                    self.results.append(ValidationResult(
                        component=f"Package: {package}",
                        status="PASS",
                        message=f"Found {package} version {current_version}",
                        details={"required": version_spec, "installed": current_version}
                    ))
                else:
                    self.results.append(ValidationResult(
                        component=f"Package: {package}",
                        status="WARN",
                        message=f"Package {package} found but version unknown",
                        fix_suggestion="Reinstall package to ensure proper version"
                    ))

            except ImportError:
                missing_packages.append(package)
                self.results.append(ValidationResult(
                    component=f"Package: {package}",
                    status="FAIL",
                    message=f"Required package {package} not found",
                    fix_suggestion=f"Install with: pip install '{package}{version_spec}'"
                ))

        # Check optional packages
        for package, description in self.optional_packages.items():
            try:
                importlib.import_module(package.replace('-', '_'))
                self.results.append(ValidationResult(
                    component=f"Optional: {package}",
                    status="PASS",
                    message=f"Optional package {package} available ({description})"
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    component=f"Optional: {package}",
                    status="WARN",
                    message=f"Optional package {package} not found ({description})",
                    fix_suggestion=f"Install for enhanced performance: pip install {package}"
                ))

    def validate_existing_system(self) -> None:
        """Validate existing RAG system components and compatibility."""
        # Check for existing configuration files
        config_files = [
            ".env",
            "requirements.txt",
            "src/enhanced_config.py",
            "src/nvidia_embeddings.py",
            "src/vector_database.py"
        ]

        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                self.results.append(ValidationResult(
                    component=f"Config: {config_file}",
                    status="PASS",
                    message=f"Found existing {config_file}",
                    details={"size_bytes": file_path.stat().st_size}
                ))
            else:
                status = "WARN" if config_file in [".env"] else "FAIL"
                self.results.append(ValidationResult(
                    component=f"Config: {config_file}",
                    status=status,
                    message=f"Missing {config_file}",
                    fix_suggestion=f"Create {config_file} or ensure correct path"
                ))

        # Check for vector database files
        vector_db_paths = ["./vector_db", "Data/vector_db", "vector_store"]
        vector_db_found = False

        for db_path in vector_db_paths:
            path = Path(db_path)
            if path.exists() and any(path.iterdir()):
                vector_db_found = True
                self.results.append(ValidationResult(
                    component="Vector Database",
                    status="PASS",
                    message=f"Found existing vector database at {db_path}",
                    details={"path": str(path.absolute())}
                ))
                break

        if not vector_db_found:
            self.results.append(ValidationResult(
                component="Vector Database",
                status="WARN",
                message="No existing vector database found",
                fix_suggestion="Will need to rebuild vector database during migration"
            ))

    def validate_network_connectivity(self) -> None:
        """Validate network connectivity to required services."""
        test_endpoints = {
            "NVIDIA API": "https://ai.api.nvidia.com/health",
            "Microsoft Learn (MCP)": "https://learn.microsoft.com",
            "PubMed API": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        }

        for service, endpoint in test_endpoints.items():
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code < 400:
                    self.results.append(ValidationResult(
                        component=f"Network: {service}",
                        status="PASS",
                        message=f"Can reach {service}",
                        details={"response_time_ms": response.elapsed.total_seconds() * 1000}
                    ))
                else:
                    self.results.append(ValidationResult(
                        component=f"Network: {service}",
                        status="WARN",
                        message=f"{service} returned status {response.status_code}",
                        fix_suggestion="Check service availability or network configuration"
                    ))
            except requests.exceptions.RequestException as e:
                self.results.append(ValidationResult(
                    component=f"Network: {service}",
                    status="FAIL",
                    message=f"Cannot reach {service}: {str(e)}",
                    fix_suggestion="Check network connectivity and firewall settings"
                ))

    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive results."""
        print("🔍 Validating NeMo Retriever Environment Prerequisites...")
        print("=" * 60)

        validation_steps = [
            ("Python Environment", self.validate_python_environment),
            ("NVIDIA API Access", self.validate_nvidia_api_access),
            ("GPU Environment", self.validate_gpu_environment),
            ("Dependencies", self.validate_existing_dependencies),
            ("Existing System", self.validate_existing_system),
            ("Network Connectivity", self.validate_network_connectivity)
        ]

        for step_name, validation_func in validation_steps:
            print(f"📋 Validating {step_name}...")
            try:
                validation_func()
                print(f"✅ {step_name} validation completed")
            except Exception as e:
                self.results.append(ValidationResult(
                    component=step_name,
                    status="FAIL",
                    message=f"Validation failed with error: {str(e)}",
                    fix_suggestion="Check logs and retry validation"
                ))
                print(f"❌ {step_name} validation failed: {e}")

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        pass_count = sum(1 for r in self.results if r.status == "PASS")
        warn_count = sum(1 for r in self.results if r.status == "WARN")
        fail_count = sum(1 for r in self.results if r.status == "FAIL")

        # Determine overall status
        if fail_count > 0:
            overall_status = "BLOCKED"
        elif warn_count > 0:
            overall_status = "CAUTION"
        else:
            overall_status = "READY"

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": len(self.results),
                "passed": pass_count,
                "warnings": warn_count,
                "failed": fail_count
            },
            "results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "fix_suggestion": r.fix_suggestion
                }
                for r in self.results
            ],
            "recommendations": self.generate_recommendations()
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Critical failures
        critical_failures = [r for r in self.results if r.status == "FAIL"]
        if critical_failures:
            recommendations.append("🚨 CRITICAL: Address all failed validations before proceeding with migration")
            for failure in critical_failures[:3]:  # Top 3 critical issues
                if failure.fix_suggestion:
                    recommendations.append(f"   • {failure.component}: {failure.fix_suggestion}")

        # Performance optimizations
        gpu_results = [r for r in self.results if "GPU" in r.component]
        if not any(r.status == "PASS" for r in gpu_results):
            recommendations.append("💡 Consider cloud GPU instances for better performance")

        # Environment optimization
        if any("Virtual Environment" in r.component and r.status == "WARN" for r in self.results):
            recommendations.append("🔧 Create a dedicated virtual environment for NeMo migration")

        # Network optimization
        network_failures = [r for r in self.results if "Network" in r.component and r.status != "PASS"]
        if network_failures:
            recommendations.append("🌐 Verify network connectivity for all external services")

        if not recommendations:
            recommendations.append("✅ Environment looks ready for NeMo Retriever migration!")

        return recommendations

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted validation report to console."""
        print("\n" + "=" * 60)
        print("📊 NEMO RETRIEVER ENVIRONMENT VALIDATION REPORT")
        print("=" * 60)

        # Overall status with color coding
        status_emoji = {
            "READY": "🟢",
            "CAUTION": "🟡",
            "BLOCKED": "🔴"
        }

        print(f"\n{status_emoji.get(report['overall_status'], '⚪')} OVERALL STATUS: {report['overall_status']}")

        # Summary
        summary = report["summary"]
        print(f"\n📈 SUMMARY:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ⚠️  Warnings: {summary['warnings']}")
        print(f"   ❌ Failed: {summary['failed']}")

        # Detailed results by status
        for status in ["FAIL", "WARN", "PASS"]:
            status_results = [r for r in report["results"] if r["status"] == status]
            if status_results:
                status_emoji_map = {"FAIL": "❌", "WARN": "⚠️", "PASS": "✅"}
                print(f"\n{status_emoji_map[status]} {status} ({len(status_results)}):")

                for result in status_results:
                    print(f"   • {result['component']}: {result['message']}")
                    if result.get('fix_suggestion') and status != "PASS":
                        print(f"     💡 Fix: {result['fix_suggestion']}")

        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   {rec}")

        print("\n" + "=" * 60)

def main():
    """Main validation entry point."""
    validator = NeMoEnvironmentValidator()
    report = validator.run_all_validations()
    validator.print_report(report)

    # Save report to file
    report_file = f"nemo_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    exit_code = 0 if report["overall_status"] == "READY" else 1
    return exit_code

if __name__ == "__main__":
    sys.exit(main())