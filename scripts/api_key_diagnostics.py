"""
NVIDIA API Key Comprehensive Diagnostics

Performs deep analysis of API key permissions, service availability,
and provides actionable recommendations for resolving access issues.

This script helps identify:
- API key validity and format
- Service-specific permissions
- Regional restrictions
- Account status issues
- Optimal configuration paths

Usage:
  python scripts/api_key_diagnostics.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

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

def check_api_key_format(api_key: str) -> Dict[str, Any]:
    """Analyze API key format and structure."""
    analysis = {
        "format_valid": False,
        "key_type": "unknown",
        "estimated_service": "unknown",
        "issues": []
    }

    if not api_key:
        analysis["issues"].append("API key is empty")
        return analysis

    if api_key.startswith("nvapi-"):
        analysis["format_valid"] = True
        analysis["key_type"] = "NVIDIA API"
        analysis["estimated_service"] = "NVIDIA Build/AI Platform"
    elif api_key.startswith("sk-"):
        analysis["issues"].append("This appears to be an OpenAI API key, not NVIDIA")
    elif len(api_key) < 20:
        analysis["issues"].append("API key appears too short")
    elif len(api_key) > 200:
        analysis["issues"].append("API key appears too long")
    else:
        analysis["issues"].append("Unknown API key format")

    return analysis

def test_basic_connectivity() -> Dict[str, Any]:
    """Test basic connectivity to NVIDIA services."""
    endpoints = [
        "https://api.nvidia.com",
        "https://ai.api.nvidia.com",
        "https://integrate.api.nvidia.com",
        "https://build.nvidia.com"
    ]

    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            results[endpoint] = {
                "reachable": True,
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
        except Exception as e:
            results[endpoint] = {
                "reachable": False,
                "error": str(e)
            }

    return results

def test_authentication_endpoints(api_key: str) -> Dict[str, Any]:
    """Test authentication across different NVIDIA endpoints."""

    test_configs = [
        {
            "name": "NVIDIA Build Platform",
            "url": "https://integrate.api.nvidia.com/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"},
            "expected_success_codes": [200, 401, 403]  # 401/403 tell us about auth, not connectivity
        },
        {
            "name": "NVIDIA AI Platform",
            "url": "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings",
            "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            "method": "POST",
            "data": {"input": ["test"], "model": "nvidia/nv-embedqa-e5-v5"},
            "expected_success_codes": [200, 400, 401, 403, 404]
        }
    ]

    results = {}
    for config in test_configs:
        try:
            method = config.get("method", "GET")
            if method == "POST":
                response = requests.post(
                    config["url"],
                    headers=config["headers"],
                    json=config.get("data"),
                    timeout=15
                )
            else:
                response = requests.get(config["url"], headers=config["headers"], timeout=15)

            results[config["name"]] = {
                "status_code": response.status_code,
                "response_text": response.text[:500],  # First 500 chars
                "auth_recognized": response.status_code in [200, 400, 403],  # vs 401 = not recognized
                "likely_valid_endpoint": response.status_code != 404
            }
        except Exception as e:
            results[config["name"]] = {
                "error": str(e),
                "auth_recognized": False,
                "likely_valid_endpoint": False
            }

    return results

def analyze_error_patterns(auth_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze error patterns to determine likely issues."""
    patterns = {
        "invalid_key": False,
        "insufficient_permissions": False,
        "service_unavailable": False,
        "account_inactive": False,
        "regional_restriction": False,
        "billing_required": False
    }

    recommendations = []

    # Check for common error patterns
    for service, result in auth_results.items():
        if "error" in result:
            patterns["service_unavailable"] = True
            continue

        status_code = result.get("status_code")
        response_text = result.get("response_text", "").lower()

        if status_code == 401:
            patterns["invalid_key"] = True
        elif status_code == 403:
            patterns["insufficient_permissions"] = True
            if "billing" in response_text or "payment" in response_text:
                patterns["billing_required"] = True
            elif "region" in response_text or "country" in response_text:
                patterns["regional_restriction"] = True
            elif "account" in response_text or "inactive" in response_text:
                patterns["account_inactive"] = True

    # Generate recommendations based on patterns
    if patterns["invalid_key"]:
        recommendations.append("🔑 API key appears invalid - regenerate from NVIDIA platform")
    if patterns["insufficient_permissions"]:
        recommendations.append("🔒 API key lacks permissions - check account tier/services")
    if patterns["billing_required"]:
        recommendations.append("💳 Billing setup may be required for service access")
    if patterns["account_inactive"]:
        recommendations.append("📧 Account may be inactive - contact NVIDIA support")
    if patterns["regional_restriction"]:
        recommendations.append("🌍 Regional restrictions may apply - check service availability")
    if patterns["service_unavailable"]:
        recommendations.append("🔧 Service connectivity issues - check internet/firewall")

    return {
        "patterns": patterns,
        "recommendations": recommendations
    }

def get_account_info_suggestions() -> List[str]:
    """Provide suggestions for getting account information."""
    return [
        "🌐 Visit build.nvidia.com to check Build platform access",
        "🔍 Check developer.nvidia.com for API documentation",
        "📧 Contact NVIDIA support with your API key for account status",
        "💡 Try creating a new API key if current one is old",
        "📋 Verify account email confirmation and setup completion"
    ]

def main():
    print("NVIDIA API Key Comprehensive Diagnostics")
    print("=" * 60)

    # Get API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        print("\nPlease set your NVIDIA API key in .env file:")
        print("NVIDIA_API_KEY=your_key_here")
        return 1

    print(f"🔑 Analyzing API Key: {api_key[:15]}...")

    # 1. Check API key format
    print("\n1️⃣  API KEY FORMAT ANALYSIS")
    print("-" * 40)
    format_analysis = check_api_key_format(api_key)

    if format_analysis["format_valid"]:
        print(f"✅ Format: Valid {format_analysis['key_type']}")
        print(f"🎯 Service: {format_analysis['estimated_service']}")
    else:
        print("❌ Format: Invalid or unrecognized")
        for issue in format_analysis["issues"]:
            print(f"   ⚠️  {issue}")

    # 2. Test basic connectivity
    print("\n2️⃣  CONNECTIVITY TEST")
    print("-" * 40)
    connectivity = test_basic_connectivity()

    reachable_count = sum(1 for r in connectivity.values() if r.get("reachable", False))
    print(f"📡 Reachable endpoints: {reachable_count}/{len(connectivity)}")

    for endpoint, result in connectivity.items():
        if result.get("reachable"):
            print(f"   ✅ {endpoint} ({result['status_code']}, {result['response_time_ms']:.0f}ms)")
        else:
            print(f"   ❌ {endpoint}: {result.get('error', 'Failed')}")

    # 3. Test authentication
    print("\n3️⃣  AUTHENTICATION TEST")
    print("-" * 40)
    auth_results = test_authentication_endpoints(api_key)

    for service, result in auth_results.items():
        if "error" in result:
            print(f"❌ {service}: Connection failed - {result['error']}")
        else:
            status = result["status_code"]
            auth_status = "🟢 Recognized" if result["auth_recognized"] else "🔴 Not recognized"
            print(f"🔍 {service}: HTTP {status} - {auth_status}")

            # Show relevant response snippet
            if result.get("response_text"):
                snippet = result["response_text"][:100].replace('\n', ' ')
                print(f"   💬 Response: {snippet}...")

    # 4. Analyze error patterns
    print("\n4️⃣  ERROR PATTERN ANALYSIS")
    print("-" * 40)
    error_analysis = analyze_error_patterns(auth_results)

    if any(error_analysis["patterns"].values()):
        print("🔍 Detected issues:")
        for pattern, detected in error_analysis["patterns"].items():
            if detected:
                print(f"   ⚠️  {pattern.replace('_', ' ').title()}")
    else:
        print("✅ No clear error patterns detected")

    # 5. Recommendations
    print("\n5️⃣  RECOMMENDATIONS")
    print("-" * 40)

    if error_analysis["recommendations"]:
        print("🎯 Specific recommendations:")
        for rec in error_analysis["recommendations"]:
            print(f"   {rec}")

    print("\n💡 General suggestions:")
    for suggestion in get_account_info_suggestions():
        print(f"   {suggestion}")

    # 6. Next steps
    print("\n6️⃣  NEXT STEPS")
    print("-" * 40)

    if format_analysis["format_valid"] and reachable_count > 0:
        print("🔧 Your setup appears technically correct. Try:")
        print("   1. Contact NVIDIA support with diagnostic output")
        print("   2. Verify account status and billing setup")
        print("   3. Request access to specific services if needed")
        print("   4. Consider generating a new API key")
    else:
        print("🚨 Fundamental issues detected. Priority actions:")
        print("   1. Fix API key format issues")
        print("   2. Check internet connectivity")
        print("   3. Verify NVIDIA account setup")

    print(f"\n{'=' * 60}")
    print("Diagnostics Complete")
    print("\n💬 For support, share this output with NVIDIA support")

    return 0

if __name__ == "__main__":
    sys.exit(main())