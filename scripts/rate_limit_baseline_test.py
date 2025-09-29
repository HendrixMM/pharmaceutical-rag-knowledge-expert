"""
NVIDIA Build API Rate Limiting Baseline Test

Establishes baseline request patterns and rate limiting behavior
to optimize free tier usage and prepare for production deployment.

Even with 403 errors, this test measures:
- Response times for rate limit planning
- Error consistency for access tier validation
- Request throttling behavior
- Optimal request spacing

Usage:
  python scripts/rate_limit_baseline_test.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure local src is importable
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:
    from src.nvidia_build_client import NVIDBuildClient
except ImportError as e:
    print(f"❌ Error importing NVIDIA Build client: {e}")
    sys.exit(1)

def test_rate_limiting_baseline() -> Dict[str, Any]:
    """Test rate limiting behavior with current access level."""
    print("🔄 Testing NVIDIA Build API rate limiting baseline...")

    client = NVIDBuildClient(enable_credit_monitoring=False)  # Disable for testing

    results = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "response_times_ms": [],
        "error_patterns": {},
        "rate_limit_detected": False,
        "optimal_request_spacing_ms": None
    }

    # Test with conservative spacing (1 request per second)
    test_requests = 5  # Conservative for baseline
    print(f"📊 Testing {test_requests} requests with 1-second spacing...")

    for i in range(test_requests):
        print(f"  Request {i+1}/{test_requests}...", end=" ")

        start_time = time.time()
        try:
            # Use lightweight embedding test
            response = client.create_embeddings(
                texts=["Rate limit test query"],
                model="nvidia/nv-embed-v1"
            )
            results["successful_requests"] += 1
            print("✅ SUCCESS")

        except Exception as e:
            results["failed_requests"] += 1
            error_str = str(e)

            # Track error patterns
            if "403" in error_str:
                results["error_patterns"]["403_forbidden"] = results["error_patterns"].get("403_forbidden", 0) + 1
                print("❌ 403 Forbidden")
            elif "429" in error_str:
                results["error_patterns"]["429_rate_limit"] = results["error_patterns"].get("429_rate_limit", 0) + 1
                results["rate_limit_detected"] = True
                print("⚠️ Rate Limited")
            else:
                results["error_patterns"]["other"] = results["error_patterns"].get("other", 0) + 1
                print(f"❌ Error: {error_str[:50]}...")

        # Track response time
        response_time_ms = int((time.time() - start_time) * 1000)
        results["response_times_ms"].append(response_time_ms)
        results["total_requests"] += 1

        # Wait between requests
        if i < test_requests - 1:  # Don't wait after last request
            time.sleep(1.0)

    # Calculate statistics
    if results["response_times_ms"]:
        avg_response_time = sum(results["response_times_ms"]) / len(results["response_times_ms"])
        results["avg_response_time_ms"] = int(avg_response_time)
        results["min_response_time_ms"] = min(results["response_times_ms"])
        results["max_response_time_ms"] = max(results["response_times_ms"])

    # Recommend optimal spacing
    if results["rate_limit_detected"]:
        results["optimal_request_spacing_ms"] = 2000  # 2 seconds if rate limited
    else:
        # Conservative spacing for production
        results["optimal_request_spacing_ms"] = 1000  # 1 second baseline

    return results

def analyze_baseline_results(results: Dict[str, Any]) -> None:
    """Analyze and report baseline test results."""
    print(f"\n{'='*60}")
    print("RATE LIMITING BASELINE ANALYSIS")
    print(f"{'='*60}")

    print(f"📊 Request Summary:")
    print(f"   Total Requests: {results['total_requests']}")
    print(f"   Successful: {results['successful_requests']}")
    print(f"   Failed: {results['failed_requests']}")

    if results["response_times_ms"]:
        print(f"\n⏱️  Response Time Analysis:")
        print(f"   Average: {results['avg_response_time_ms']}ms")
        print(f"   Range: {results['min_response_time_ms']}-{results['max_response_time_ms']}ms")

    print(f"\n🚦 Error Pattern Analysis:")
    for error_type, count in results["error_patterns"].items():
        print(f"   {error_type}: {count} occurrences")

    print(f"\n🎯 Rate Limiting Assessment:")
    if results["rate_limit_detected"]:
        print("   ⚠️  Rate limiting detected - adjust request spacing")
    else:
        print("   ✅ No rate limiting detected in baseline test")

    print(f"   Recommended spacing: {results['optimal_request_spacing_ms']}ms between requests")

    # Access tier analysis
    print(f"\n🔍 Access Tier Analysis:")
    if results["error_patterns"].get("403_forbidden", 0) == results["total_requests"]:
        print("   📋 Consistent 403 errors indicate Discovery Tier access")
        print("   💡 Model listing works, but inference requires tier upgrade")
        print("   🔧 Action: Complete billing setup or account verification")
    elif results["successful_requests"] > 0:
        print("   🎉 Inference access confirmed!")
        print("   💰 Free tier appears functional")
    else:
        print("   ❓ Mixed results - requires further investigation")

    # Production recommendations
    print(f"\n📋 Production Recommendations:")
    print(f"   • Use {results['optimal_request_spacing_ms']}ms spacing between requests")
    print(f"   • Monitor 429 responses for rate limit adjustment")
    print(f"   • Batch requests when possible to maximize free tier")

    if results["error_patterns"].get("403_forbidden", 0) > 0:
        print(f"   • Complete NVIDIA Build account activation for inference access")
        print(f"   • Contact NVIDIA support if activation doesn't resolve 403 errors")

def main():
    print("NVIDIA Build API Rate Limiting Baseline Test")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in environment variables")
        return 1

    print(f"🔑 API Key: {api_key[:15]}...")

    # Run baseline test
    try:
        results = test_rate_limiting_baseline()
        analyze_baseline_results(results)

        print(f"\n{'='*60}")
        print("BASELINE TEST COMPLETE")
        print("=" * 60)

        return 0 if results["total_requests"] > 0 else 1

    except Exception as e:
        print(f"❌ Baseline test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())