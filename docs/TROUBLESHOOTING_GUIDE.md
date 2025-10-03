# Troubleshooting and Maintenance Guide

<!-- TOC -->

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Quick Diagnostic Tools](#quick-diagnostic-tools)
  - [System Health Check Script](#system-health-check-script)
  - [Quick Diagnostic Commands](#quick-diagnostic-commands)
- [Common Issues and Solutions](#common-issues-and-solutions)
  - [Issue 1: "ModuleNotFoundError" when importing](#issue-1-modulenotfounderror-when-importing)
  - [Issue 2: "NVIDIA API key not found"](#issue-2-nvidia-api-key-not-found)
  - [Issue 3: "403 Forbidden" API Errors](#issue-3-403-forbidden-api-errors)
  - [Issue 4: Slow Response Times](#issue-4-slow-response-times)
- [API Connection Problems](#api-connection-problems)
  - [NVIDIA Build API Connection Issues](#nvidia-build-api-connection-issues)
- [Authentication and Authorization Issues](#authentication-and-authorization-issues)
  - [API Key Problems](#api-key-problems)
- [Performance Issues](#performance-issues)
  - [Slow Response Times](#slow-response-times)
- [Cost and Billing Problems](#cost-and-billing-problems)
  - [Unexpected High Costs](#unexpected-high-costs)
- [Safety Alert System Issues](#safety-alert-system-issues)
  - [Alert Not Triggering](#alert-not-triggering)
  - [Alert Delivery Issues](#alert-delivery-issues)
- [Batch Processing Problems](#batch-processing-problems)
  - [Batch Jobs Not Processing](#batch-jobs-not-processing)
- [Configuration Issues](#configuration-issues)
  - [Environment Variable Problems](#environment-variable-problems)
- [Maintenance Procedures](#maintenance-procedures)
  - [Daily Maintenance Tasks](#daily-maintenance-tasks)
  - [Weekly Maintenance Tasks](#weekly-maintenance-tasks)
  - [Monthly Maintenance Tasks](#monthly-maintenance-tasks)
- [Recovery Procedures](#recovery-procedures)
  - [System Recovery Checklist](#system-recovery-checklist)
  - [Emergency Procedures](#emergency-procedures)
  - [Data Recovery](#data-recovery)
- [Support and Resources](#support-and-resources)
  - [Log Files and Debugging](#log-files-and-debugging)
  - [Getting Help](#getting-help)
  <!-- /TOC -->

---

Last Updated: 2025-10-03
Owner: Support Team
Review Cadence: Bi-weekly

---

**Comprehensive Troubleshooting and Maintenance Documentation for Pharmaceutical RAG System**

## Overview

This guide provides comprehensive troubleshooting procedures and maintenance protocols for the pharmaceutical RAG system, covering common issues, diagnostic procedures, and preventive maintenance strategies.

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [API Connection Problems](#api-connection-problems)
4. [Authentication and Authorization Issues](#authentication-and-authorization-issues)
5. [Performance Issues](#performance-issues)
6. [Cost and Billing Problems](#cost-and-billing-problems)
7. [Safety Alert System Issues](#safety-alert-system-issues)
8. [Batch Processing Problems](#batch-processing-problems)
9. [Configuration Issues](#configuration-issues)
10. [Maintenance Procedures](#maintenance-procedures)
11. [Monitoring and Health Checks](#monitoring-and-health-checks)
12. [Recovery Procedures](#recovery-procedures)

---

## Quick Diagnostic Tools

### System Health Check Script

Create and run this comprehensive health check:

```bash
#!/bin/bash
# System Health Check Script

echo "🏥 PHARMACEUTICAL RAG SYSTEM HEALTH CHECK"
echo "=========================================="
date

# Check Python environment
echo -e "\n🐍 Python Environment:"
python --version
pip --version

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment active"
fi

# Check environment variables
echo -e "\n🔧 Environment Configuration:"
if [ -n "$NVIDIA_API_KEY" ]; then
    echo "✅ NVIDIA API key configured"
else
    echo "❌ NVIDIA API key missing"
fi

if [ "$ENABLE_CLOUD_FIRST_STRATEGY" = "true" ]; then
    echo "✅ Cloud-first strategy enabled"
else
    echo "⚠️  Cloud-first strategy disabled"
fi

if [ "$ENABLE_PHARMACEUTICAL_OPTIMIZATION" = "true" ]; then
    echo "✅ Pharmaceutical optimization enabled"
else
    echo "⚠️  Pharmaceutical optimization disabled"
fi

# Check API connectivity
echo -e "\n🌐 API Connectivity:"
python -c "
import sys
try:
    from src.clients.openai_wrapper import create_nvidia_build_client
    client = create_nvidia_build_client()
    models = client.list_available_models()
    print(f'✅ NVIDIA Build API accessible ({len(models)} models)')
except Exception as e:
    print(f'❌ NVIDIA Build API error: {str(e)}')
    sys.exit(1)
"

# Check pharmaceutical capabilities
echo -e "\n💊 Pharmaceutical Capabilities:"
python -c "
try:
    from src.clients.nemo_client_enhanced import create_pharmaceutical_client
    client = create_pharmaceutical_client()
    capabilities = client.test_pharmaceutical_capabilities()
    print(f'✅ Pharmaceutical client: {capabilities[\"overall_status\"]}')
except Exception as e:
    print(f'❌ Pharmaceutical capabilities error: {str(e)}')
"

# Check disk space
echo -e "\n💾 System Resources:"
df -h . | head -2
free -h | head -2

echo -e "\n✅ Health check completed"
```

### Quick Diagnostic Commands

```bash
# Quick API test
python -c "
from src.clients.openai_wrapper import create_nvidia_build_client
client = create_nvidia_build_client()
result = client.test_connection()
print(f'API Status: {\"✅ Working\" if result[\"success\"] else \"❌ Failed\"}')
print(f'Response Time: {result.get(\"response_time_ms\", 0)}ms')
"

# Quick pharmaceutical test
python -c "
from src.pharmaceutical.query_classifier import classify_pharmaceutical_query
result = classify_pharmaceutical_query('test metformin safety query')
print(f'Classification: {result.domain.value} (confidence: {result.confidence_score:.2f})')
"

# Check configuration
python -c "
from src.enhanced_config import EnhancedRAGConfig
config = EnhancedRAGConfig.from_env()
strategy = config.get_cloud_first_strategy()
print(f'Cloud-first: {strategy[\"cloud_first_enabled\"]}')
print(f'Pharmaceutical: {config.pharmaceutical_optimized}')
"

# Check logs
tail -n 20 logs/*.log 2>/dev/null || echo "No log files found"
```

---

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError" when importing

**Symptoms:**

```
ModuleNotFoundError: No module named 'src.clients'
```

**Diagnosis:**

```bash
# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Check if src directory exists
ls -la src/
```

**Solutions:**

1. **Set Python Path:**

   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Permanent Solution (add to ~/.bashrc):**

   ```bash
   echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/project/src"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Virtual Environment Fix:**
   ```bash
   # If using virtual environment
   echo "$(pwd)/src" > venv/lib/python*/site-packages/pharmaceutical-rag.pth
   ```

### Issue 2: "NVIDIA API key not found"

**Symptoms:**

```
NVIDIABuildError: NVIDIA API key required. Set NVIDIA_API_KEY environment variable
```

**Diagnosis:**

```bash
# Check if API key is set
echo "API Key set: ${NVIDIA_API_KEY:+Yes}"

# Check .env file
if [ -f .env ]; then
    grep "NVIDIA_API_KEY" .env
else
    echo "No .env file found"
fi
```

**Solutions:**

1. **Set Environment Variable:**

   ```bash
   export NVIDIA_API_KEY="your_api_key_here"
   ```

2. **Create/Update .env File:**

   ```bash
   echo "NVIDIA_API_KEY=your_api_key_here" >> .env
   ```

3. **Load .env in Python:**
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Add this before importing other modules
   ```

### Issue 3: "403 Forbidden" API Errors

**Symptoms:**

```
OpenAIError: 403 Forbidden - You don't have access to this model
```

**Diagnosis:**

```python
# Check API access level
from src.clients.openai_wrapper import create_nvidia_build_client
client = create_nvidia_build_client()
try:
    models = client.list_available_models()
    print(f"Available models: {len(models)}")
    for model in models[:3]:
        print(f"  - {model['id']}")
except Exception as e:
    print(f"Access level issue: {e}")
```

**Solutions:**

1. **Verify API Key:**

   - Log into NVIDIA Developer Portal
   - Check API key is valid and active
   - Verify account tier (Discovery vs. Full access)

2. **Request Model Access:**

   - Contact NVIDIA support for model access
   - Upgrade account tier if needed

3. **Use Available Models:**
   ```python
   # List actually available models
   available_models = client.list_available_models()
   embedding_models = [m for m in available_models if "embed" in m["id"].lower()]
   chat_models = [m for m in available_models if any(x in m["id"].lower() for x in ["llama", "mistral", "gemma"])]
   ```

### Issue 4: Slow Response Times

**Symptoms:**

```
Response times consistently > 10 seconds
Timeouts occurring frequently
```

**Diagnosis:**

```python
# Performance benchmark
import time
from src.clients.nemo_client_enhanced import create_pharmaceutical_client

client = create_pharmaceutical_client()

# Test response times
start = time.time()
response = client.create_embeddings(["test query"])
end = time.time()

print(f"Response time: {(end - start) * 1000:.0f}ms")
print(f"Success: {response.success}")
print(f"Endpoint: {response.endpoint_type}")
```

**Solutions:**

1. **Check Network Connectivity:**

   ```bash
   # Test connectivity to NVIDIA endpoints
   curl -w "Time: %{time_total}s\n" -o /dev/null -s https://integrate.api.nvidia.com
   ```

2. **Optimize Configuration:**

   ```python
   # Reduce timeout for faster failover
   config = NVIDIABuildConfig(
       timeout=30,  # Reduce from default 60
       max_retries=2  # Reduce retries
   )
   ```

3. **Enable Batch Processing:**
   ```python
   # Use batch processing for multiple queries
   async with create_pharmaceutical_research_session() as batch_client:
       # Queue multiple queries for batch processing
       pass
   ```

---

## API Connection Problems

### NVIDIA Build API Connection Issues

#### Connection Timeout Problems

**Symptoms:**

- Requests timing out after 60 seconds
- "Connection timeout" errors
- Intermittent connectivity issues

**Diagnostic Commands:**

```bash
# Test direct API connectivity
curl -H "Authorization: Bearer $NVIDIA_API_KEY" \
     -H "Content-Type: application/json" \
     --max-time 30 \
     "https://integrate.api.nvidia.com/v1/models"

# Check DNS resolution
nslookup integrate.api.nvidia.com

# Test network latency
ping -c 5 integrate.api.nvidia.com
```

**Solutions:**

1. **Adjust Timeout Settings:**

   ```python
   config = NVIDIABuildConfig(
       timeout=120,  # Increase timeout
       max_retries=5  # Increase retries
   )
   ```

2. **Implement Retry Logic:**

   ```python
   import time
   import asyncio

   async def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await func()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise e
               wait_time = (2 ** attempt) * 1.0  # Exponential backoff
               await asyncio.sleep(wait_time)
   ```

3. **Use Connection Pooling:**

   ```python
   from src.pharmaceutical.model_optimization import PharmaceuticalClientPool

   # Use connection pool to manage connections efficiently
   client_pool = PharmaceuticalClientPool(pool_size=3)
   async with client_pool.acquire_client() as client:
       response = client.create_embeddings(["query"])
   ```

#### Rate Limiting Issues

**Symptoms:**

- "429 Too Many Requests" errors
- Requests being throttled
- Inconsistent response times

**Diagnostic Script:**

```python
import time
from collections import deque

def diagnose_rate_limiting():
    """Diagnose rate limiting issues."""

    # Track request times
    request_times = deque(maxlen=60)  # Last minute

    for i in range(10):
        start_time = time.time()
        try:
            client = create_pharmaceutical_client()
            response = client.create_embeddings([f"test query {i}"])

            end_time = time.time()
            request_times.append(end_time - start_time)

            print(f"Request {i+1}: {(end_time - start_time)*1000:.0f}ms - {'✅' if response.success else '❌'}")

        except Exception as e:
            if "429" in str(e):
                print(f"Request {i+1}: ❌ Rate limited")
            else:
                print(f"Request {i+1}: ❌ Error: {e}")

        time.sleep(1)  # 1 second between requests

    if request_times:
        avg_time = sum(request_times) / len(request_times)
        print(f"\nAverage response time: {avg_time*1000:.0f}ms")
```

**Solutions:**

1. **Implement Rate Limiting:**

   ```python
   import asyncio
   from datetime import datetime, timedelta

   class RateLimiter:
       def __init__(self, max_requests=50, time_window=60):
           self.max_requests = max_requests
           self.time_window = timedelta(seconds=time_window)
           self.requests = deque()

       async def acquire(self):
           now = datetime.now()
           # Remove old requests
           while self.requests and now - self.requests[0] > self.time_window:
               self.requests.popleft()

           if len(self.requests) >= self.max_requests:
               # Wait until we can make a request
               wait_time = (self.requests[0] + self.time_window - now).total_seconds()
               await asyncio.sleep(max(0, wait_time))

           self.requests.append(now)
   ```

2. **Use Batch Processing:**

   ```python
   # Batch multiple queries to reduce API calls
   queries = ["query1", "query2", "query3", "query4", "query5"]

   # Instead of 5 separate calls, batch them
   batch_response = client.create_embeddings(queries)
   ```

---

## Authentication and Authorization Issues

### API Key Problems

#### Invalid or Expired API Key

**Symptoms:**

- "401 Unauthorized" errors
- "Invalid API key" messages
- Authentication failures

**Diagnostic Steps:**

```python
def diagnose_api_key_issues():
    """Diagnose API key authentication issues."""

    import os
    import requests

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        print("❌ No API key found in environment")
        return

    if len(api_key) < 10:
        print("❌ API key appears too short")
        return

    if not api_key.startswith(('nvapi-', 'sk-')):
        print("⚠️  API key format may be incorrect")

    # Test API key validity
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            print("✅ API key is valid")
            models = response.json()
            print(f"   Available models: {len(models.get('data', []))}")
        elif response.status_code == 401:
            print("❌ API key is invalid or expired")
        elif response.status_code == 403:
            print("⚠️  API key valid but insufficient permissions")
        else:
            print(f"❌ Unexpected response: {response.status_code}")

    except requests.RequestException as e:
        print(f"❌ Connection error: {e}")

# Run diagnosis
diagnose_api_key_issues()
```

**Solutions:**

1. **Regenerate API Key:**

   ```bash
   # Log into NVIDIA Developer Portal
   # Navigate to API Keys section
   # Generate new API key
   # Update environment variable
   export NVIDIA_API_KEY="new_api_key_here"
   ```

2. **Secure API Key Storage:**

   ```python
   from cryptography.fernet import Fernet

   def secure_api_key_setup():
       # Generate encryption key
       key = Fernet.generate_key()
       with open('.encryption_key', 'wb') as f:
           f.write(key)

       # Encrypt API key
       f = Fernet(key)
       api_key = "your_api_key_here"
       encrypted_key = f.encrypt(api_key.encode())

       with open('.encrypted_api_key', 'wb') as f:
           f.write(encrypted_key)

       print("✅ API key securely stored")
   ```

#### Permission and Access Level Issues

**Symptoms:**

- Access to some models but not others
- "Insufficient permissions" errors
- Limited functionality

**Diagnostic Commands:**

```python
def check_access_levels():
    """Check API access levels and permissions."""

    from src.clients.openai_wrapper import create_nvidia_build_client

    client = create_nvidia_build_client()

    try:
        # Check model access
        models = client.list_available_models()

        # Categorize available models
        embedding_models = [m for m in models if "embed" in m["id"].lower()]
        chat_models = [m for m in models if any(x in m["id"].lower() for x in ["llama", "mistral", "gemma"])]

        print(f"Access Level Analysis:")
        print(f"  Total models: {len(models)}")
        print(f"  Embedding models: {len(embedding_models)}")
        print(f"  Chat models: {len(chat_models)}")

        if len(models) < 10:
            print("⚠️  Limited model access - may be Discovery Tier")
        else:
            print("✅ Full model access available")

        # Test actual model usage
        if embedding_models:
            try:
                test_response = client.create_embeddings(["test"], model=embedding_models[0]["id"])
                print("✅ Embedding API functional")
            except Exception as e:
                print(f"❌ Embedding API error: {e}")

        if chat_models:
            try:
                test_response = client.create_chat_completion(
                    [{"role": "user", "content": "test"}],
                    model=chat_models[0]["id"]
                )
                print("✅ Chat API functional")
            except Exception as e:
                print(f"❌ Chat API error: {e}")

    except Exception as e:
        print(f"❌ Model access check failed: {e}")

check_access_levels()
```

---

## Performance Issues

### Slow Response Times

#### Network Latency Issues

**Diagnostic Script:**

```python
import time
import statistics
from src.clients.nemo_client_enhanced import create_pharmaceutical_client

def diagnose_network_performance():
    """Diagnose network performance issues."""

    client = create_pharmaceutical_client()
    response_times = []

    print("🌐 Network Performance Diagnosis")
    print("Testing 10 requests...")

    for i in range(10):
        start_time = time.time()
        try:
            response = client.create_embeddings([f"test query {i}"])
            end_time = time.time()

            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)

            status = "✅" if response.success else "❌"
            endpoint = response.endpoint_type.value if response.endpoint_type else "unknown"

            print(f"  {i+1:2d}. {status} {response_time:6.0f}ms ({endpoint})")

        except Exception as e:
            print(f"  {i+1:2d}. ❌ Error: {e}")

        time.sleep(0.5)  # Brief pause between requests

    if response_times:
        print(f"\nPerformance Summary:")
        print(f"  Average: {statistics.mean(response_times):6.0f}ms")
        print(f"  Median:  {statistics.median(response_times):6.0f}ms")
        print(f"  Min:     {min(response_times):6.0f}ms")
        print(f"  Max:     {max(response_times):6.0f}ms")

        avg_time = statistics.mean(response_times)
        if avg_time > 10000:
            print("❌ Performance issue detected (>10s average)")
            print("   Consider: network optimization, server location, batch processing")
        elif avg_time > 5000:
            print("⚠️  Performance suboptimal (>5s average)")
            print("   Consider: connection pooling, caching, timeout optimization")
        else:
            print("✅ Performance acceptable (<5s average)")

diagnose_network_performance()
```

#### Memory and CPU Issues

**Diagnostic Commands:**

```bash
# Check system resources
echo "System Resource Usage:"
echo "Memory:"
free -h

echo -e "\nCPU:"
top -n 1 -b | head -5

echo -e "\nDisk I/O:"
iostat 1 3 2>/dev/null || echo "iostat not available"

# Check Python memory usage
python -c "
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()

print(f'Python Memory Usage:')
print(f'  RSS: {memory_info.rss / 1024 / 1024:.1f} MB')
print(f'  VMS: {memory_info.vms / 1024 / 1024:.1f} MB')
print(f'  CPU: {process.cpu_percent():.1f}%')
"
```

**Solutions:**

1. **Memory Optimization:**

   ```python
   # Implement memory-efficient processing
   def process_large_batch_efficiently(queries, batch_size=10):
       results = []

       for i in range(0, len(queries), batch_size):
           batch = queries[i:i + batch_size]

           # Process batch
           batch_result = client.create_embeddings(batch)
           results.extend(batch_result.data["embeddings"])

           # Clear memory
           del batch_result

           # Optional: garbage collection
           import gc
           gc.collect()

       return results
   ```

2. **CPU Optimization:**

   ```python
   # Use async processing for CPU-intensive tasks
   import asyncio

   async def async_pharmaceutical_processing(queries):
       tasks = []
       semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

       async def process_query(query):
           async with semaphore:
               return await client.create_embeddings([query])

       for query in queries:
           tasks.append(process_query(query))

       results = await asyncio.gather(*tasks)
       return results
   ```

---

## Cost and Billing Problems

### Unexpected High Costs

#### Cost Tracking and Analysis

**Diagnostic Script:**

```python
from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker

def diagnose_cost_issues():
    """Diagnose unexpected cost issues."""

    cost_tracker = create_pharmaceutical_cost_tracker()

    # Analyze recent costs
    analysis = cost_tracker.get_cost_analysis(days_back=7)

    print("💰 Cost Analysis (Last 7 Days)")
    print("=" * 40)

    cost_breakdown = analysis["cost_breakdown"]
    print(f"Total Cost: ${cost_breakdown['total_cost_usd']:.4f}")
    print(f"Free Tier: ${cost_breakdown['free_tier_cost_usd']:.4f}")
    print(f"Infrastructure: ${cost_breakdown['infrastructure_cost_usd']:.4f}")
    print(f"Average per Query: ${cost_breakdown['cost_per_query_avg']:.4f}")

    # Check free tier utilization
    free_tier_opt = analysis["free_tier_optimization"]
    print(f"\nFree Tier Utilization: {free_tier_opt['efficiency_score']:.1%}")

    if free_tier_opt['efficiency_score'] < 0.7:
        print("⚠️  Low free tier utilization detected")
        print("   Consider: batch processing, query scheduling")

    # Analyze cost by query type
    cost_by_type = cost_breakdown.get("cost_by_query_type", {})
    if cost_by_type:
        print(f"\nCost by Query Type:")
        for qtype, data in cost_by_type.items():
            print(f"  {qtype}: ${data['total_cost']:.4f} ({data['query_count']} queries)")

    # Check for cost recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Cost Optimization Recommendations:")
        for rec in recommendations:
            if rec.get("potential_savings_usd"):
                print(f"  • {rec['title']}: Save ${rec['potential_savings_usd']:.3f}")
            else:
                print(f"  • {rec['title']}")

diagnose_cost_issues()
```

#### Free Tier Exhaustion

**Symptoms:**

- Queries automatically falling back to infrastructure
- High costs despite targeting free tier
- "Free tier limit exceeded" messages

**Diagnostic Commands:**

```python
def check_free_tier_status():
    """Check free tier usage status."""

    from src.monitoring.credit_tracker import PharmaceuticalCreditTracker

    tracker = PharmaceuticalCreditTracker()
    usage_stats = tracker.get_usage_stats()

    print("📊 Free Tier Usage Status")
    print("=" * 30)
    print(f"Requests today: {usage_stats['requests_today']}")
    print(f"Requests this month: {usage_stats['requests_this_month']}")
    print(f"Monthly limit: 10,000")
    print(f"Remaining: {10000 - usage_stats['requests_this_month']}")

    # Calculate daily burn rate
    import datetime
    day_of_month = datetime.datetime.now().day
    if day_of_month > 0:
        daily_average = usage_stats['requests_this_month'] / day_of_month
        print(f"Daily average: {daily_average:.1f}")

        days_remaining = 30 - day_of_month
        projected_usage = usage_stats['requests_this_month'] + (daily_average * days_remaining)
        print(f"Projected monthly: {projected_usage:.0f}")

        if projected_usage > 10000:
            print("⚠️  Projected to exceed free tier limit")
            print("   Consider: aggressive batching, query optimization")
        else:
            print("✅ On track for free tier usage")

check_free_tier_status()
```

**Solutions:**

1. **Implement Conservation Mode:**

   ```python
   def activate_conservation_mode():
       """Activate conservation mode when approaching limits."""

       from src.optimization.batch_processor import BatchOptimizationStrategy

       conservation_strategy = BatchOptimizationStrategy(
           max_batch_size=100,  # Larger batches
           max_wait_time_seconds=60,  # Longer wait times
           enable_cost_optimization=True,
           enable_intelligent_scheduling=True
       )

       print("🛡️  Conservation mode activated")
       print("   - Increased batch sizes")
       print("   - Extended wait times")
       print("   - Priority for critical queries only")

       return conservation_strategy
   ```

2. **Smart Query Scheduling:**

   ```python
   async def implement_smart_scheduling():
       """Implement smart query scheduling for free tier optimization."""

       # Check current usage
       usage_stats = tracker.get_usage_stats()
       monthly_usage = usage_stats['requests_this_month']

       if monthly_usage > 8000:  # 80% of limit
           print("⚠️  High monthly usage - implementing restrictions")

           # Only process critical and high priority queries
           allowed_priorities = [RequestPriority.CRITICAL, RequestPriority.HIGH]

           # Increase batch processing
           batch_threshold = 5  # Process in batches of 5+

       elif monthly_usage > 6000:  # 60% of limit
           print("⚠️  Moderate usage - optimizing batch processing")

           allowed_priorities = [
               RequestPriority.CRITICAL,
               RequestPriority.HIGH,
               RequestPriority.NORMAL
           ]

           batch_threshold = 3  # Process in batches of 3+

       else:
           # Normal operation
           allowed_priorities = list(RequestPriority)
           batch_threshold = 1

       return {
           "allowed_priorities": allowed_priorities,
           "batch_threshold": batch_threshold,
           "conservation_mode": monthly_usage > 8000
       }
   ```

---

## Safety Alert System Issues

### Alert Not Triggering

**Symptoms:**

- Safety-critical queries not generating alerts
- Missing drug interaction warnings
- Silent failures in alert system

**Diagnostic Script:**

```python
def diagnose_alert_system():
    """Diagnose safety alert system issues."""

    from src.pharmaceutical.safety_alert_integration import DrugSafetyAlertIntegration
    from src.pharmaceutical.query_classifier import classify_pharmaceutical_query

    safety_integration = DrugSafetyAlertIntegration()

    # Test critical safety query
    critical_query = "Patient experiencing severe bleeding while taking warfarin and aspirin together"

    print("🚨 Safety Alert System Diagnosis")
    print("=" * 40)
    print(f"Test Query: {critical_query}")

    # Test classification
    context = classify_pharmaceutical_query(critical_query)
    print(f"\nClassification Results:")
    print(f"  Domain: {context.domain.value}")
    print(f"  Safety Urgency: {context.safety_urgency.name}")
    print(f"  Priority: {context.research_priority.name}")
    print(f"  Drug Names: {context.drug_names}")
    print(f"  Confidence: {context.confidence_score:.3f}")

    # Test alert generation
    try:
        context, alerts = await safety_integration.process_pharmaceutical_query(critical_query)

        print(f"\nAlert Generation Results:")
        print(f"  Alerts Generated: {len(alerts)}")

        for i, alert in enumerate(alerts, 1):
            print(f"  Alert {i}:")
            print(f"    Type: {alert.alert_type.value}")
            print(f"    Urgency: {alert.urgency.value}")
            print(f"    Message: {alert.safety_message}")
            print(f"    Drugs: {alert.drug_names}")

        if not alerts:
            print("❌ No alerts generated for critical safety query")
            print("   Check: alert generation logic, safety keywords")
        elif context.safety_urgency.name not in ["CRITICAL", "HIGH"]:
            print("⚠️  Query not classified as critical/high safety")
            print("   Check: safety keyword detection, classification logic")
        else:
            print("✅ Alert system functioning correctly")

    except Exception as e:
        print(f"❌ Alert generation failed: {e}")

# Run diagnosis
import asyncio
asyncio.run(diagnose_alert_system())
```

**Solutions:**

1. **Fix Safety Keyword Detection:**

   ```python
   # Enhanced safety keyword detection
   ENHANCED_SAFETY_KEYWORDS = {
       "immediate_danger": [
           "severe bleeding", "life-threatening", "emergency", "urgent",
           "overdose", "toxic reaction", "anaphylaxis", "cardiac arrest"
       ],
       "drug_interactions": [
           "taking together", "combined with", "interaction between",
           "contraindicated combination", "dangerous combination"
       ],
       "contraindications": [
           "contraindicated", "should not use", "avoid in patients",
           "not recommended", "increased risk"
       ]
   }

   def enhance_safety_detection(query_text):
       """Enhanced safety detection logic."""
       query_lower = query_text.lower()

       safety_score = 0
       detected_categories = []

       for category, keywords in ENHANCED_SAFETY_KEYWORDS.items():
           for keyword in keywords:
               if keyword in query_lower:
                   safety_score += 1
                   detected_categories.append(category)

       return safety_score, detected_categories
   ```

2. **Alert System Validation:**

   ```python
   def validate_alert_system():
       """Validate alert system configuration."""

       # Test cases with expected results
       test_cases = [
           {
               "query": "Warfarin and aspirin interaction bleeding risk",
               "expected_urgency": "URGENT",
               "expected_alerts": 1
           },
           {
               "query": "Metformin contraindications in kidney disease",
               "expected_urgency": "HIGH",
               "expected_alerts": 1
           },
           {
               "query": "ACE inhibitor mechanism of action",
               "expected_urgency": "NONE",
               "expected_alerts": 0
           }
       ]

       for test_case in test_cases:
           context, alerts = process_pharmaceutical_query(test_case["query"])

           success = (
               context.safety_urgency.name == test_case["expected_urgency"] and
               len(alerts) == test_case["expected_alerts"]
           )

           print(f"{'✅' if success else '❌'} {test_case['query'][:50]}...")
   ```

### Alert Delivery Issues

**Symptoms:**

- Alerts generated but not delivered
- Email/Slack notifications not working
- Alert acknowledgment issues

**Diagnostic Commands:**

```bash
# Check email configuration
python -c "
import os
print('Email Configuration:')
print(f'  SMTP Host: {os.getenv(\"SMTP_HOST\", \"Not set\")}')
print(f'  SMTP Port: {os.getenv(\"SMTP_PORT\", \"Not set\")}')
print(f'  From Email: {os.getenv(\"ALERT_EMAIL_FROM\", \"Not set\")}')
print(f'  To Email: {os.getenv(\"ALERT_EMAIL_TO\", \"Not set\")}')
"

# Test email sending
python -c "
import smtplib
from email.mime.text import MIMEText
import os

try:
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    username = os.getenv('SMTP_USERNAME')
    password = os.getenv('SMTP_PASSWORD')

    if all([smtp_host, username, password]):
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(username, password)
        server.quit()
        print('✅ Email configuration working')
    else:
        print('❌ Email configuration incomplete')
except Exception as e:
    print(f'❌ Email test failed: {e}')
"
```

---

## Batch Processing Problems

### Batch Jobs Not Processing

**Symptoms:**

- Queries queued but not processed
- Batch processing hanging
- Inconsistent batch execution

**Diagnostic Script:**

```python
async def diagnose_batch_processing():
    """Diagnose batch processing issues."""

    from src.optimization.batch_integration import create_pharmaceutical_research_session

    print("📦 Batch Processing Diagnosis")
    print("=" * 35)

    try:
        # Create batch session
        batch_client = await create_pharmaceutical_research_session(
            auto_process_seconds=10,  # Short interval for testing
            aggressive_optimization=True
        )

        print("✅ Batch client created successfully")

        # Queue test requests
        print("Queueing test requests...")
        test_queries = [
            "test pharmaceutical query 1",
            "test pharmaceutical query 2",
            "test pharmaceutical query 3"
        ]

        request_ids = []
        for i, query in enumerate(test_queries):
            request_id = await batch_client.queue_embedding_request(
                texts=[query],
                priority=RequestPriority.NORMAL
            )
            request_ids.append(request_id)
            print(f"  Queued {i+1}: {request_id}")

        # Check queue status
        status = batch_client.get_comprehensive_status()
        queue_status = status["batch_processor"]

        print(f"\nQueue Status:")
        print(f"  Total queued: {queue_status['total_queued_requests']}")
        print(f"  Processing active: {queue_status['processing_active']}")

        if queue_status['total_queued_requests'] == 0:
            print("❌ Queries not queued properly")
            return

        # Attempt immediate processing
        print("Attempting immediate batch processing...")
        result = await batch_client.process_batches_now(max_concurrent=2)

        if result.success:
            print("✅ Batch processing successful")
            print(f"  Batches processed: {len(result.results)}")
        else:
            print("❌ Batch processing failed")
            print(f"  Errors: {result.errors}")

        # Check final queue status
        final_status = batch_client.get_comprehensive_status()
        final_queue = final_status["batch_processor"]
        print(f"\nFinal Queue Status:")
        print(f"  Remaining queued: {final_queue['total_queued_requests']}")

    except Exception as e:
        print(f"❌ Batch diagnosis failed: {e}")
        import traceback
        traceback.print_exc()

# Run diagnosis
import asyncio
asyncio.run(diagnose_batch_processing())
```

**Solutions:**

1. **Fix Queue Management:**

   ```python
   def fix_queue_issues():
       """Fix common queue management issues."""

       # Clear stuck queues
       batch_processor.clear_queues()
       print("✅ Queues cleared")

       # Reset batch processing state
       batch_processor._stop_processing = False
       batch_processor.is_processing = False
       print("✅ Processing state reset")

       # Validate batch processor configuration
       strategy = batch_processor.strategy
       print(f"Batch Configuration:")
       print(f"  Max batch size: {strategy.max_batch_size}")
       print(f"  Max wait time: {strategy.max_wait_time_seconds}")
       print(f"  Cost optimization: {strategy.enable_cost_optimization}")
   ```

2. **Debug Auto-Processing:**

   ```python
   async def debug_auto_processing():
       """Debug automatic batch processing."""

       # Create batch client with debugging
       batch_client = await create_pharmaceutical_research_session(
           auto_process_seconds=5,  # Very short interval
           aggressive_optimization=False  # Disable for debugging
       )

       # Monitor auto-processing
       print("Monitoring auto-processing...")
       for i in range(5):
           # Queue a test request
           await batch_client.queue_embedding_request(
               texts=[f"debug query {i}"],
               priority=RequestPriority.NORMAL
           )

           status = batch_client.get_comprehensive_status()
           print(f"  {i}: Queued={status['batch_processor']['total_queued_requests']}")

           # Wait for auto-processing
           await asyncio.sleep(6)  # Slightly longer than auto-process interval

           status_after = batch_client.get_comprehensive_status()
           print(f"     After auto-process: Queued={status_after['batch_processor']['total_queued_requests']}")
   ```

---

## Configuration Issues

### Environment Variable Problems

**Symptoms:**

- Configuration not loading properly
- Feature flags not working
- Default values being used instead of configured values

**Diagnostic Script:**

```python
def diagnose_configuration():
    """Diagnose configuration issues."""

    import os
    from src.enhanced_config import EnhancedRAGConfig

    print("⚙️  Configuration Diagnosis")
    print("=" * 30)

    # Check environment variables
    required_vars = [
        "NVIDIA_API_KEY",
        "ENABLE_CLOUD_FIRST_STRATEGY",
        "ENABLE_PHARMACEUTICAL_OPTIMIZATION"
    ]

    print("Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask API key for security
            if "API_KEY" in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: Not set")

    # Check .env file loading
    env_file_exists = os.path.exists('.env')
    print(f"\n.env file: {'✅ Found' if env_file_exists else '❌ Not found'}")

    if env_file_exists:
        with open('.env', 'r') as f:
            env_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"  Lines in .env: {len(env_lines)}")

    # Test configuration loading
    try:
        config = EnhancedRAGConfig.from_env()
        print(f"\n✅ Configuration loaded successfully")

        # Check key configuration values
        strategy = config.get_cloud_first_strategy()
        print(f"  Cloud-first enabled: {strategy['cloud_first_enabled']}")
        print(f"  Pharmaceutical optimized: {config.pharmaceutical_optimized}")

        # Check feature flags
        feature_flags = config.get_feature_flags()
        print(f"  Feature flags loaded: {len(feature_flags)}")

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")

diagnose_configuration()
```

**Solutions:**

1. **Fix Environment Loading:**

   ```python
   # Ensure .env is loaded before imports
   from dotenv import load_dotenv
   import os

   # Load .env file explicitly
   env_path = '.env'
   if os.path.exists(env_path):
       load_dotenv(env_path)
       print(f"✅ Loaded .env from {env_path}")
   else:
       print(f"⚠️  No .env file found at {env_path}")

   # Verify loading
   test_var = os.getenv('NVIDIA_API_KEY')
   if test_var:
       print("✅ Environment variables accessible")
   else:
       print("❌ Environment variables not loaded")
   ```

2. **Configuration Validation:**

   ```python
   def validate_configuration():
       """Validate complete configuration setup."""

       config = EnhancedRAGConfig.from_env()

       validation_results = {
           "api_key_configured": bool(os.getenv("NVIDIA_API_KEY")),
           "cloud_first_enabled": config.get_cloud_first_strategy()["cloud_first_enabled"],
           "pharmaceutical_optimized": config.pharmaceutical_optimized,
           "alerts_configured": bool(config.get_alerts_config()),
           "feature_flags_loaded": len(config.get_feature_flags()) > 0
       }

       all_valid = all(validation_results.values())

       print("Configuration Validation:")
       for key, valid in validation_results.items():
           print(f"  {'✅' if valid else '❌'} {key.replace('_', ' ').title()}")

       if all_valid:
           print("✅ All configuration valid")
       else:
           print("❌ Configuration issues detected")

       return validation_results
   ```

---

## Maintenance Procedures

### Daily Maintenance Tasks

```bash
#!/bin/bash
# Daily maintenance script

echo "🔧 Daily Pharmaceutical RAG Maintenance"
echo "Date: $(date)"
echo "======================================="

# 1. Check system health
echo -e "\n1. System Health Check:"
./scripts/system_health_check.sh

# 2. Check API connectivity
echo -e "\n2. API Connectivity:"
python -c "
from src.clients.openai_wrapper import create_nvidia_build_client
try:
    client = create_nvidia_build_client()
    result = client.test_connection()
    print(f'✅ API Status: {\"Working\" if result[\"success\"] else \"Failed\"}')
    print(f'   Response Time: {result.get(\"response_time_ms\", 0)}ms')
except Exception as e:
    print(f'❌ API Error: {e}')
"

# 3. Check free tier usage
echo -e "\n3. Free Tier Usage:"
python -c "
from src.monitoring.credit_tracker import PharmaceuticalCreditTracker
tracker = PharmaceuticalCreditTracker()
usage = tracker.get_usage_stats()
print(f'Requests today: {usage[\"requests_today\"]}')
print(f'Monthly usage: {usage[\"requests_this_month\"]}/10,000')
remaining = 10000 - usage['requests_this_month']
if remaining < 1000:
    print('⚠️  Low free tier remaining')
else:
    print('✅ Free tier usage healthy')
"

# 4. Check for active alerts
echo -e "\n4. Active Alerts:"
python -c "
from src.monitoring.endpoint_health_monitor import create_endpoint_health_monitor
import asyncio

async def check_alerts():
    monitor = create_endpoint_health_monitor()
    alerts = monitor.get_active_alerts()
    if alerts:
        print(f'⚠️  {len(alerts)} active alerts:')
        for alert in alerts[:3]:
            print(f'   - {alert[\"severity\"]}: {alert[\"message\"]}')
    else:
        print('✅ No active alerts')

asyncio.run(check_alerts())
"

# 5. Log rotation
echo -e "\n5. Log Maintenance:"
find logs/ -name "*.log" -size +100M -exec echo "Large log file: {}" \;
find logs/ -name "*.log" -mtime +30 -exec echo "Old log file: {}" \;

# 6. Backup configuration
echo -e "\n6. Configuration Backup:"
if [ -f ./scripts/backup_config.sh ]; then
    ./scripts/backup_config.sh
    echo "✅ Configuration backed up"
else
    echo "⚠️  Backup script not found"
fi

echo -e "\n✅ Daily maintenance completed: $(date)"
```

### Weekly Maintenance Tasks

```bash
#!/bin/bash
# Weekly maintenance script

echo "📊 Weekly Pharmaceutical RAG Maintenance"
echo "Date: $(date)"
echo "========================================"

# 1. Performance analysis
echo -e "\n1. Performance Analysis:"
python -c "
from src.pharmaceutical.model_optimization import PharmaceuticalClientPool
import asyncio
import statistics
import time

async def weekly_performance_analysis():
    response_times = []

    pool = PharmaceuticalClientPool(pool_size=3)
    await pool.initialize()

    for i in range(20):  # More comprehensive test
        start = time.time()
        async with pool.acquire_client() as client:
            response = client.create_embeddings([f'weekly test query {i}'])
        end = time.time()

        if response.success:
            response_times.append((end - start) * 1000)

    if response_times:
        avg = statistics.mean(response_times)
        median = statistics.median(response_times)
        print(f'Performance over 20 requests:')
        print(f'  Average: {avg:.0f}ms')
        print(f'  Median: {median:.0f}ms')
        print(f'  Success rate: {len(response_times)/20:.1%}')

        if avg > 5000:
            print('⚠️  Performance degradation detected')
        else:
            print('✅ Performance healthy')

asyncio.run(weekly_performance_analysis())
"

# 2. Cost analysis
echo -e "\n2. Weekly Cost Analysis:"
python -c "
from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker

tracker = create_pharmaceutical_cost_tracker()
analysis = tracker.get_cost_analysis(days_back=7)

cost_breakdown = analysis['cost_breakdown']
print(f'Weekly costs:')
print(f'  Total: \${cost_breakdown[\"total_cost_usd\"]:.4f}')
print(f'  Free tier: \${cost_breakdown[\"free_tier_cost_usd\"]:.4f}')
print(f'  Infrastructure: \${cost_breakdown[\"infrastructure_cost_usd\"]:.4f}')

free_tier_opt = analysis['free_tier_optimization']
efficiency = free_tier_opt['efficiency_score']
print(f'  Free tier efficiency: {efficiency:.1%}')

if efficiency < 0.75:
    print('⚠️  Low free tier efficiency')
else:
    print('✅ Free tier optimization healthy')
"

# 3. Security audit
echo -e "\n3. Security Check:"
python ./scripts/security_audit.py

# 4. Database maintenance
echo -e "\n4. Database Maintenance:"
if [ -f metrics.db ]; then
    echo "Database size: $(du -h metrics.db | cut -f1)"

    # Simple database cleanup
    python -c "
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('metrics.db')
cutoff_date = datetime.now() - timedelta(days=90)

# Clean old entries (if table exists)
try:
    cursor = conn.execute('SELECT COUNT(*) FROM pharmaceutical_queries WHERE timestamp < ?', (cutoff_date,))
    old_count = cursor.fetchone()[0]

    conn.execute('DELETE FROM pharmaceutical_queries WHERE timestamp < ?', (cutoff_date,))
    conn.execute('VACUUM')
    conn.commit()

    print(f'Cleaned {old_count} old database entries')
except:
    print('Database cleanup skipped (table may not exist)')
finally:
    conn.close()
"
else
    echo "No metrics database found"
fi

# 5. Update checks
echo -e "\n5. System Updates:"
echo "Python packages that could be updated:"
pip list --outdated --format=freeze | head -5

echo -e "\n✅ Weekly maintenance completed: $(date)"
```

### Monthly Maintenance Tasks

```bash
#!/bin/bash
# Monthly maintenance script

echo "📅 Monthly Pharmaceutical RAG Maintenance"
echo "Date: $(date)"
echo "========================================"

# 1. Comprehensive model validation
echo -e "\n1. Comprehensive Model Validation:"
python -c "
import asyncio
from src.validation.model_validator import validate_nvidia_build_compatibility

async def monthly_validation():
    results = await validate_nvidia_build_compatibility(pharmaceutical_optimized=True)

    print(f'Model validation results:')
    print(f'  Overall status: {results[\"overall_status\"]}')
    print(f'  NGC independent: {results[\"ngc_independent\"]}')
    print(f'  Pharmaceutical optimized: {results[\"pharmaceutical_optimized\"]}')

    # Check for any model compatibility issues
    model_validation = results.get('model_validation', {})
    for model_id, result in list(model_validation.items())[:5]:  # First 5 models
        if hasattr(result, 'available'):
            status = '✅' if result.available and result.compatible else '❌'
            print(f'  {status} {model_id}: Available={result.available}, Compatible={result.compatible}')

asyncio.run(monthly_validation())
"

# 2. Complete cost analysis
echo -e "\n2. Monthly Cost Analysis:"
python -c "
from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker

tracker = create_pharmaceutical_cost_tracker()
analysis = tracker.get_cost_analysis(days_back=30)

print('Monthly cost summary:')
cost_breakdown = analysis['cost_breakdown']
print(f'  Total queries: {cost_breakdown[\"total_cost_usd\"] / 0.001:.0f}')  # Estimate
print(f'  Free tier usage: {analysis[\"free_tier_optimization\"][\"efficiency_score\"]:.1%}')

pharma_insights = analysis.get('pharmaceutical_insights', {})
if 'total_research_value' in pharma_insights:
    print(f'  Research value: {pharma_insights[\"total_research_value\"]:.1f}')

# Generate monthly report
report_path = tracker.export_cost_report(days_back=30)
print(f'  Detailed report: {report_path}')
"

# 3. Configuration audit
echo -e "\n3. Configuration Audit:"
python -c "
from src.enhanced_config import EnhancedRAGConfig
config = EnhancedRAGConfig.from_env()

print('Configuration status:')
strategy = config.get_cloud_first_strategy()
print(f'  Cloud-first: {strategy[\"cloud_first_enabled\"]}')

compatibility = config.validate_openai_sdk_compatibility()
print(f'  OpenAI SDK compatible: {compatibility[\"compatible\"]}')

feature_flags = config.get_feature_flags()
enabled_flags = sum(1 for flag in feature_flags.values() if flag)
print(f'  Feature flags: {enabled_flags}/{len(feature_flags)} enabled')
"

# 4. System optimization recommendations
echo -e "\n4. Optimization Recommendations:"
python -c "
# Generate optimization recommendations based on usage patterns
from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker

tracker = create_pharmaceutical_cost_tracker()
analysis = tracker.get_cost_analysis(days_back=30)

recommendations = analysis.get('recommendations', [])
if recommendations:
    print('Monthly optimization recommendations:')
    for i, rec in enumerate(recommendations[:5], 1):
        print(f'  {i}. {rec[\"title\"]}')
        print(f'     {rec[\"description\"]}')
        if 'potential_savings_usd' in rec:
            print(f'     Potential savings: \${rec[\"potential_savings_usd\"]:.3f}')
else:
    print('✅ No specific optimization recommendations')
"

echo -e "\n✅ Monthly maintenance completed: $(date)"
```

---

## Recovery Procedures

### System Recovery Checklist

When the system is not functioning properly:

1. **Immediate Assessment:**

   ```bash
   # Quick system health check
   ./scripts/system_health_check.sh
   ```

2. **API Connectivity Recovery:**

   ```bash
   # Test and recover API connectivity
   python -c "
   from src.clients.openai_wrapper import create_nvidia_build_client
   try:
       client = create_nvidia_build_client()
       result = client.test_connection()
       if result['success']:
           print('✅ API connectivity restored')
       else:
           print('❌ API still not accessible')
   except Exception as e:
       print(f'❌ API recovery failed: {e}')
   "
   ```

3. **Configuration Recovery:**

   ```bash
   # Restore from backup if needed
   ./scripts/restore_config.sh latest_backup.tar.gz
   ```

4. **Clear System State:**

   ```bash
   # Clear potentially corrupted cache/state
   rm -rf __pycache__/
   rm -rf src/__pycache__/
   rm -rf .pytest_cache/

   # Clear batch processing state
   python -c "
   from src.optimization.batch_processor import create_pharmaceutical_batch_processor
   processor = create_pharmaceutical_batch_processor()
   processor.clear_queues()
   print('✅ Batch queues cleared')
   "
   ```

5. **Restart Services:**
   ```bash
   # Restart monitoring services if running
   sudo systemctl restart pharma-health-monitor 2>/dev/null || echo "No health monitor service"
   ```

### Emergency Procedures

#### Critical System Failure

```bash
#!/bin/bash
# Emergency recovery script

echo "🚨 EMERGENCY PHARMACEUTICAL RAG RECOVERY"
echo "========================================"

# 1. Backup current state
echo "1. Backing up current state..."
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p "emergency_backup_$timestamp"
cp -r logs/ "emergency_backup_$timestamp/" 2>/dev/null
cp .env "emergency_backup_$timestamp/" 2>/dev/null
cp metrics.db "emergency_backup_$timestamp/" 2>/dev/null

# 2. Stop all processes
echo "2. Stopping processes..."
pkill -f "python.*pharmaceutical" 2>/dev/null

# 3. Reset to known good state
echo "3. Resetting to clean state..."
python -c "
import sys
sys.path.append('src')

# Clear all caches and temporary files
import shutil
import os

for cache_dir in ['__pycache__', '.pytest_cache', 'query_cache', 'tmp_cache']:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f'Cleared {cache_dir}')

# Reset batch processing
try:
    from optimization.batch_processor import create_pharmaceutical_batch_processor
    processor = create_pharmaceutical_batch_processor()
    processor.clear_queues()
    print('Batch queues cleared')
except:
    print('Batch queue clearing failed')
"

# 4. Validate critical components
echo "4. Validating critical components..."
python -c "
# Test minimum functionality
import os
if os.getenv('NVIDIA_API_KEY'):
    print('✅ API key available')
else:
    print('❌ API key missing - system cannot function')
    exit(1)

try:
    from src.clients.openai_wrapper import create_nvidia_build_client
    client = create_nvidia_build_client()
    result = client.test_connection()
    if result['success']:
        print('✅ Basic API connectivity working')
    else:
        print('❌ API not responding')
        exit(1)
except Exception as e:
    print(f'❌ Client creation failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Emergency recovery successful"
    echo "System restored to basic functionality"
else
    echo "❌ Emergency recovery failed"
    echo "Manual intervention required"
    exit 1
fi
```

### Data Recovery

#### Recovering Lost Configuration

```python
def recover_lost_configuration():
    """Recover configuration from backup or defaults."""

    import os
    import shutil
    from datetime import datetime

    print("🔧 Configuration Recovery")
    print("=" * 25)

    # Check for backup files
    backup_files = [
        ".env.backup",
        "config/alerts.yaml.backup",
        ".encryption_key.backup"
    ]

    recovered_files = []

    for backup_file in backup_files:
        if os.path.exists(backup_file):
            original_file = backup_file.replace(".backup", "")

            # Create backup of current (potentially corrupted) file
            if os.path.exists(original_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy(original_file, f"{original_file}.corrupted_{timestamp}")

            # Restore from backup
            shutil.copy(backup_file, original_file)
            recovered_files.append(original_file)
            print(f"✅ Recovered: {original_file}")

    if not recovered_files:
        print("⚠️  No backup files found")
        print("Creating default configuration...")

        # Create minimal .env
        default_env = """# Minimal configuration for recovery
NVIDIA_API_KEY=your_api_key_here
NVIDIA_BUILD_BASE_URL=https://integrate.api.nvidia.com/v1
ENABLE_CLOUD_FIRST_STRATEGY=true
ENABLE_PHARMACEUTICAL_OPTIMIZATION=true
ENABLE_BATCH_OPTIMIZATION=true
ENABLE_SAFETY_MONITORING=true
LOG_LEVEL=INFO
"""

        with open(".env", "w") as f:
            f.write(default_env)

        print("✅ Created default .env file")
        print("⚠️  Please update NVIDIA_API_KEY with your actual key")

    return recovered_files

# Run recovery if needed
if __name__ == "__main__":
    recover_lost_configuration()
```

---

## Support and Resources

### Log Files and Debugging

**Important Log Locations:**

- `logs/application.log` - General application logs
- `logs/pharmaceutical_alerts.log` - Safety alert logs
- `logs/cost_tracking.log` - Cost monitoring logs
- `logs/batch_processing.log` - Batch operation logs
- `logs/compliance_audit.log` - Compliance audit trail

**Debug Mode Activation:**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export ENABLE_DEBUG_LOGGING=true

# Run with verbose output
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from src.clients.nemo_client_enhanced import create_pharmaceutical_client
client = create_pharmaceutical_client()
# Debug output will be printed
"
```

### Getting Help

**Internal Resources:**

1. Check this troubleshooting guide
2. Review logs for error patterns
3. Run diagnostic scripts
4. Check configuration validity

**External Resources:**

1. NVIDIA Developer Documentation
2. OpenAI SDK Documentation
3. Python/AsyncIO Documentation
4. System-specific documentation

**Escalation Path:**

1. Run comprehensive diagnostics
2. Collect relevant logs
3. Document error reproduction steps
4. Contact system administrator
5. Engage technical support if needed

---

**Document Version**: 1.0.0
**Last Updated**: September 24, 2025
**Maintained By**: Pharmaceutical RAG Team
**Emergency Contact**: System Administrator
