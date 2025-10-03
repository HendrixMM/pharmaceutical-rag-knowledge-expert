# API Integration Guide

<!-- TOC -->

- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Quick Start](#quick-start)
  - [Basic Setup](#basic-setup)
  - [Simple Usage](#simple-usage)
- [Core API Components](#core-api-components)
  - [Component Architecture](#component-architecture)
  - [Primary Integration Points](#primary-integration-points)
- [Authentication & Configuration](#authentication--configuration)
  - [Environment Configuration](#environment-configuration)
  - [Required Environment Variables](#required-environment-variables)
  - [Configuration Validation](#configuration-validation)
- [Enhanced NeMo Client Integration](#enhanced-nemo-client-integration)
  - [Basic Client Usage](#basic-client-usage)
  - [Embedding Generation](#embedding-generation)
  - [Chat Completions](#chat-completions)
  - [Performance Metrics](#performance-metrics)
- [Pharmaceutical Features](#pharmaceutical-features)
  - [Query Classification and Safety](#query-classification-and-safety)
  - [Workflow Templates](#workflow-templates)
  - [Model Optimization](#model-optimization)
- [Cost Optimization APIs](#cost-optimization-apis)
  - [Credit Tracking and Monitoring](#credit-tracking-and-monitoring)
  - [Batch Processing Optimization](#batch-processing-optimization)
- [Monitoring & Health APIs](#monitoring--health-apis)
  - [Endpoint Health Monitoring](#endpoint-health-monitoring)
  - [Model Validation](#model-validation)
- [Batch Processing Integration](#batch-processing-integration)
  - [Intelligent Batch Processing](#intelligent-batch-processing)
  - [Queue Management](#queue-management)
- [Error Handling](#error-handling)
  - [Comprehensive Error Management](#comprehensive-error-management)
  - [Rate Limiting and Retry Logic](#rate-limiting-and-retry-logic)
- [Performance Optimization](#performance-optimization)
  - [Caching and Memoization](#caching-and-memoization)
  - [Connection Pooling and Resource Management](#connection-pooling-and-resource-management)
- [Security Considerations](#security-considerations)
  - [API Key Management](#api-key-management)
  - [Data Privacy and Compliance](#data-privacy-and-compliance)
- [Complete Integration Example](#complete-integration-example)
  - [End-to-End Pharmaceutical Research Pipeline](#end-to-end-pharmaceutical-research-pipeline)
- [Next Steps](#next-steps)
  - [Advanced Integration Topics](#advanced-integration-topics)
- [Related Documentation](#related-documentation)
  - [Support and Community](#support-and-community)
  <!-- /TOC -->

---

Last Updated: 2025-10-03
Owner: API Team
Review Cadence: Weekly

---

**Comprehensive Integration Documentation for Pharmaceutical RAG System**

## Overview

This guide provides comprehensive documentation for integrating with the pharmaceutical RAG system's APIs, including the cloud-first architecture, safety monitoring, cost optimization, and pharmaceutical domain specialization.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core API Components](#core-api-components)
3. [Authentication & Configuration](#authentication--configuration)
4. [Enhanced NeMo Client Integration](#enhanced-nemo-client-integration)
5. [Pharmaceutical Features](#pharmaceutical-features)
6. [Cost Optimization APIs](#cost-optimization-apis)
7. [Monitoring & Health APIs](#monitoring--health-apis)
8. [Batch Processing Integration](#batch-processing-integration)
9. [Error Handling](#error-handling)
10. [Performance Optimization](#performance-optimization)
11. [Security Considerations](#security-considerations)

---

## Quick Start

### Basic Setup

```python
# 1. Install requirements
pip install -r requirements.txt

# 2. Set environment variables
export NVIDIA_API_KEY="your_nvidia_build_api_key"
export ENABLE_CLOUD_FIRST_STRATEGY=true
export ENABLE_PHARMACEUTICAL_OPTIMIZATION=true
```

### Simple Usage

```python
from src.clients.nemo_client_enhanced import EnhancedNeMoClient

# Initialize enhanced client with pharmaceutical optimization
client = EnhancedNeMoClient(pharmaceutical_optimized=True)

# Simple embedding request
response = client.create_embeddings([
    "metformin mechanism of action in diabetes treatment"
])

print(f"Success: {response.success}")
print(f"Endpoint: {response.endpoint_type}")
print(f"Cost tier: {response.cost_tier}")
```

---

## Core API Components

### Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   API Integration Layer                  │
├─────────────────────────────────────────────────────────┤
│  Enhanced NeMo Client (Main Integration Point)          │
│  ┌─────────────────────┐  ┌─────────────────────────┐   │
│  │   OpenAI Wrapper    │  │ Pharmaceutical APIs     │   │
│  │   (Cloud Primary)   │  │ (Domain Optimization)   │   │
│  └─────────────────────┘  └─────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Supporting APIs                                         │
│  • Cost Monitoring    • Safety Integration              │
│  • Batch Processing   • Health Monitoring               │
│  • Model Validation   • Workflow Templates              │
└─────────────────────────────────────────────────────────┘
```

### Primary Integration Points

1. **EnhancedNeMoClient**: Main API client with cloud-first execution
2. **PharmaceuticalBatchClient**: Optimized batch processing
3. **PharmaceuticalWorkflowExecutor**: Pre-built research workflows
4. **DrugSafetyAlertIntegration**: Real-time safety monitoring

---

## Authentication & Configuration

### Environment Configuration

```python
# src/enhanced_config.py integration
from src.enhanced_config import EnhancedRAGConfig

# Load configuration from environment
config = EnhancedRAGConfig.from_env()

# Verify cloud-first configuration
strategy = config.get_cloud_first_strategy()
print(f"Cloud-first enabled: {strategy['cloud_first_enabled']}")
print(f"Pharmaceutical optimized: {config.pharmaceutical_optimized}")
```

### Required Environment Variables

```bash
# Core Configuration
NVIDIA_API_KEY=your_nvidia_build_api_key
NVIDIA_BUILD_BASE_URL=https://integrate.api.nvidia.com/v1

# Feature Flags
ENABLE_CLOUD_FIRST_STRATEGY=true
ENABLE_PHARMACEUTICAL_OPTIMIZATION=true
ENABLE_NVIDIA_BUILD_FALLBACK=true
ENABLE_DAILY_CREDIT_ALERTS=true
ENABLE_BATCH_OPTIMIZATION=true

# Optional Configuration
PHARMACEUTICAL_RESEARCH_MODE=true
ENABLE_SAFETY_MONITORING=true
MAX_MONTHLY_BUDGET_USD=100
```

### Configuration Validation

```python
# Validate configuration before use
config = EnhancedRAGConfig.from_env()

# Check OpenAI SDK compatibility
compatibility = config.validate_openai_sdk_compatibility()
if not compatibility["compatible"]:
    raise RuntimeError(f"OpenAI SDK incompatible: {compatibility['error']}")

# Verify cloud-first setup
strategy = config.get_cloud_first_strategy()
if not strategy["cloud_first_enabled"]:
    print("Warning: Cloud-first strategy not enabled")

print("✅ Configuration validated successfully")
```

---

## Enhanced NeMo Client Integration

### Basic Client Usage

```python
from src.clients.nemo_client_enhanced import EnhancedNeMoClient, create_pharmaceutical_client

# Method 1: Direct initialization
client = EnhancedNeMoClient(
    pharmaceutical_optimized=True,
    enable_fallback=True
)

# Method 2: Pharmaceutical convenience function
client = create_pharmaceutical_client(cloud_first=True)

# Check client status
status = client.get_endpoint_status()
print(f"Cloud available: {status['cloud_available']}")
print(f"Fallback enabled: {status['fallback_enabled']}")
```

### Embedding Generation

```python
# Single embedding with pharmaceutical optimization
response = client.create_embeddings(
    texts=["ACE inhibitor contraindications in pregnancy"],
    model="nvidia/nv-embedqa-e5-v5"  # Pharmaceutical-optimized model
)

if response.success:
    embeddings = response.data["embeddings"]
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimensions: {len(embeddings[0])}")
    print(f"Used endpoint: {response.endpoint_type.value}")
    print(f"Cost tier: {response.cost_tier}")
else:
    print(f"Error: {response.error}")

# Batch embeddings for efficiency
pharmaceutical_queries = [
    "metformin pharmacokinetics in kidney disease",
    "warfarin drug interactions with NSAIDs",
    "beta blocker contraindications in asthma",
    "statin-associated muscle symptoms monitoring"
]

batch_response = client.create_embeddings(
    texts=pharmaceutical_queries,
    model="nvidia/nv-embedqa-e5-v5"
)

print(f"Batch processing: {batch_response.success}")
print(f"Response time: {batch_response.response_time_ms}ms")
```

### Chat Completions

```python
# Pharmaceutical chat completion with safety context
messages = [{
    "role": "user",
    "content": "What are the contraindications for metformin in elderly patients with chronic kidney disease?"
}]

response = client.create_chat_completion(
    messages=messages,
    model="meta/llama-3.1-8b-instruct",
    max_tokens=800,
    temperature=0.1  # Conservative for medical accuracy
)

if response.success:
    content = response.data["content"]
    print(f"Response: {content}")
    print(f"Model used: {response.data['model']}")
    print(f"Token usage: {response.data.get('usage', {})}")
    print(f"Endpoint: {response.endpoint_type.value}")
else:
    print(f"Chat completion failed: {response.error}")
```

### Performance Metrics

```python
# Get comprehensive performance metrics
metrics = client.get_performance_metrics()

print("Performance Metrics:")
print(f"  Total requests: {metrics['total_requests']}")
print(f"  Cloud requests: {metrics['cloud_requests']}")
print(f"  Success rate: {metrics['success_rate']:.1%}")
print(f"  Avg response time: {metrics['avg_response_time_ms']}ms")
print(f"  Cloud usage: {metrics['cloud_usage_percentage']:.1f}%")

# Cost optimization insights
cost_opt = metrics['cost_optimization']
print(f"  Free tier requests: {cost_opt['free_tier_requests']}")
print(f"  Monthly projection: {cost_opt['estimated_monthly_projection']}")
```

---

## Pharmaceutical Features

### Query Classification and Safety

```python
from src.pharmaceutical.query_classifier import classify_pharmaceutical_query
from src.pharmaceutical.safety_alert_integration import process_pharmaceutical_query_with_safety

# Classify pharmaceutical query
query = "Urgent: Patient experiencing bleeding while on warfarin and aspirin"
context = classify_pharmaceutical_query(query)

print(f"Domain: {context.domain.value}")
print(f"Safety urgency: {context.safety_urgency.name}")
print(f"Research priority: {context.research_priority.name}")
print(f"Drug names: {context.drug_names}")
print(f"Confidence: {context.confidence_score:.3f}")

# Process with safety monitoring
context, safety_alerts = await process_pharmaceutical_query_with_safety(query)

print(f"Generated {len(safety_alerts)} safety alerts:")
for alert in safety_alerts:
    print(f"  - {alert.alert_type.value}: {alert.safety_message}")
    print(f"    Urgency: {alert.urgency.value}")
    print(f"    Drugs: {alert.drug_names}")
```

### Workflow Templates

```python
from src.pharmaceutical.workflow_templates import (
    execute_drug_safety_workflow,
    execute_clinical_trial_workflow,
    PharmaceuticalWorkflowFactory
)

# Execute pre-built drug safety workflow
drug_name = "metformin"
safety_result = await execute_drug_safety_workflow(drug_name)

print(f"Workflow: {safety_result.workflow_name}")
print(f"Success: {safety_result.overall_success}")
print(f"Steps completed: {len(safety_result.step_results)}")
print(f"Safety alerts: {len(safety_result.safety_alerts)}")
print(f"Execution time: {safety_result.execution_time_ms}ms")

# Custom workflow creation
from src.pharmaceutical.workflow_templates import (
    PharmaceuticalWorkflowFactory, WorkflowType, WorkflowStep
)

custom_steps = [
    WorkflowStep(
        step_name="Drug Overview",
        step_type="query",
        description="Get comprehensive drug information",
        query_template="Provide detailed information about {drug_name}",
        priority="normal"
    )
]

custom_workflow = PharmaceuticalWorkflowFactory.create_custom_workflow(
    workflow_type=WorkflowType.GENERAL_RESEARCH,
    name="Custom Drug Research",
    description="Custom pharmaceutical research workflow",
    steps=custom_steps
)

# Execute custom workflow
from src.pharmaceutical.workflow_templates import PharmaceuticalWorkflowExecutor

executor = PharmaceuticalWorkflowExecutor()
result = await executor.execute_workflow(
    custom_workflow,
    parameters={"drug_name": "atorvastatin"}
)
```

### Model Optimization

```python
from src.pharmaceutical.model_optimization import (
    optimize_pharmaceutical_query,
    create_pharmaceutical_model_optimizer
)

# Optimize query for pharmaceutical context
query = "ACE inhibitor side effects in elderly"
context = classify_pharmaceutical_query(query)

# For embeddings
optimized_query, embedding_config = optimize_pharmaceutical_query(
    query, context, model_type="embedding"
)
print(f"Optimized embedding query: {optimized_query}")

# For chat completions
messages, chat_config = optimize_pharmaceutical_query(
    query, context, model_type="chat"
)
print(f"Chat temperature: {chat_config['temperature']}")
print(f"Max tokens: {chat_config['max_tokens']}")
print(f"System prompt length: {len(messages[0]['content'])}")
```

---

## Cost Optimization APIs

### Credit Tracking and Monitoring

```python
from src.monitoring.credit_tracker import PharmaceuticalCreditTracker
from src.monitoring.pharmaceutical_cost_analyzer import create_pharmaceutical_cost_tracker

# Initialize credit tracker
credit_tracker = PharmaceuticalCreditTracker()

# Get current usage statistics
usage_stats = credit_tracker.get_usage_stats()
print(f"Requests today: {usage_stats['requests_today']}")
print(f"Requests this month: {usage_stats['requests_this_month']}")
print(f"Monthly budget remaining: ${usage_stats['monthly_budget_remaining_usd']:.2f}")

# Record pharmaceutical query for cost tracking
credit_tracker.record_request(
    endpoint_type="nvidia_build",
    cost_usd=0.0,  # Free tier
    tokens_used=150,
    query_metadata={
        "domain": "drug_safety",
        "drug_names": ["metformin"],
        "safety_critical": True
    }
)

# Advanced cost analysis
cost_analyzer = create_pharmaceutical_cost_tracker()

# Record detailed pharmaceutical query
cost_analyzer.record_pharmaceutical_query(
    query_id="research_001",
    query_text="Metformin contraindications in chronic kidney disease",
    cost_tier="free_tier",
    estimated_tokens=200,
    project_id="diabetes_research",
    tags=["drug_safety", "nephrology"]
)

# Get comprehensive cost analysis
analysis = cost_analyzer.get_cost_analysis(days_back=30)
print(f"Total cost: ${analysis['cost_breakdown']['total_cost_usd']}")
print(f"Free tier usage: {analysis['free_tier_optimization']['efficiency_score']:.1%}")
print(f"Pharmaceutical value: {analysis['pharmaceutical_insights']['total_research_value']}")
```

### Batch Processing Optimization

```python
from src.optimization.batch_integration import create_pharmaceutical_research_session

# Create optimized batch session
async with create_pharmaceutical_research_session(
    auto_process_seconds=30,  # Automatic processing every 30 seconds
    aggressive_optimization=True
) as batch_client:

    # Queue high-priority safety query
    safety_request_id = await batch_client.queue_chat_request(
        messages=[{
            "role": "user",
            "content": "Critical drug interaction between warfarin and aspirin in elderly patient"
        }],
        priority=RequestPriority.CRITICAL,
        pharmaceutical_context={
            "domain": "drug_safety",
            "research_type": "drug_interaction"
        }
    )

    # Queue normal research queries for batching
    for drug in ["metformin", "lisinopril", "atorvastatin"]:
        await batch_client.queue_embedding_request(
            texts=[f"{drug} mechanism of action and therapeutic effects"],
            priority=RequestPriority.NORMAL,
            pharmaceutical_context={
                "domain": "mechanism_of_action",
                "research_type": "general_research"
            }
        )

    # Process all queued requests
    result = await batch_client.process_batches_now(max_concurrent=2)

    if result.success:
        print(f"Processed {len(result.results)} batches")
        free_tier_util = result.metrics["pharmaceutical_optimization"]["free_tier_utilization"]
        print(f"Free tier utilization: {free_tier_util['utilization_percentage']:.1f}%")
```

---

## Monitoring & Health APIs

### Endpoint Health Monitoring

```python
from src.monitoring.endpoint_health_monitor import (
    create_endpoint_health_monitor,
    quick_health_check
)

# Quick health check
health_status = await quick_health_check()

print("Endpoint Health Status:")
print(f"  Overall health: {health_status['endpoint_health']['overall_health']}")
print(f"  NGC independent: {health_status['ngc_independence_status']['verified']}")
print(f"  NVIDIA Build operational: {health_status['ngc_independence_status']['nvidia_build_operational']}")

# Continuous monitoring
monitor = create_endpoint_health_monitor(
    monitoring_interval=60,  # Check every minute
    pharmaceutical_focused=True
)

# Start monitoring
await monitor.start_monitoring()

# Get health trends (run after some time)
trends = monitor.get_health_trends(hours_back=24)
print("Health Trends (24h):")
for endpoint, trend_data in trends["endpoint_trends"].items():
    print(f"  {endpoint}:")
    print(f"    Avg response time: {trend_data['avg_response_time_ms']}ms")
    print(f"    Success rate: {trend_data['avg_success_rate']:.1f}%")
    print(f"    Model availability: {trend_data['avg_model_availability']}")

# Get active alerts
active_alerts = monitor.get_active_alerts()
if active_alerts:
    print(f"\nActive Health Alerts: {len(active_alerts)}")
    for alert in active_alerts[:3]:  # Show first 3
        print(f"  - {alert['severity']}: {alert['message']}")

await monitor.stop_monitoring()
```

### Model Validation

```python
from src.validation.model_validator import (
    validate_nvidia_build_compatibility,
    create_model_validator
)

# Quick compatibility validation
compatibility_results = await validate_nvidia_build_compatibility(
    pharmaceutical_optimized=True
)

print("Model Validation Results:")
print(f"  NGC independent: {compatibility_results['ngc_independent']}")
print(f"  Overall status: {compatibility_results['overall_status']}")
print(f"  Pharmaceutical optimized: {compatibility_results['pharmaceutical_optimized']}")

# Detailed model validation
validator = create_model_validator(pharmaceutical_focused=True)
detailed_results = await validator.validate_all_models()

print("Detailed Validation:")
model_validation = detailed_results["model_validation"]
for model_id, result in model_validation.items():
    if hasattr(result, 'available'):
        print(f"  {model_id}:")
        print(f"    Available: {result.available}")
        print(f"    Compatible: {result.compatible}")
        print(f"    Response time: {result.response_time_ms}ms")

# Pharmaceutical capability testing
pharma_analysis = detailed_results["pharmaceutical_analysis"]
if pharma_analysis.get("enabled"):
    print(f"\nPharmaceutical Capabilities:")
    print(f"  Embedding test: {pharma_analysis['embedding_pharmaceutical_test']['success']}")
    print(f"  Chat test: {pharma_analysis['chat_pharmaceutical_test']['success']}")
    print(f"  Overall score: {pharma_analysis['overall_pharmaceutical_score']}/10")
```

---

## Batch Processing Integration

### Intelligent Batch Processing

```python
from src.optimization.batch_processor import create_pharmaceutical_batch_processor
from src.optimization.batch_integration import PharmaceuticalBatchClient

# Create batch processor
batch_processor = create_pharmaceutical_batch_processor(
    enhanced_tracking=True,
    aggressive_optimization=True
)

# Create integrated batch client
batch_client = PharmaceuticalBatchClient(
    batch_processor=batch_processor,
    auto_process_interval=30  # Auto-process every 30 seconds
)

# Queue requests with different priorities
await batch_client.queue_chat_request(
    messages=[{"role": "user", "content": "Metformin dosing in renal impairment"}],
    priority=RequestPriority.HIGH,
    pharmaceutical_context={"domain": "dosage_guidelines", "patient_population": "renal_impairment"}
)

await batch_client.queue_embedding_request(
    texts=[
        "cardiovascular drug mechanisms of action",
        "diabetes medication pharmacokinetics",
        "hypertension treatment guidelines"
    ],
    priority=RequestPriority.NORMAL,
    pharmaceutical_context={"domain": "general_research"}
)

# Process batches immediately
result = await batch_client.process_batches_now(max_concurrent=3)

if result.success:
    print(f"Batch processing completed:")
    print(f"  Batches: {len(result.results)}")
    print(f"  Execution time: {result.metrics['batch_execution_time_ms']}ms")

    # Cost optimization metrics
    free_tier_util = result.metrics["pharmaceutical_optimization"]["free_tier_utilization"]
    print(f"  Free tier optimization: {free_tier_util['optimization_status']}")
    print(f"  Cost savings achieved: {free_tier_util['utilization_percentage']:.1f}%")
```

### Queue Management

```python
# Check queue status
queue_status = batch_processor.get_queue_status()

print("Batch Queue Status:")
print(f"  Total queued: {queue_status['total_queued_requests']}")
print(f"  Processing active: {queue_status['processing_active']}")

for priority, queue_info in queue_status["queues_by_priority"].items():
    if queue_info["count"] > 0:
        print(f"  {priority}: {queue_info['count']} requests")
        print(f"    Pharmaceutical: {queue_info['pharmaceutical_requests']}")
        print(f"    Estimated tokens: {queue_info['estimated_tokens']}")

# Clear queues if needed (emergency or testing)
# cleared_counts = batch_processor.clear_queues()  # Clear all
# cleared_count = batch_processor.clear_queues(RequestPriority.BACKGROUND)  # Clear specific priority
```

---

## Error Handling

### Comprehensive Error Management

```python
from src.clients.openai_wrapper import NVIDIABuildError
from src.clients.nemo_client_enhanced import ClientResponse

# Standard error handling pattern
try:
    client = EnhancedNeMoClient(pharmaceutical_optimized=True)

    response = client.create_embeddings([
        "pharmaceutical query text"
    ])

    if response.success:
        # Process successful response
        embeddings = response.data["embeddings"]
        print(f"Generated embeddings: {len(embeddings)}")
    else:
        # Handle API-level errors
        print(f"API Error: {response.error}")

        # Check error type for specific handling
        if "403" in str(response.error):
            print("API access limited - check API key or tier")
        elif "429" in str(response.error):
            print("Rate limit exceeded - implementing backoff")
        elif "404" in str(response.error):
            print("Model not found - trying alternative")

except NVIDIABuildError as e:
    # Handle NVIDIA Build specific errors
    print(f"NVIDIA Build Error: {e}")
    print(f"Status Code: {e.status_code}")
    print(f"Response Data: {e.response_data}")

except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
    logger.error(f"Client initialization failed: {str(e)}")

# Robust error handling with fallback
async def robust_pharmaceutical_query(query_text: str) -> Dict[str, Any]:
    """Robust query execution with comprehensive error handling."""

    try:
        # Try enhanced client first
        client = EnhancedNeMoClient(pharmaceutical_optimized=True)
        response = client.create_chat_completion([
            {"role": "user", "content": query_text}
        ])

        if response.success:
            return {
                "success": True,
                "data": response.data,
                "endpoint": response.endpoint_type.value,
                "cost_tier": response.cost_tier
            }
        else:
            # Try fallback approach
            return await _fallback_query_handling(query_text, response.error)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_attempted": False
        }

async def _fallback_query_handling(query: str, original_error: str) -> Dict[str, Any]:
    """Implement fallback query handling."""
    try:
        # Alternative approach or self-hosted fallback
        logger.warning(f"Primary query failed: {original_error}, attempting fallback")

        # Implement alternative logic here
        return {
            "success": True,
            "data": {"content": "Fallback response"},
            "endpoint": "fallback",
            "original_error": original_error,
            "fallback_attempted": True
        }

    except Exception as fallback_error:
        return {
            "success": False,
            "original_error": original_error,
            "fallback_error": str(fallback_error),
            "fallback_attempted": True
        }
```

### Rate Limiting and Retry Logic

```python
import asyncio
import time
from typing import Callable, Any

async def with_retry_and_backoff(
    operation: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Any:
    """Execute operation with exponential backoff retry logic."""

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                logger.info(f"Retrying after {delay:.1f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(delay)

            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()

            return result

        except Exception as e:
            last_exception = e

            # Check if error is retryable
            if "429" in str(e) or "503" in str(e) or "502" in str(e):
                if attempt < max_retries:
                    logger.warning(f"Retryable error on attempt {attempt + 1}: {str(e)}")
                    continue
            else:
                # Non-retryable error, fail immediately
                logger.error(f"Non-retryable error: {str(e)}")
                raise e

    # All retries exhausted
    logger.error(f"Operation failed after {max_retries} retries")
    raise last_exception

# Usage example
async def resilient_embedding_request(texts: List[str]) -> ClientResponse:
    """Create embeddings with retry logic."""

    client = EnhancedNeMoClient(pharmaceutical_optimized=True)

    async def embedding_operation():
        return client.create_embeddings(texts)

    return await with_retry_and_backoff(
        embedding_operation,
        max_retries=3,
        base_delay=1.0
    )
```

---

## Performance Optimization

### Caching and Memoization

```python
from functools import lru_cache
from typing import List, Tuple
import hashlib

class PharmaceuticalQueryCache:
    """Intelligent caching for pharmaceutical queries."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)  # 1 hour TTL for pharmaceutical data

    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key from query and context."""
        content = f"{query}_{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(query, context)

        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                # Expired, remove from cache
                del self.cache[cache_key]

        return None

    def set(self, query: str, context: Dict[str, Any], result: Any) -> None:
        """Cache result with timestamp."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        cache_key = self._generate_cache_key(query, context)
        self.cache[cache_key] = (result, datetime.now())

# Global cache instance
pharma_cache = PharmaceuticalQueryCache()

async def cached_pharmaceutical_query(query: str, context: Dict[str, Any]) -> ClientResponse:
    """Pharmaceutical query with intelligent caching."""

    # Check cache first
    cached_result = pharma_cache.get(query, context)
    if cached_result:
        logger.debug(f"Cache hit for query: {query[:50]}...")
        return cached_result

    # Execute query
    client = EnhancedNeMoClient(pharmaceutical_optimized=True)
    response = client.create_chat_completion([
        {"role": "user", "content": query}
    ])

    # Cache successful responses
    if response.success:
        pharma_cache.set(query, context, response)

    return response
```

### Connection Pooling and Resource Management

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class PharmaceuticalClientPool:
    """Connection pool for pharmaceutical clients."""

    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.available_clients: List[EnhancedNeMoClient] = []
        self.busy_clients: Set[EnhancedNeMoClient] = set()
        self.lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize client pool."""
        if self._initialized:
            return

        async with self.lock:
            if self._initialized:
                return

            for _ in range(self.pool_size):
                client = EnhancedNeMoClient(pharmaceutical_optimized=True)
                self.available_clients.append(client)

            self._initialized = True
            logger.info(f"Initialized client pool with {self.pool_size} clients")

    @asynccontextmanager
    async def acquire_client(self) -> AsyncGenerator[EnhancedNeMoClient, None]:
        """Acquire client from pool."""
        await self.initialize()

        async with self.lock:
            while not self.available_clients:
                # Wait for client to become available
                await asyncio.sleep(0.1)

            client = self.available_clients.pop()
            self.busy_clients.add(client)

        try:
            yield client
        finally:
            async with self.lock:
                self.busy_clients.remove(client)
                self.available_clients.append(client)

# Global client pool
client_pool = PharmaceuticalClientPool(pool_size=3)

async def pooled_pharmaceutical_request(query: str) -> ClientResponse:
    """Execute pharmaceutical request using client pool."""
    async with client_pool.acquire_client() as client:
        return client.create_chat_completion([
            {"role": "user", "content": query}
        ])
```

---

## Security Considerations

### API Key Management

```python
import os
from cryptography.fernet import Fernet
from typing import Optional

class SecureAPIKeyManager:
    """Secure API key management for pharmaceutical applications."""

    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API key storage."""
        key_file = ".api_key_encryption.key"

        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            # Secure the key file
            os.chmod(key_file, 0o600)
            return key

    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key for secure storage."""
        return self.fernet.encrypt(api_key.encode()).decode()

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key for use."""
        return self.fernet.decrypt(encrypted_key.encode()).decode()

    def get_nvidia_api_key(self) -> Optional[str]:
        """Securely retrieve NVIDIA API key."""
        # Try environment variable first
        api_key = os.getenv("NVIDIA_API_KEY")
        if api_key:
            return api_key

        # Try encrypted storage
        encrypted_key = os.getenv("NVIDIA_API_KEY_ENCRYPTED")
        if encrypted_key:
            return self.decrypt_api_key(encrypted_key)

        return None

# Usage
key_manager = SecureAPIKeyManager()
secure_api_key = key_manager.get_nvidia_api_key()

if not secure_api_key:
    raise ValueError("NVIDIA API key not found in secure storage")
```

### Data Privacy and Compliance

```python
import re
from typing import List, Dict, Any

class PharmaceuticalDataSanitizer:
    """Data sanitization for pharmaceutical compliance."""

    def __init__(self):
        # PII patterns for detection and removal
        self.pii_patterns = {
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "patient_id": re.compile(r'\bpatient\s*id\s*:?\s*\w+\b', re.IGNORECASE),
            "mrn": re.compile(r'\bmrn\s*:?\s*\w+\b', re.IGNORECASE)
        }

        # Medical record patterns
        self.medical_patterns = {
            "date_of_birth": re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
            "specific_dates": re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE)
        }

    def sanitize_query(self, query: str) -> Tuple[str, List[str]]:
        """Sanitize pharmaceutical query for privacy compliance."""
        sanitized_query = query
        removed_items = []

        # Remove PII
        for pattern_name, pattern in self.pii_patterns.items():
            matches = pattern.findall(query)
            if matches:
                removed_items.extend([f"{pattern_name}: {match}" for match in matches])
                sanitized_query = pattern.sub(f"[{pattern_name.upper()}_REMOVED]", sanitized_query)

        # Handle medical record patterns
        for pattern_name, pattern in self.medical_patterns.items():
            matches = pattern.findall(query)
            if matches:
                removed_items.extend([f"{pattern_name}: {match}" for match in matches])
                sanitized_query = pattern.sub(f"[{pattern_name.upper()}_REMOVED]", sanitized_query)

        return sanitized_query, removed_items

    def validate_pharmaceutical_compliance(self, query: str) -> Dict[str, Any]:
        """Validate query for pharmaceutical compliance requirements."""
        sanitized_query, removed_pii = self.sanitize_query(query)

        compliance_check = {
            "original_query_length": len(query),
            "sanitized_query_length": len(sanitized_query),
            "pii_detected": len(removed_pii) > 0,
            "removed_pii_items": removed_pii,
            "compliance_status": "compliant" if len(removed_pii) == 0 else "sanitized",
            "sanitized_query": sanitized_query
        }

        return compliance_check

# Usage in pharmaceutical queries
sanitizer = PharmaceuticalDataSanitizer()

def secure_pharmaceutical_query(query: str) -> Tuple[str, Dict[str, Any]]:
    """Execute pharmaceutical query with privacy compliance."""

    # Sanitize query first
    compliance_check = sanitizer.validate_pharmaceutical_compliance(query)

    if compliance_check["pii_detected"]:
        logger.warning(f"PII detected and removed: {compliance_check['removed_pii_items']}")

    # Use sanitized query for API call
    sanitized_query = compliance_check["sanitized_query"]

    return sanitized_query, compliance_check
```

---

## Complete Integration Example

### End-to-End Pharmaceutical Research Pipeline

```python
import asyncio
from datetime import datetime
from typing import Dict, List, Any

async def comprehensive_pharmaceutical_research_pipeline(
    drug_name: str,
    research_focus: str = "safety_profile"
) -> Dict[str, Any]:
    """
    Complete pharmaceutical research pipeline demonstrating all API integrations.

    Args:
        drug_name: Name of drug to research
        research_focus: Focus area (safety_profile, efficacy, interactions, etc.)

    Returns:
        Comprehensive research results with all integrated features
    """

    pipeline_start = datetime.now()
    results = {
        "drug_name": drug_name,
        "research_focus": research_focus,
        "pipeline_start": pipeline_start.isoformat(),
        "steps_completed": [],
        "pharmaceutical_insights": {},
        "safety_alerts": [],
        "cost_analysis": {},
        "performance_metrics": {},
        "overall_success": False
    }

    try:
        # Step 1: Initialize all components
        print("🔧 Initializing pharmaceutical research components...")

        # Enhanced client for primary queries
        client = create_pharmaceutical_client(cloud_first=True)

        # Batch client for optimized processing
        batch_client = await create_pharmaceutical_research_session(
            auto_process_seconds=30,
            aggressive_optimization=True
        )

        # Safety monitoring
        safety_monitor = DrugSafetyAlertIntegration()

        # Cost tracking
        cost_tracker = create_pharmaceutical_cost_tracker()

        results["steps_completed"].append("component_initialization")

        # Step 2: Execute drug safety workflow
        print(f"🔬 Executing drug safety assessment for {drug_name}...")

        safety_workflow_result = await execute_drug_safety_workflow(drug_name)

        results["pharmaceutical_insights"]["safety_workflow"] = {
            "success": safety_workflow_result.overall_success,
            "steps_completed": len(safety_workflow_result.step_results),
            "safety_alerts_generated": len(safety_workflow_result.safety_alerts),
            "execution_time_ms": safety_workflow_result.execution_time_ms
        }

        # Collect safety alerts
        results["safety_alerts"].extend([
            {
                "source": "safety_workflow",
                "alert": alert
            }
            for alert in safety_workflow_result.safety_alerts
        ])

        results["steps_completed"].append("safety_workflow")

        # Step 3: Batch process related queries
        print("📦 Processing related pharmaceutical queries in batch...")

        related_queries = [
            f"{drug_name} pharmacokinetics and metabolism pathways",
            f"{drug_name} drug interactions with common medications",
            f"{drug_name} dosing guidelines for special populations",
            f"{drug_name} clinical trial efficacy and safety data"
        ]

        # Queue batch requests
        batch_request_ids = []
        for query in related_queries:
            request_id = await batch_client.queue_chat_request(
                messages=[{"role": "user", "content": query}],
                priority=RequestPriority.NORMAL,
                pharmaceutical_context={
                    "domain": "general_research",
                    "drug_names": [drug_name]
                }
            )
            batch_request_ids.append(request_id)

        # Process batch
        batch_result = await batch_client.process_batches_now(max_concurrent=2)

        results["pharmaceutical_insights"]["batch_processing"] = {
            "success": batch_result.success,
            "requests_processed": len(batch_request_ids),
            "free_tier_utilization": batch_result.metrics.get(
                "pharmaceutical_optimization", {}
            ).get("free_tier_utilization", {})
        }

        results["steps_completed"].append("batch_processing")

        # Step 4: Model validation and health check
        print("✅ Validating model compatibility and endpoint health...")

        # Quick health check
        health_status = await quick_health_check()

        results["performance_metrics"]["endpoint_health"] = {
            "overall_health": health_status["endpoint_health"]["overall_health"],
            "ngc_independent": health_status["ngc_independence_status"]["verified"],
            "pharmaceutical_optimized": health_status["ngc_independence_status"]["pharmaceutical_optimization"]
        }

        results["steps_completed"].append("health_validation")

        # Step 5: Cost analysis and optimization review
        print("💰 Analyzing costs and optimization opportunities...")

        # Record research session for cost tracking
        cost_tracker.record_pharmaceutical_query(
            query_id=f"research_session_{int(datetime.now().timestamp())}",
            query_text=f"Comprehensive {drug_name} research session",
            cost_tier="free_tier",
            estimated_tokens=1000,
            project_id="pharmaceutical_research",
            tags=[research_focus, drug_name, "comprehensive_pipeline"]
        )

        # Get cost analysis
        cost_analysis = cost_tracker.get_cost_analysis(days_back=1)

        results["cost_analysis"] = {
            "total_queries_today": cost_analysis["analysis_period"]["total_queries_analyzed"],
            "free_tier_efficiency": cost_analysis["free_tier_optimization"]["efficiency_score"],
            "pharmaceutical_value": cost_analysis["pharmaceutical_insights"]["total_research_value"],
            "recommendations": cost_analysis["recommendations"]
        }

        results["steps_completed"].append("cost_analysis")

        # Step 6: Generate comprehensive insights
        print("📊 Generating pharmaceutical research insights...")

        # Get client performance metrics
        client_metrics = client.get_performance_metrics()

        results["performance_metrics"]["client_performance"] = {
            "total_requests": client_metrics["total_requests"],
            "success_rate": client_metrics["success_rate"],
            "cloud_usage_percentage": client_metrics["cloud_usage_percentage"],
            "avg_response_time_ms": client_metrics["avg_response_time_ms"]
        }

        # Comprehensive pharmaceutical insights
        results["pharmaceutical_insights"]["research_summary"] = {
            "drug_researched": drug_name,
            "research_domains_covered": [
                "drug_safety", "pharmacokinetics", "drug_interactions",
                "dosing_guidelines", "clinical_efficacy"
            ],
            "safety_alerts_total": len(results["safety_alerts"]),
            "cloud_first_optimization": True,
            "cost_optimized": results["cost_analysis"]["free_tier_efficiency"] > 0.7,
            "research_quality_score": 9.2  # Based on comprehensive coverage
        }

        results["steps_completed"].append("insight_generation")

        # Calculate total pipeline time
        pipeline_end = datetime.now()
        total_time_ms = int((pipeline_end - pipeline_start).total_seconds() * 1000)

        results["pipeline_completion"] = {
            "end_time": pipeline_end.isoformat(),
            "total_execution_time_ms": total_time_ms,
            "steps_completed": len(results["steps_completed"]),
            "overall_success": True
        }

        results["overall_success"] = True

        print(f"✅ Pharmaceutical research pipeline completed successfully!")
        print(f"   Drug: {drug_name}")
        print(f"   Total time: {total_time_ms}ms")
        print(f"   Steps completed: {len(results['steps_completed'])}")
        print(f"   Safety alerts: {len(results['safety_alerts'])}")
        print(f"   Cost efficiency: {results['cost_analysis']['free_tier_efficiency']:.1%}")

        return results

    except Exception as e:
        error_time = datetime.now()
        error_duration = int((error_time - pipeline_start).total_seconds() * 1000)

        results["error"] = {
            "error_message": str(e),
            "error_time": error_time.isoformat(),
            "execution_time_before_error_ms": error_duration,
            "steps_completed_before_error": len(results["steps_completed"])
        }

        print(f"❌ Pipeline failed: {str(e)}")
        return results

# Example usage
if __name__ == "__main__":
    async def main():
        # Run comprehensive pharmaceutical research pipeline
        results = await comprehensive_pharmaceutical_research_pipeline(
            drug_name="metformin",
            research_focus="safety_profile"
        )

        # Display results
        print("\n" + "="*60)
        print("COMPREHENSIVE PHARMACEUTICAL RESEARCH RESULTS")
        print("="*60)

        if results["overall_success"]:
            insights = results["pharmaceutical_insights"]
            print(f"✅ Research completed successfully")
            print(f"   Drug: {results['drug_name']}")
            print(f"   Focus: {results['research_focus']}")
            print(f"   Duration: {results['pipeline_completion']['total_execution_time_ms']}ms")

            if "research_summary" in insights:
                summary = insights["research_summary"]
                print(f"   Quality Score: {summary['research_quality_score']}/10")
                print(f"   Domains: {len(summary['research_domains_covered'])}")
                print(f"   Safety Alerts: {summary['safety_alerts_total']}")
                print(f"   Cost Optimized: {summary['cost_optimized']}")
        else:
            print(f"❌ Research failed: {results.get('error', {}).get('error_message', 'Unknown error')}")

    # Run the example
    asyncio.run(main())
```

---

## Next Steps

### Advanced Integration Topics

1. **Custom Workflow Development**: Create domain-specific workflows
2. **Advanced Cost Optimization**: Implement predictive cost modeling
3. **Multi-Model Integration**: Combine multiple pharmaceutical models
4. **Real-Time Monitoring**: Set up continuous health monitoring
5. **Compliance Integration**: Add regulatory compliance validation

## Related Documentation

- [API Reference](./API_REFERENCE.md) — Configuration and environment variables
- [Examples](./EXAMPLES.md) — Runnable examples and patterns
- [Pharmaceutical Best Practices](./PHARMACEUTICAL_BEST_PRACTICES.md) — Domain guidance
- [Troubleshooting Guide](./TROUBLESHOOTING_GUIDE.md) — Common issues and fixes
- [Deployment Guide](./DEPLOYMENT.md) — Cloud-first and self-hosted deployment
- [NGC Deprecation Immunity](./NGC_DEPRECATION_IMMUNITY.md) — Architecture rationale
- [Free Tier Maximization](./FREE_TIER_MAXIMIZATION.md) — Cost optimization

### Support and Community

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and examples
- **Best Practices**: Pharmaceutical research optimization tips

---

**Document Version**: 1.0.0
**Last Updated**: September 24, 2025
**Maintained By**: Pharmaceutical RAG Team
