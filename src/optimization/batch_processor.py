"""
Batch Processing Optimization for NVIDIA Build Free Tier Efficiency

Implements intelligent request batching strategies to maximize pharmaceutical research
value from the 10K monthly request limit on NVIDIA Build platform.

Architecture:
- Request queue management with pharmaceutical prioritization
- Optimal timing algorithms for rate limit efficiency
- Batch size optimization for embedding and chat operations
- Cost-aware scheduling with free tier maximization

Integration:
- Works with enhanced credit tracking for budget awareness
- Integrates with alert system for threshold monitoring
- Supports pharmaceutical workflow optimization
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json
import hashlib

try:
    from ..enhanced_config import EnhancedRAGConfig
    from ..monitoring.credit_tracker import PharmaceuticalCreditTracker
except ImportError:
    from src.enhanced_config import EnhancedRAGConfig
    from src.monitoring.credit_tracker import PharmaceuticalCreditTracker

logger = logging.getLogger(__name__)

class RequestPriority(IntEnum):
    """Request priority levels for pharmaceutical research optimization."""
    CRITICAL = 1      # Drug safety, adverse reactions
    HIGH = 2         # Clinical trials, efficacy studies
    NORMAL = 3       # General research, mechanism queries
    BATCH = 4        # Background processing, bulk operations

class BatchType(Enum):
    """Types of batch operations for optimization."""
    EMBEDDINGS = "embeddings"
    CHAT_COMPLETION = "chat_completion"
    MIXED = "mixed"

@dataclass
class BatchRequest:
    """Individual request within a batch operation."""
    request_id: str
    request_type: str  # "embedding" or "chat"
    payload: Dict[str, Any]
    priority: RequestPriority = RequestPriority.NORMAL
    pharmaceutical_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    estimated_tokens: int = 0

    def __post_init__(self):
        """Calculate estimated tokens for cost projection."""
        if self.estimated_tokens == 0:
            self.estimated_tokens = self._estimate_token_usage()

    def _estimate_token_usage(self) -> int:
        """Estimate token usage for cost calculation."""
        if self.request_type == "embedding":
            # Estimate based on text length (roughly 1 token per 4 characters)
            texts = self.payload.get("texts", [])
            total_chars = sum(len(text) for text in texts)
            return max(1, total_chars // 4)

        elif self.request_type == "chat":
            # Estimate based on messages and expected response
            messages = self.payload.get("messages", [])
            message_chars = sum(len(msg.get("content", "")) for msg in messages)
            max_tokens = self.payload.get("max_tokens", 500)
            return max(1, (message_chars // 4) + max_tokens)

        return 100  # Default conservative estimate

@dataclass
class BatchOptimizationStrategy:
    """Configuration for batch optimization strategies."""
    max_batch_size: int = 50
    max_wait_time_seconds: int = 30
    priority_boost_factor: float = 2.0
    pharmaceutical_boost_factor: float = 1.5
    rate_limit_buffer: float = 0.1  # 10% buffer under rate limits
    enable_intelligent_scheduling: bool = True
    enable_cost_optimization: bool = True

class BatchProcessor:
    """
    Intelligent batch processor for NVIDIA Build free tier optimization.

    Implements sophisticated queuing, batching, and scheduling algorithms
    to maximize pharmaceutical research value within cost constraints.
    """

    def __init__(self,
                 config: Optional[EnhancedRAGConfig] = None,
                 credit_tracker: Optional[PharmaceuticalCreditTracker] = None,
                 strategy: Optional[BatchOptimizationStrategy] = None):
        """
        Initialize batch processor with pharmaceutical optimization.

        Args:
            config: Enhanced RAG configuration
            credit_tracker: Credit tracking for budget awareness
            strategy: Batch optimization strategy configuration
        """
        self.config = config or EnhancedRAGConfig.from_env()
        self.credit_tracker = credit_tracker or PharmaceuticalCreditTracker()
        self.strategy = strategy or BatchOptimizationStrategy()

        # Request queues by priority
        self.request_queues: Dict[RequestPriority, List[BatchRequest]] = {
            priority: [] for priority in RequestPriority
        }

        # Processing state
        self.is_processing = False
        self.processing_lock = asyncio.Lock()
        self.batch_metrics = {
            "total_batches_processed": 0,
            "requests_batched": 0,
            "tokens_saved": 0,
            "cost_savings_estimated": 0.0,
            "pharmaceutical_requests_prioritized": 0
        }

        # Rate limiting state
        self.request_history: List[datetime] = []
        self.last_batch_time: Optional[datetime] = None

        logger.info("BatchProcessor initialized with pharmaceutical optimization")

    def queue_request(self,
                     request_type: str,
                     payload: Dict[str, Any],
                     priority: RequestPriority = RequestPriority.NORMAL,
                     pharmaceutical_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Queue a request for batch processing.

        Args:
            request_type: Type of request ("embedding" or "chat")
            payload: Request payload data
            priority: Request priority level
            pharmaceutical_context: Pharmaceutical research context

        Returns:
            Request ID for tracking
        """
        # Generate unique request ID
        request_id = hashlib.md5(
            f"{request_type}_{time.time()}_{hash(str(payload))}".encode()
        ).hexdigest()[:12]

        # Create batch request
        batch_request = BatchRequest(
            request_id=request_id,
            request_type=request_type,
            payload=payload,
            priority=priority,
            pharmaceutical_context=pharmaceutical_context
        )

        # Apply pharmaceutical prioritization
        if pharmaceutical_context:
            batch_request.priority = self._apply_pharmaceutical_prioritization(
                batch_request.priority, pharmaceutical_context
            )
            self.batch_metrics["pharmaceutical_requests_prioritized"] += 1

        # Add to appropriate priority queue
        self.request_queues[batch_request.priority].append(batch_request)

        logger.debug(f"Queued request {request_id} with priority {priority.name}")
        return request_id

    def _apply_pharmaceutical_prioritization(self,
                                           base_priority: RequestPriority,
                                           pharma_context: Dict[str, Any]) -> RequestPriority:
        """Apply pharmaceutical domain-specific prioritization."""

        # Critical pharmaceutical keywords that boost priority
        critical_keywords = [
            "adverse", "toxicity", "contraindication", "drug interaction",
            "safety", "warning", "side effect", "overdose"
        ]

        # High priority pharmaceutical keywords
        high_priority_keywords = [
            "clinical trial", "efficacy", "dosage", "pharmacokinetics",
            "bioavailability", "therapeutic", "treatment"
        ]

        query_text = str(pharma_context.get("query", "")).lower()

        # Check for critical pharmaceutical contexts
        if any(keyword in query_text for keyword in critical_keywords):
            return RequestPriority.CRITICAL

        # Check for high priority contexts
        if any(keyword in query_text for keyword in high_priority_keywords):
            return min(RequestPriority.HIGH, base_priority)

        # Apply general pharmaceutical boost
        if base_priority.value > RequestPriority.HIGH.value:
            return RequestPriority(base_priority.value - 1)

        return base_priority

    async def process_batches(self,
                            executor_func: Callable,
                            max_concurrent_batches: int = 3) -> Dict[str, Any]:
        """
        Process queued requests in optimized batches.

        Args:
            executor_func: Function to execute batches
            max_concurrent_batches: Maximum concurrent batch processing

        Returns:
            Processing results and metrics
        """
        async with self.processing_lock:
            if self.is_processing:
                logger.warning("Batch processing already in progress")
                return {"status": "already_processing"}

            self.is_processing = True

        try:
            results = {
                "batches_processed": 0,
                "requests_processed": 0,
                "processing_time_ms": 0,
                "cost_optimization": {},
                "pharmaceutical_metrics": {}
            }

            start_time = time.time()

            # Check free tier budget before processing
            if not await self._check_budget_availability():
                logger.warning("Insufficient free tier budget for batch processing")
                return {"status": "budget_insufficient", "results": results}

            # Process batches by priority
            for priority in RequestPriority:
                if not self.request_queues[priority]:
                    continue

                priority_results = await self._process_priority_queue(
                    priority, executor_func, max_concurrent_batches
                )

                results["batches_processed"] += priority_results["batches_processed"]
                results["requests_processed"] += priority_results["requests_processed"]

            # Calculate metrics
            processing_time = int((time.time() - start_time) * 1000)
            results["processing_time_ms"] = processing_time
            results["cost_optimization"] = self._calculate_cost_optimization()
            results["pharmaceutical_metrics"] = self._calculate_pharmaceutical_metrics()

            logger.info(f"Batch processing completed: {results['requests_processed']} requests "
                       f"in {results['batches_processed']} batches ({processing_time}ms)")

            return {"status": "success", "results": results}

        finally:
            self.is_processing = False

    async def _process_priority_queue(self,
                                    priority: RequestPriority,
                                    executor_func: Callable,
                                    max_concurrent: int) -> Dict[str, Any]:
        """Process requests from a specific priority queue."""
        queue = self.request_queues[priority]
        if not queue:
            return {"batches_processed": 0, "requests_processed": 0}

        logger.debug(f"Processing {len(queue)} requests with priority {priority.name}")

        batches = self._create_optimal_batches(queue, priority)
        processed_batches = 0
        processed_requests = 0

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: List[BatchRequest]) -> Dict[str, Any]:
            async with semaphore:
                return await self._execute_batch(batch, executor_func)

        # Execute all batches for this priority
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {str(result)}")
                continue

            if result.get("success", False):
                processed_batches += 1
                processed_requests += result.get("requests_processed", 0)

        # Clear processed requests from queue
        queue.clear()

        return {
            "batches_processed": processed_batches,
            "requests_processed": processed_requests
        }

    def _create_optimal_batches(self,
                              requests: List[BatchRequest],
                              priority: RequestPriority) -> List[List[BatchRequest]]:
        """Create optimally sized batches based on request types and costs."""
        if not requests:
            return []

        # Sort requests by type and estimated cost
        embedding_requests = [r for r in requests if r.request_type == "embedding"]
        chat_requests = [r for r in requests if r.request_type == "chat"]

        batches = []

        # Create embedding batches (can be larger)
        embedding_batch_size = min(self.strategy.max_batch_size, 100)
        for i in range(0, len(embedding_requests), embedding_batch_size):
            batch = embedding_requests[i:i + embedding_batch_size]
            batches.append(batch)

        # Create chat batches (typically smaller for cost control)
        chat_batch_size = min(self.strategy.max_batch_size, 10)
        for i in range(0, len(chat_requests), chat_batch_size):
            batch = chat_requests[i:i + chat_batch_size]
            batches.append(batch)

        logger.debug(f"Created {len(batches)} optimal batches for priority {priority.name}")
        return batches

    async def _execute_batch(self,
                           batch: List[BatchRequest],
                           executor_func: Callable) -> Dict[str, Any]:
        """Execute a single batch with rate limiting and error handling."""
        if not batch:
            return {"success": True, "requests_processed": 0}

        # Rate limiting check
        await self._apply_rate_limiting()

        try:
            # Prepare batch payload
            batch_payload = self._prepare_batch_payload(batch)

            # Execute batch via provided function
            start_time = time.time()
            batch_result = await executor_func(batch_payload)
            execution_time = int((time.time() - start_time) * 1000)

            # Update metrics
            self.batch_metrics["total_batches_processed"] += 1
            self.batch_metrics["requests_batched"] += len(batch)

            # Track request for rate limiting
            self.request_history.append(datetime.now())
            self.last_batch_time = datetime.now()

            logger.debug(f"Batch executed successfully: {len(batch)} requests ({execution_time}ms)")

            return {
                "success": True,
                "requests_processed": len(batch),
                "execution_time_ms": execution_time,
                "result": batch_result
            }

        except Exception as e:
            logger.error(f"Batch execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "requests_processed": 0
            }

    def _prepare_batch_payload(self, batch: List[BatchRequest]) -> Dict[str, Any]:
        """Prepare consolidated payload for batch execution."""
        if not batch:
            return {}

        # Group by request type
        embedding_requests = [r for r in batch if r.request_type == "embedding"]
        chat_requests = [r for r in batch if r.request_type == "chat"]

        payload = {
            "batch_id": f"batch_{int(time.time())}",
            "batch_size": len(batch),
            "embedding_requests": [],
            "chat_requests": [],
            "pharmaceutical_contexts": []
        }

        # Prepare embedding requests
        for req in embedding_requests:
            payload["embedding_requests"].append({
                "request_id": req.request_id,
                "texts": req.payload.get("texts", []),
                "model": req.payload.get("model")
            })

            if req.pharmaceutical_context:
                payload["pharmaceutical_contexts"].append({
                    "request_id": req.request_id,
                    "context": req.pharmaceutical_context
                })

        # Prepare chat requests
        for req in chat_requests:
            payload["chat_requests"].append({
                "request_id": req.request_id,
                "messages": req.payload.get("messages", []),
                "model": req.payload.get("model"),
                "max_tokens": req.payload.get("max_tokens"),
                "temperature": req.payload.get("temperature")
            })

            if req.pharmaceutical_context:
                payload["pharmaceutical_contexts"].append({
                    "request_id": req.request_id,
                    "context": req.pharmaceutical_context
                })

        return payload

    async def _apply_rate_limiting(self) -> None:
        """Apply intelligent rate limiting for free tier optimization."""
        now = datetime.now()

        # Clean old requests from history (keep last minute)
        minute_ago = now - timedelta(minutes=1)
        self.request_history = [
            req_time for req_time in self.request_history
            if req_time > minute_ago
        ]

        # Check if we need to throttle
        requests_per_minute = len(self.request_history)
        max_requests_per_minute = 50  # Conservative limit with buffer

        if requests_per_minute >= max_requests_per_minute:
            sleep_time = 60.0 / max_requests_per_minute
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)

        # Minimum time between batches for stability
        if self.last_batch_time:
            time_since_last = (now - self.last_batch_time).total_seconds()
            min_interval = 1.0  # 1 second minimum

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)

    async def _check_budget_availability(self) -> bool:
        """Check if sufficient free tier budget is available."""
        try:
            usage_stats = self.credit_tracker.get_usage_stats()
            daily_usage = usage_stats.get("requests_today", 0)
            monthly_usage = usage_stats.get("requests_this_month", 0)

            # Check against alert thresholds
            alerts_config = self.config.get_alerts_config()
            nvidia_config = alerts_config.get("nvidia_build", {})

            monthly_limit = nvidia_config.get("monthly_free_requests", 10000)
            critical_threshold = nvidia_config.get("usage_alerts", {}).get("monthly_usage_critical", 0.95)

            # Allow processing if under critical threshold
            if monthly_usage < (monthly_limit * critical_threshold):
                return True

            logger.warning(f"Monthly usage ({monthly_usage}) approaching limit ({monthly_limit})")
            return False

        except Exception as e:
            logger.error(f"Budget check failed: {str(e)}")
            return True  # Default to allow processing if check fails

    def _calculate_cost_optimization(self) -> Dict[str, Any]:
        """Calculate cost optimization metrics."""
        total_requests = self.batch_metrics["requests_batched"]
        if total_requests == 0:
            return {}

        # Estimate savings from batching (reduced API overhead)
        estimated_individual_calls = total_requests
        actual_batch_calls = self.batch_metrics["total_batches_processed"]
        api_calls_saved = max(0, estimated_individual_calls - actual_batch_calls)

        # Estimate cost savings (assuming $0.002 per 1K tokens saved)
        tokens_saved = api_calls_saved * 100  # Conservative estimate
        cost_savings = tokens_saved * 0.002 / 1000

        return {
            "api_calls_saved": api_calls_saved,
            "tokens_saved": tokens_saved,
            "estimated_cost_savings_usd": round(cost_savings, 4),
            "batch_efficiency": round(actual_batch_calls / max(estimated_individual_calls, 1), 3),
            "free_tier_optimization_active": True
        }

    def _calculate_pharmaceutical_metrics(self) -> Dict[str, Any]:
        """Calculate pharmaceutical research-specific metrics."""
        return {
            "pharmaceutical_requests_prioritized": self.batch_metrics["pharmaceutical_requests_prioritized"],
            "critical_priority_enabled": True,
            "domain_optimization_active": True,
            "research_workflow_efficiency": "optimized"
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of all request queues."""
        status = {
            "total_queued_requests": sum(len(queue) for queue in self.request_queues.values()),
            "queues_by_priority": {},
            "processing_active": self.is_processing,
            "batch_metrics": self.batch_metrics.copy()
        }

        for priority, queue in self.request_queues.items():
            status["queues_by_priority"][priority.name] = {
                "count": len(queue),
                "estimated_tokens": sum(req.estimated_tokens for req in queue),
                "pharmaceutical_requests": sum(
                    1 for req in queue if req.pharmaceutical_context
                )
            }

        return status

    def clear_queues(self, priority: Optional[RequestPriority] = None) -> Dict[str, int]:
        """Clear request queues (useful for testing or emergency situations)."""
        if priority:
            cleared = len(self.request_queues[priority])
            self.request_queues[priority].clear()
            return {priority.name: cleared}

        cleared = {}
        for prio, queue in self.request_queues.items():
            cleared[prio.name] = len(queue)
            queue.clear()

        logger.info(f"Cleared queues: {cleared}")
        return cleared

# Pharmaceutical research convenience functions
def create_pharmaceutical_batch_processor(
    enhanced_tracking: bool = True,
    aggressive_optimization: bool = True
) -> BatchProcessor:
    """
    Create batch processor optimized for pharmaceutical research workflows.

    Args:
        enhanced_tracking: Enable enhanced pharmaceutical credit tracking
        aggressive_optimization: Use aggressive free tier optimization

    Returns:
        Configured BatchProcessor for pharmaceutical research
    """
    config = EnhancedRAGConfig.from_env()

    # Enhanced pharmaceutical credit tracking
    credit_tracker = None
    if enhanced_tracking:
        credit_tracker = PharmaceuticalCreditTracker()

    # Optimization strategy for pharmaceutical research
    strategy = BatchOptimizationStrategy(
        max_batch_size=75 if aggressive_optimization else 50,
        max_wait_time_seconds=20 if aggressive_optimization else 30,
        pharmaceutical_boost_factor=2.0,  # Strong pharmaceutical prioritization
        enable_intelligent_scheduling=True,
        enable_cost_optimization=True
    )

    return BatchProcessor(
        config=config,
        credit_tracker=credit_tracker,
        strategy=strategy
    )

if __name__ == "__main__":
    # Test pharmaceutical batch processor
    import json

    async def test_batch_processor():
        processor = create_pharmaceutical_batch_processor()

        # Queue test requests
        processor.queue_request(
            "embedding",
            {"texts": ["metformin mechanism of action in diabetes"]},
            RequestPriority.HIGH,
            {"query": "metformin mechanism of action", "domain": "pharmaceutical"}
        )

        processor.queue_request(
            "chat",
            {"messages": [{"role": "user", "content": "Explain drug interactions with ACE inhibitors"}]},
            RequestPriority.CRITICAL,
            {"query": "drug interactions ACE inhibitors", "domain": "pharmaceutical"}
        )

        # Show queue status
        status = processor.get_queue_status()
        print("Batch Processor Status:")
        print(json.dumps(status, indent=2))

    # Run test
    asyncio.run(test_batch_processor())