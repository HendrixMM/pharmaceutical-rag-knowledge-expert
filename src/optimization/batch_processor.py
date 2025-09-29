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
            "pharmaceutical_requests_prioritized": 0,
            # Cost-per-query working aggregates
            "tokens_processed": 0,
            "requests_by_type": {"embedding": 0, "chat": 0},
            "tokens_by_type": {"embedding": 0, "chat": 0},
        }
        # Load optional centralized alert config
        self._alerts_cfg = self._load_alerts_config()

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
            # Track token usage aggregates
            try:
                total_tokens_batch = sum(max(0, int(r.estimated_tokens or 0)) for r in batch)
                self.batch_metrics["tokens_processed"] += total_tokens_batch
                for r in batch:
                    rt = r.request_type if r.request_type in ("embedding", "chat") else "chat"
                    self.batch_metrics["requests_by_type"][rt] = self.batch_metrics["requests_by_type"].get(rt, 0) + 1
                    self.batch_metrics["tokens_by_type"][rt] = self.batch_metrics["tokens_by_type"].get(rt, 0) + max(0, int(r.estimated_tokens or 0))
            except Exception:
                pass

            # Track request for rate limiting
            self.request_history.append(datetime.now())
            self.last_batch_time = datetime.now()

            # Optionally feed approximate request metrics into credit tracker
            try:
                if self.credit_tracker is not None:
                    for req in batch:
                        qtext = ""
                        qtype = "general"
                        model_used = req.request_type
                        if req.request_type == "chat":
                            try:
                                messages = req.payload.get("messages", [])
                                for m in reversed(messages):
                                    if m.get("role") == "user":
                                        qtext = str(m.get("content", ""))
                                        break
                            except Exception:
                                pass
                            qtype = self._classify_query_text(qtext)
                        elif req.request_type == "embedding":
                            qtype = "embedding"

                        self.credit_tracker.track_pharmaceutical_query(
                            query_type=qtype,
                            model_used=model_used,
                            tokens_consumed=req.estimated_tokens,
                            response_time_ms=execution_time,
                            cost_tier="free_tier",
                            research_context=qtext if qtext else None,
                        )
            except Exception:
                logger.debug("Skipped credit tracker logging for batch")

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
        """Check if sufficient free tier budget is available.

        Uses the credit tracker/base monitor summaries and thresholds
        loaded from the centralized alerts config (`self._alerts_cfg`).
        Returns False when monthly usage meets/exceeds the configured
        critical threshold; True otherwise. On failure, defaults to True.
        """
        try:
            monthly_usage = 0
            # Prefer analytics from tracker, fall back to base monitor summary
            try:
                analytics = self.credit_tracker.get_pharmaceutical_analytics()
                if isinstance(analytics, dict):
                    base_summary = analytics.get("base_monitor_summary", {}) or {}
                    monthly_usage = int(base_summary.get("requests_this_month", 0) or 0)
            except Exception:
                pass

            if not monthly_usage:
                try:
                    base = getattr(self.credit_tracker, "base_monitor", None)
                    if base and hasattr(base, "get_usage_summary"):
                        summary = base.get_usage_summary() or {}
                        monthly_usage = int(summary.get("requests_this_month", 0) or 0)
                except Exception:
                    pass

            # Thresholds from centralized alerts config (best-effort)
            nvidia_cfg = (self._alerts_cfg.get("nvidia_build") if isinstance(self._alerts_cfg, dict) else {}) or {}
            monthly_limit = int(nvidia_cfg.get("monthly_free_requests", 10000))
            usage_alerts = (nvidia_cfg.get("usage_alerts") or {}) if isinstance(nvidia_cfg, dict) else {}
            critical_threshold = float(usage_alerts.get("monthly_usage_critical", 0.95))

            limit_at_critical = monthly_limit * critical_threshold
            if monthly_usage >= limit_at_critical:
                logger.warning(
                    "Monthly usage (%.0f) reached critical threshold (%.0f of %s)",
                    monthly_usage,
                    limit_at_critical,
                    monthly_limit,
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Budget check failed: {str(e)}")
            # Default to allow processing if check fails
            return True

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
        metrics = {
            "pharmaceutical_requests_prioritized": self.batch_metrics["pharmaceutical_requests_prioritized"],
            "critical_priority_enabled": True,
            "domain_optimization_active": True,
            "research_workflow_efficiency": "optimized",
        }

        # Merge cost-per-query metrics
        try:
            metrics["cost_per_query"] = self._calculate_cost_per_query_metrics()
        except Exception:
            metrics["cost_per_query"] = {
                "average_cost_per_query_usd": 0.0,
                "avg_tokens_per_query": 0,
                "over_threshold": False,
            }

        return metrics

    def _calculate_cost_per_query_metrics(self) -> Dict[str, Any]:
        """Compute average cost-per-pharmaceutical-query metrics.

        Uses estimated token counts and a conservative $/1k tokens rate
        (aligned with cost savings assumptions) to provide an approximate
        cost-per-query. Also returns per-type breakdowns when available
        and evaluates against workflow_efficiency.cost_per_query_warning
        threshold from alerts config.
        """
        total_requests = int(self.batch_metrics.get("requests_batched", 0) or 0)
        tokens_total = int(self.batch_metrics.get("tokens_processed", 0) or 0)

        # Conservative estimate rate ($ per 1k tokens)
        usd_per_1k = 0.002

        avg_tokens = int(tokens_total / total_requests) if total_requests > 0 else 0
        avg_cost = (avg_tokens / 1000.0) * usd_per_1k if total_requests > 0 else 0.0

        # By type
        req_by_type = self.batch_metrics.get("requests_by_type", {}) or {}
        tok_by_type = self.batch_metrics.get("tokens_by_type", {}) or {}
        by_type_cost = {}
        by_type_tokens = {}
        by_type_over = {}

        # Threshold from alerts config
        wf_cfg = {}
        try:
            wf_cfg = ((self._alerts_cfg.get("pharmaceutical") or {}).get("workflow_efficiency") or {}) if isinstance(self._alerts_cfg, dict) else {}
        except Exception:
            wf_cfg = {}
        warn_threshold = float(wf_cfg.get("cost_per_query_warning", 0.1))

        for t in ("embedding", "chat"):
            rcount = int(req_by_type.get(t, 0) or 0)
            tkns = int(tok_by_type.get(t, 0) or 0)
            by_type_tokens[t] = int(tkns / rcount) if rcount > 0 else 0
            cost_t = (by_type_tokens[t] / 1000.0) * usd_per_1k if rcount > 0 else 0.0
            by_type_cost[t] = round(cost_t, 6)
            by_type_over[t] = bool(cost_t > warn_threshold)

        return {
            "average_cost_per_query_usd": round(avg_cost, 6),
            "avg_tokens_per_query": avg_tokens,
            "cost_per_query_by_type_usd": by_type_cost,
            "avg_tokens_per_query_by_type": by_type_tokens,
            "warning_threshold_usd": warn_threshold,
            "over_threshold": bool(avg_cost > warn_threshold),
            "over_threshold_by_type": by_type_over,
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

    # ------------------------ Internal helpers ------------------------
    def _load_alerts_config(self) -> Dict[str, Any]:
        """Load centralized alert configuration (best-effort)."""
        try:
            import yaml  # type: ignore
            from pathlib import Path
            cfg_path = Path("config/alerts.yaml")
            if not cfg_path.exists():
                return {}
            with open(cfg_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _classify_query_text(self, text: str) -> str:
        """Basic pharma query classification used for tracking only."""
        t = (text or "").lower()
        if any(k in t for k in ("interaction", "contraindication", "together", "combined")):
            return "drug_interaction"
        if any(k in t for k in ("pharmacokinetic", "half-life", "clearance", "absorption", "metabolism")):
            return "pharmacokinetics"
        if any(k in t for k in ("trial", "study", "patient", "efficacy", "safety")):
            return "clinical_trial"
        if any(k in t for k in ("mechanism", "pathway", "target", "receptor")):
            return "mechanism"
        return "general"

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
