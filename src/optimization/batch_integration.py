"""
Batch Processing Integration for Enhanced NeMo Client

Integrates the BatchProcessor with EnhancedNeMoClient to provide
seamless batched operations with cloud-first optimization and
pharmaceutical domain prioritization.

Architecture:
- Bridges BatchProcessor with EnhancedNeMoClient execution
- Implements batch execution strategies for embeddings and chat
- Provides pharmaceutical workflow optimization
- Maintains cost monitoring and free tier maximization

Integration Flow:
1. Requests queued via BatchProcessor with pharmaceutical prioritization
2. Optimal batches created based on request types and costs
3. Batches executed via EnhancedNeMoClient with fallback logic
4. Results aggregated and returned with performance metrics
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

try:
    from ..clients.nemo_client_enhanced import EnhancedNeMoClient
    from .batch_processor import BatchProcessor, RequestPriority, create_pharmaceutical_batch_processor
except ImportError:
    from src.clients.nemo_client_enhanced import EnhancedNeMoClient
    from src.optimization.batch_processor import BatchProcessor, RequestPriority, create_pharmaceutical_batch_processor

logger = logging.getLogger(__name__)


class BatchExecutionResult:
    """Result wrapper for batch execution operations."""

    def __init__(
        self,
        success: bool,
        results: List[Dict[str, Any]] = None,
        metrics: Dict[str, Any] = None,
        errors: List[str] = None,
    ):
        self.success = success
        self.results = results or []
        self.metrics = metrics or {}
        self.errors = errors or []
        self.timestamp = datetime.now()


class PharmaceuticalBatchClient:
    """
    Integrated batch client for pharmaceutical research workflows.

    Combines BatchProcessor optimization with EnhancedNeMoClient execution
    to maximize free tier value while maintaining pharmaceutical domain advantages.
    """

    def __init__(
        self,
        enhanced_client: Optional[EnhancedNeMoClient] = None,
        batch_processor: Optional[BatchProcessor] = None,
        auto_process_interval: Optional[int] = None,
    ):
        """
        Initialize pharmaceutical batch client.

        Args:
            enhanced_client: Enhanced NeMo client for execution
            batch_processor: Batch processor for optimization
            auto_process_interval: Automatic processing interval in seconds
        """
        self.enhanced_client = enhanced_client or EnhancedNeMoClient(pharmaceutical_optimized=True)
        self.batch_processor = batch_processor or create_pharmaceutical_batch_processor(
            enhanced_tracking=True, aggressive_optimization=True
        )

        # Auto-processing configuration
        self.auto_process_interval = auto_process_interval
        self._auto_process_task: Optional[asyncio.Task] = None
        self._stop_auto_processing = False

        # Performance tracking
        self.execution_metrics = {
            "total_batch_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_requests_processed": 0,
            "pharmaceutical_requests_processed": 0,
            "free_tier_requests": 0,
            "fallback_requests": 0,
            "total_execution_time_ms": 0.0,
        }

        logger.info("PharmaceuticalBatchClient initialized with enhanced optimization")

    async def __aenter__(self):
        """Async context manager entry."""
        if self.auto_process_interval:
            await self.start_auto_processing()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_auto_processing()

    async def queue_embedding_request(
        self,
        texts: List[str],
        model: Optional[str] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        pharmaceutical_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Queue embedding request for batch processing.

        Args:
            texts: List of texts to embed
            model: Embedding model name
            priority: Request priority level
            pharmaceutical_context: Pharmaceutical research context

        Returns:
            Request ID for tracking
        """
        payload = {"texts": texts, "model": model}

        request_id = self.batch_processor.queue_request(
            request_type="embedding", payload=payload, priority=priority, pharmaceutical_context=pharmaceutical_context
        )

        logger.debug(f"Queued embedding request {request_id} for {len(texts)} texts")
        return request_id

    async def queue_chat_request(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        pharmaceutical_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Queue chat completion request for batch processing.

        Args:
            messages: List of message dictionaries
            model: Chat model name
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            priority: Request priority level
            pharmaceutical_context: Pharmaceutical research context

        Returns:
            Request ID for tracking
        """
        payload = {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature}

        request_id = self.batch_processor.queue_request(
            request_type="chat", payload=payload, priority=priority, pharmaceutical_context=pharmaceutical_context
        )

        logger.debug(f"Queued chat request {request_id}")
        return request_id

    async def process_batches_now(self, max_concurrent: int = 3) -> BatchExecutionResult:
        """
        Process all queued batches immediately.

        Args:
            max_concurrent: Maximum concurrent batch processing

        Returns:
            BatchExecutionResult with comprehensive results
        """
        start_time = time.time()

        try:
            # Execute batches via processor
            processing_result = await self.batch_processor.process_batches(
                executor_func=self._execute_batch_payload, max_concurrent_batches=max_concurrent
            )

            execution_time = int((time.time() - start_time) * 1000)

            # Update metrics
            self.execution_metrics["total_batch_executions"] += 1
            self.execution_metrics["total_execution_time_ms"] += execution_time

            if processing_result.get("status") == "success":
                self.execution_metrics["successful_executions"] += 1
                results = processing_result.get("results", {})
                self.execution_metrics["total_requests_processed"] += results.get("requests_processed", 0)

                return BatchExecutionResult(
                    success=True, results=[results], metrics=self._compile_execution_metrics(results, execution_time)
                )
            else:
                self.execution_metrics["failed_executions"] += 1
                return BatchExecutionResult(
                    success=False,
                    errors=[f"Batch processing failed: {processing_result.get('status')}"],
                    metrics=self._compile_execution_metrics({}, execution_time),
                )

        except Exception as e:
            self.execution_metrics["failed_executions"] += 1
            logger.error(f"Batch execution failed: {str(e)}")

            return BatchExecutionResult(
                success=False,
                errors=[str(e)],
                metrics=self._compile_execution_metrics({}, int((time.time() - start_time) * 1000)),
            )

    async def _execute_batch_payload(self, batch_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a batch payload using the enhanced NeMo client.

        Args:
            batch_payload: Prepared batch payload from BatchProcessor

        Returns:
            Batch execution results
        """
        results = {
            "batch_id": batch_payload.get("batch_id"),
            "embedding_results": [],
            "chat_results": [],
            "execution_summary": {
                "total_requests": batch_payload.get("batch_size", 0),
                "successful_requests": 0,
                "failed_requests": 0,
                "cloud_requests": 0,
                "fallback_requests": 0,
            },
        }

        # Process embedding requests
        embedding_requests = batch_payload.get("embedding_requests", [])
        for embed_req in embedding_requests:
            try:
                response = self.enhanced_client.create_embeddings(
                    texts=embed_req.get("texts", []), model=embed_req.get("model")
                )

                results["embedding_results"].append(
                    {
                        "request_id": embed_req.get("request_id"),
                        "success": response.success,
                        "data": response.data,
                        "endpoint_type": response.endpoint_type.value if response.endpoint_type else None,
                        "cost_tier": response.cost_tier,
                        "response_time_ms": response.response_time_ms,
                    }
                )

                # Update execution metrics
                if response.success:
                    results["execution_summary"]["successful_requests"] += 1
                    if response.cost_tier == "free_tier":
                        results["execution_summary"]["cloud_requests"] += 1
                        self.execution_metrics["free_tier_requests"] += 1
                    else:
                        results["execution_summary"]["fallback_requests"] += 1
                        self.execution_metrics["fallback_requests"] += 1
                else:
                    results["execution_summary"]["failed_requests"] += 1

            except Exception as e:
                logger.error(f"Embedding request {embed_req.get('request_id')} failed: {str(e)}")
                results["embedding_results"].append(
                    {"request_id": embed_req.get("request_id"), "success": False, "error": str(e)}
                )
                results["execution_summary"]["failed_requests"] += 1

        # Process chat requests
        chat_requests = batch_payload.get("chat_requests", [])
        for chat_req in chat_requests:
            try:
                response = self.enhanced_client.create_chat_completion(
                    messages=chat_req.get("messages", []),
                    model=chat_req.get("model"),
                    max_tokens=chat_req.get("max_tokens"),
                    temperature=chat_req.get("temperature"),
                )

                results["chat_results"].append(
                    {
                        "request_id": chat_req.get("request_id"),
                        "success": response.success,
                        "data": response.data,
                        "endpoint_type": response.endpoint_type.value if response.endpoint_type else None,
                        "cost_tier": response.cost_tier,
                        "response_time_ms": response.response_time_ms,
                    }
                )

                # Update execution metrics
                if response.success:
                    results["execution_summary"]["successful_requests"] += 1
                    if response.cost_tier == "free_tier":
                        results["execution_summary"]["cloud_requests"] += 1
                        self.execution_metrics["free_tier_requests"] += 1
                    else:
                        results["execution_summary"]["fallback_requests"] += 1
                        self.execution_metrics["fallback_requests"] += 1
                else:
                    results["execution_summary"]["failed_requests"] += 1

            except Exception as e:
                logger.error(f"Chat request {chat_req.get('request_id')} failed: {str(e)}")
                results["chat_results"].append(
                    {"request_id": chat_req.get("request_id"), "success": False, "error": str(e)}
                )
                results["execution_summary"]["failed_requests"] += 1

        # Count pharmaceutical requests
        pharmaceutical_contexts = batch_payload.get("pharmaceutical_contexts", [])
        self.execution_metrics["pharmaceutical_requests_processed"] += len(pharmaceutical_contexts)

        logger.info(
            f"Batch {results['batch_id']} executed: "
            f"{results['execution_summary']['successful_requests']} successful, "
            f"{results['execution_summary']['failed_requests']} failed"
        )

        return results

    def _compile_execution_metrics(self, batch_results: Dict[str, Any], execution_time_ms: int) -> Dict[str, Any]:
        """Compile comprehensive execution metrics."""
        return {
            "batch_execution_time_ms": execution_time_ms,
            "batch_results": batch_results,
            "cumulative_metrics": self.execution_metrics.copy(),
            "queue_status": self.batch_processor.get_queue_status(),
            "client_performance": self.enhanced_client.get_performance_metrics(),
            "pharmaceutical_optimization": {
                "pharmaceutical_requests_processed": self.execution_metrics["pharmaceutical_requests_processed"],
                "free_tier_utilization": self._calculate_free_tier_utilization(),
                "cost_optimization_active": True,
            },
        }

    def _calculate_free_tier_utilization(self) -> Dict[str, Any]:
        """Calculate free tier utilization metrics."""
        total_requests = self.execution_metrics["total_requests_processed"]
        free_tier_requests = self.execution_metrics["free_tier_requests"]

        if total_requests == 0:
            return {"utilization_percentage": 0, "optimization_status": "no_data"}

        utilization_pct = (free_tier_requests / total_requests) * 100

        return {
            "utilization_percentage": round(utilization_pct, 2),
            "free_tier_requests": free_tier_requests,
            "total_requests": total_requests,
            "optimization_status": "excellent"
            if utilization_pct > 80
            else "good"
            if utilization_pct > 60
            else "needs_improvement",
        }

    async def start_auto_processing(self) -> None:
        """Start automatic batch processing at configured intervals."""
        if self._auto_process_task and not self._auto_process_task.done():
            logger.warning("Auto-processing already running")
            return

        if not self.auto_process_interval:
            logger.warning("Auto-processing interval not configured")
            return

        self._stop_auto_processing = False
        self._auto_process_task = asyncio.create_task(self._auto_process_loop())
        logger.info(f"Started auto-processing with {self.auto_process_interval}s interval")

    async def stop_auto_processing(self) -> None:
        """Stop automatic batch processing."""
        self._stop_auto_processing = True

        if self._auto_process_task and not self._auto_process_task.done():
            self._auto_process_task.cancel()
            try:
                await self._auto_process_task
            except asyncio.CancelledError:
                pass

        logger.info("Auto-processing stopped")

    async def _auto_process_loop(self) -> None:
        """Main loop for automatic batch processing."""
        while not self._stop_auto_processing:
            try:
                # Check if there are queued requests
                queue_status = self.batch_processor.get_queue_status()
                total_queued = queue_status.get("total_queued_requests", 0)

                if total_queued > 0:
                    logger.debug(f"Auto-processing {total_queued} queued requests")
                    result = await self.process_batches_now()

                    if result.success:
                        logger.info(f"Auto-processed batch successfully")
                    else:
                        logger.error(f"Auto-processing failed: {result.errors}")

                # Wait for next interval
                await asyncio.sleep(self.auto_process_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-processing error: {str(e)}")
                await asyncio.sleep(self.auto_process_interval)

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of batch client and all components."""
        return {
            "batch_processor": self.batch_processor.get_queue_status(),
            "enhanced_client": self.enhanced_client.get_endpoint_status(),
            "execution_metrics": self.execution_metrics.copy(),
            "auto_processing": {
                "enabled": self.auto_process_interval is not None,
                "interval_seconds": self.auto_process_interval,
                "active": self._auto_process_task is not None and not self._auto_process_task.done(),
            },
            "free_tier_optimization": self._calculate_free_tier_utilization(),
        }


# Pharmaceutical research convenience functions
async def create_pharmaceutical_research_session(
    auto_process_seconds: int = 30, aggressive_optimization: bool = True
) -> PharmaceuticalBatchClient:
    """
    Create optimized pharmaceutical research batch session.

    Args:
        auto_process_seconds: Automatic processing interval
        aggressive_optimization: Enable aggressive free tier optimization

    Returns:
        Configured PharmaceuticalBatchClient for research workflows
    """
    # Create enhanced client with pharmaceutical optimization
    enhanced_client = EnhancedNeMoClient(pharmaceutical_optimized=True, enable_fallback=True)

    # Create batch processor with pharmaceutical prioritization
    batch_processor = create_pharmaceutical_batch_processor(
        enhanced_tracking=True, aggressive_optimization=aggressive_optimization
    )

    return PharmaceuticalBatchClient(
        enhanced_client=enhanced_client, batch_processor=batch_processor, auto_process_interval=auto_process_seconds
    )


# Example pharmaceutical research workflow
async def pharmaceutical_research_workflow_example():
    """Example pharmaceutical research workflow using batch optimization."""

    async with await create_pharmaceutical_research_session(auto_process_seconds=20) as client:
        # High priority drug safety query
        safety_request = await client.queue_chat_request(
            messages=[
                {
                    "role": "user",
                    "content": "What are the contraindications and drug interactions for metformin in elderly patients?",
                }
            ],
            priority=RequestPriority.CRITICAL,
            pharmaceutical_context={
                "query": "metformin contraindications drug interactions elderly",
                "domain": "pharmaceutical_safety",
                "research_type": "drug_safety",
            },
        )

        # Normal priority mechanism query
        mechanism_request = await client.queue_embedding_request(
            texts=[
                "ACE inhibitor mechanism of action in hypertension treatment",
                "beta-blocker cardioselective properties and clinical applications",
                "calcium channel blocker pharmacokinetics and drug metabolism",
            ],
            priority=RequestPriority.NORMAL,
            pharmaceutical_context={
                "query": "cardiovascular drug mechanisms",
                "domain": "pharmacology",
                "research_type": "mechanism_analysis",
            },
        )

        # Process batches and get results
        result = await client.process_batches_now(max_concurrent=2)

        if result.success:
            print("Pharmaceutical research batch completed successfully")
            print(f"Processed {len(result.results)} batches")
            print(
                f"Free tier utilization: {result.metrics.get('pharmaceutical_optimization', {}).get('free_tier_utilization', {})}"
            )

        return result


if __name__ == "__main__":
    # Run pharmaceutical research workflow example
    asyncio.run(pharmaceutical_research_workflow_example())
