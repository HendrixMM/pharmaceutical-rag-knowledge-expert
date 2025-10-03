"""
Embedding Performance Monitor & Fallback System

Comprehensive performance monitoring and intelligent fallback management for
NVIDIA NeMo embedding services with pharmaceutical domain optimization.

Features:
1. Real-time performance monitoring and alerting
2. Intelligent fallback routing and circuit breaker patterns
3. Health checks and service availability monitoring
4. Performance trend analysis and optimization recommendations
5. Pharmaceutical-specific performance metrics
6. SLA monitoring and compliance reporting
7. Automated recovery and failover management

<<use_mcp microsoft-learn>>
"""
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Health status of embedding services."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"
    UNKNOWN = "unknown"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetrics:
    """Performance metrics for embedding services."""

    service_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    pharmaceutical_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_health_check: float = 0.0
    health_status: ServiceHealth = ServiceHealth.UNKNOWN


@dataclass
class PerformanceAlert:
    """Performance alert information."""

    timestamp: float
    service_name: str
    alert_level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False


@dataclass
class FallbackRoute:
    """Fallback routing configuration."""

    primary_service: str
    fallback_services: List[str]
    fallback_criteria: Dict[str, float]
    circuit_breaker_threshold: float = 0.5
    recovery_threshold: float = 0.8
    cooldown_period_seconds: int = 300


@dataclass
class SLAConfiguration:
    """Service Level Agreement configuration."""

    max_response_time_ms: float = 5000.0
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05
    min_availability: float = 0.99
    pharmaceutical_response_time_ms: float = 3000.0  # Stricter for pharma


class EmbeddingPerformanceMonitor:
    """
    Comprehensive performance monitoring system for embedding services.

    Provides real-time monitoring, intelligent fallback management, and
    performance optimization for pharmaceutical RAG systems.
    """

    def __init__(
        self,
        sla_config: Optional[SLAConfiguration] = None,
        enable_alerting: bool = True,
        enable_circuit_breaker: bool = True,
        health_check_interval_seconds: int = 60,
    ):
        """
        Initialize performance monitor.

        Args:
            sla_config: Service level agreement configuration
            enable_alerting: Enable performance alerting
            enable_circuit_breaker: Enable circuit breaker pattern
            health_check_interval_seconds: Health check frequency
        """
        self.sla_config = sla_config or SLAConfiguration()
        self.enable_alerting = enable_alerting
        self.enable_circuit_breaker = enable_circuit_breaker
        self.health_check_interval = health_check_interval_seconds

        # Service metrics tracking
        self.service_metrics: Dict[str, PerformanceMetrics] = {}

        # Fallback routing configuration
        self.fallback_routes: Dict[str, FallbackRoute] = {}

        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Performance alerts
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: deque = deque(maxlen=10000)

        # Performance thresholds
        self.performance_thresholds = {
            "response_time_warning_ms": 2000.0,
            "response_time_critical_ms": 5000.0,
            "error_rate_warning": 0.02,
            "error_rate_critical": 0.05,
            "success_rate_warning": 0.98,
            "success_rate_critical": 0.95,
            "pharmaceutical_response_time_critical_ms": 3000.0,
        }

        # Health check functions
        self.health_check_functions: Dict[str, Callable] = {}

        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None

        # Performance trends
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2880))  # 48 hours @ 1min intervals

        logger.info("Initialized Embedding Performance Monitor")

    def register_service(
        self,
        service_name: str,
        health_check_function: Optional[Callable] = None,
        fallback_config: Optional[FallbackRoute] = None,
    ):
        """
        Register a service for monitoring.

        Args:
            service_name: Name of the service to monitor
            health_check_function: Function to check service health
            fallback_config: Fallback routing configuration
        """
        self.service_metrics[service_name] = PerformanceMetrics(service_name=service_name)

        if health_check_function:
            self.health_check_functions[service_name] = health_check_function

        if fallback_config:
            self.fallback_routes[service_name] = fallback_config

        # Initialize circuit breaker
        if self.enable_circuit_breaker:
            self.circuit_breakers[service_name] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "last_failure_time": 0,
                "next_attempt_time": 0,
            }

        logger.info(f"Registered service for monitoring: {service_name}")

    def record_request(
        self,
        service_name: str,
        response_time_ms: float,
        success: bool,
        is_pharmaceutical: bool = False,
        cache_hit: bool = False,
    ):
        """
        Record a request for performance tracking.

        Args:
            service_name: Name of the service
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            is_pharmaceutical: Whether this was pharmaceutical content
            cache_hit: Whether this was a cache hit
        """
        if service_name not in self.service_metrics:
            self.register_service(service_name)

        metrics = self.service_metrics[service_name]

        # Update basic metrics
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Track response times
        metrics.response_times.append(response_time_ms)

        # Track pharmaceutical requests
        if is_pharmaceutical:
            metrics.pharmaceutical_requests += 1

        # Track cache performance
        if cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1

        # Calculate current error rate
        recent_total = min(metrics.total_requests, 100)
        error_rate = metrics.failed_requests / max(recent_total, 1)
        metrics.error_rates.append(error_rate)

        # Update circuit breaker state
        if self.enable_circuit_breaker:
            self._update_circuit_breaker(service_name, success)

        # Check for performance alerts
        if self.enable_alerting:
            self._check_performance_alerts(service_name, response_time_ms, success, is_pharmaceutical)

        # Update performance trends
        self._update_performance_trends(service_name)

    def get_service_health(self, service_name: str) -> ServiceHealth:
        """
        Get current health status of a service.

        Args:
            service_name: Name of the service

        Returns:
            Current health status
        """
        if service_name not in self.service_metrics:
            return ServiceHealth.UNKNOWN

        metrics = self.service_metrics[service_name]

        # Check circuit breaker state
        if self.enable_circuit_breaker:
            cb_state = self.circuit_breakers.get(service_name, {}).get("state", "closed")
            if cb_state == "open":
                return ServiceHealth.DOWN

        # Calculate health score based on recent performance
        if not metrics.response_times:
            return ServiceHealth.UNKNOWN

        # Recent response time
        recent_response_times = list(metrics.response_times)[-10:]
        avg_response_time = statistics.mean(recent_response_times)

        # Recent error rate
        recent_error_rate = metrics.error_rates[-1] if metrics.error_rates else 0

        # Health assessment
        if (
            avg_response_time <= self.performance_thresholds["response_time_warning_ms"]
            and recent_error_rate <= self.performance_thresholds["error_rate_warning"]
        ):
            return ServiceHealth.HEALTHY
        elif (
            avg_response_time <= self.performance_thresholds["response_time_critical_ms"]
            and recent_error_rate <= self.performance_thresholds["error_rate_critical"]
        ):
            return ServiceHealth.DEGRADED
        elif recent_error_rate > self.performance_thresholds["error_rate_critical"]:
            return ServiceHealth.UNHEALTHY
        else:
            return ServiceHealth.DOWN

    def get_optimal_service(self, primary_service: str, is_pharmaceutical: bool = False) -> Tuple[str, str]:
        """
        Get the optimal service to use considering health and fallbacks.

        Args:
            primary_service: Preferred primary service
            is_pharmaceutical: Whether this is pharmaceutical content

        Returns:
            Tuple of (selected_service, selection_reason)
        """
        # Check if primary service is healthy
        primary_health = self.get_service_health(primary_service)

        if primary_health in [ServiceHealth.HEALTHY, ServiceHealth.DEGRADED]:
            return primary_service, "primary_service_healthy"

        # Check for fallback routes
        if primary_service in self.fallback_routes:
            fallback_config = self.fallback_routes[primary_service]

            for fallback_service in fallback_config.fallback_services:
                fallback_health = self.get_service_health(fallback_service)

                if fallback_health == ServiceHealth.HEALTHY:
                    reason = f"fallback_to_{fallback_service}_primary_unhealthy"
                    return fallback_service, reason

        # If pharmaceutical content, apply stricter criteria
        if is_pharmaceutical:
            # Try to find any healthy service for pharmaceutical content
            for service_name in self.service_metrics:
                if self.get_service_health(service_name) == ServiceHealth.HEALTHY:
                    return service_name, "pharmaceutical_emergency_fallback"

        # Last resort: return primary service with warning
        return primary_service, "no_healthy_alternative_available"

    def _update_circuit_breaker(self, service_name: str, success: bool):
        """Update circuit breaker state based on request outcome."""
        if service_name not in self.circuit_breakers:
            return

        cb = self.circuit_breakers[service_name]
        current_time = time.time()

        if cb["state"] == "closed":
            if not success:
                cb["failure_count"] += 1
                cb["last_failure_time"] = current_time

                # Check if we should open the circuit
                fallback_config = self.fallback_routes.get(service_name)
                threshold = fallback_config.circuit_breaker_threshold if fallback_config else 0.5

                if cb["failure_count"] >= 5:  # Minimum failures before considering
                    metrics = self.service_metrics[service_name]
                    recent_error_rate = metrics.error_rates[-1] if metrics.error_rates else 0

                    if recent_error_rate >= threshold:
                        cb["state"] = "open"
                        cb["next_attempt_time"] = current_time + 300  # 5 minutes
                        self._create_alert(
                            service_name,
                            AlertLevel.CRITICAL,
                            f"Circuit breaker opened for {service_name}",
                            "circuit_breaker",
                            recent_error_rate,
                            threshold,
                        )
            else:
                # Reset failure count on success
                cb["failure_count"] = max(0, cb["failure_count"] - 1)

        elif cb["state"] == "open":
            if current_time >= cb["next_attempt_time"]:
                cb["state"] = "half-open"
                cb["failure_count"] = 0

        elif cb["state"] == "half-open":
            if success:
                cb["state"] = "closed"
                cb["failure_count"] = 0
                self._create_alert(
                    service_name,
                    AlertLevel.INFO,
                    f"Circuit breaker closed for {service_name} - service recovered",
                    "circuit_breaker",
                    0,
                    0,
                )
            else:
                cb["state"] = "open"
                cb["failure_count"] += 1
                cb["next_attempt_time"] = current_time + 300

    def _check_performance_alerts(
        self, service_name: str, response_time_ms: float, success: bool, is_pharmaceutical: bool
    ):
        """Check for performance threshold violations and generate alerts."""

        # Response time alerts
        if is_pharmaceutical:
            if response_time_ms > self.performance_thresholds["pharmaceutical_response_time_critical_ms"]:
                self._create_alert(
                    service_name,
                    AlertLevel.CRITICAL,
                    f"Pharmaceutical content response time exceeded critical threshold",
                    "pharmaceutical_response_time",
                    response_time_ms,
                    self.performance_thresholds["pharmaceutical_response_time_critical_ms"],
                )
        else:
            if response_time_ms > self.performance_thresholds["response_time_critical_ms"]:
                self._create_alert(
                    service_name,
                    AlertLevel.CRITICAL,
                    f"Response time exceeded critical threshold",
                    "response_time",
                    response_time_ms,
                    self.performance_thresholds["response_time_critical_ms"],
                )
            elif response_time_ms > self.performance_thresholds["response_time_warning_ms"]:
                self._create_alert(
                    service_name,
                    AlertLevel.WARNING,
                    f"Response time exceeded warning threshold",
                    "response_time",
                    response_time_ms,
                    self.performance_thresholds["response_time_warning_ms"],
                )

        # Error rate alerts
        metrics = self.service_metrics[service_name]
        if metrics.error_rates:
            current_error_rate = metrics.error_rates[-1]

            if current_error_rate > self.performance_thresholds["error_rate_critical"]:
                self._create_alert(
                    service_name,
                    AlertLevel.CRITICAL,
                    f"Error rate exceeded critical threshold",
                    "error_rate",
                    current_error_rate,
                    self.performance_thresholds["error_rate_critical"],
                )
            elif current_error_rate > self.performance_thresholds["error_rate_warning"]:
                self._create_alert(
                    service_name,
                    AlertLevel.WARNING,
                    f"Error rate exceeded warning threshold",
                    "error_rate",
                    current_error_rate,
                    self.performance_thresholds["error_rate_warning"],
                )

    def _create_alert(
        self,
        service_name: str,
        level: AlertLevel,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ):
        """Create a new performance alert."""
        alert = PerformanceAlert(
            timestamp=time.time(),
            service_name=service_name,
            alert_level=level,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        logger.warning(f"Performance Alert [{level.value.upper()}] {service_name}: {message}")

        # Auto-resolve info alerts after 5 minutes
        if level == AlertLevel.INFO:
            # Schedule alert resolution (simplified)
            pass

    def _update_performance_trends(self, service_name: str):
        """Update performance trend data."""
        metrics = self.service_metrics[service_name]

        if metrics.response_times and metrics.error_rates:
            trend_data = {
                "timestamp": time.time(),
                "avg_response_time": statistics.mean(list(metrics.response_times)[-10:]),
                "error_rate": metrics.error_rates[-1],
                "total_requests": metrics.total_requests,
                "cache_hit_rate": metrics.cache_hits / max(metrics.total_requests, 1),
            }
            self.performance_trends[service_name].append(trend_data)

    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started performance monitoring thread")

    def stop_monitoring(self):
        """Stop background monitoring thread."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped performance monitoring thread")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Perform health checks
                for service_name, health_check_func in self.health_check_functions.items():
                    try:
                        health_status = health_check_func()
                        self.service_metrics[service_name].health_status = health_status
                        self.service_metrics[service_name].last_health_check = time.time()
                    except Exception as e:
                        logger.warning(f"Health check failed for {service_name}: {e}")
                        self.service_metrics[service_name].health_status = ServiceHealth.UNKNOWN

                # Clean up old alerts
                current_time = time.time()
                self.active_alerts = [
                    alert for alert in self.active_alerts if current_time - alert.timestamp < 3600  # Keep for 1 hour
                ]

                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "overall_health": "unknown",
            "active_alerts": len(self.active_alerts),
            "sla_compliance": {},
        }

        healthy_services = 0
        total_services = len(self.service_metrics)

        for service_name, metrics in self.service_metrics.items():
            health = self.get_service_health(service_name)

            service_report = {
                "health_status": health.value,
                "total_requests": metrics.total_requests,
                "success_rate": metrics.successful_requests / max(metrics.total_requests, 1),
                "avg_response_time_ms": statistics.mean(metrics.response_times) if metrics.response_times else 0,
                "error_rate": metrics.error_rates[-1] if metrics.error_rates else 0,
                "pharmaceutical_requests": metrics.pharmaceutical_requests,
                "cache_hit_rate": metrics.cache_hits / max(metrics.total_requests, 1),
                "circuit_breaker_state": self.circuit_breakers.get(service_name, {}).get("state", "unknown"),
            }

            # SLA compliance check
            sla_compliant = self._check_sla_compliance(service_name, service_report)
            service_report["sla_compliant"] = sla_compliant

            report["services"][service_name] = service_report

            if health == ServiceHealth.HEALTHY:
                healthy_services += 1

        # Overall health assessment
        if total_services == 0:
            report["overall_health"] = "unknown"
        elif healthy_services == total_services:
            report["overall_health"] = "healthy"
        elif healthy_services >= total_services * 0.7:
            report["overall_health"] = "degraded"
        else:
            report["overall_health"] = "unhealthy"

        # Recent alerts
        recent_alerts = [
            {
                "timestamp": alert.timestamp,
                "service": alert.service_name,
                "level": alert.alert_level.value,
                "message": alert.message,
                "metric": alert.metric_name,
            }
            for alert in self.active_alerts[-10:]
        ]
        report["recent_alerts"] = recent_alerts

        return report

    def _check_sla_compliance(self, service_name: str, service_report: Dict[str, Any]) -> bool:
        """Check if service is meeting SLA requirements."""

        # Check success rate
        if service_report["success_rate"] < self.sla_config.min_success_rate:
            return False

        # Check response time
        max_response_time = (
            self.sla_config.pharmaceutical_response_time_ms
            if service_report["pharmaceutical_requests"] > 0
            else self.sla_config.max_response_time_ms
        )

        if service_report["avg_response_time_ms"] > max_response_time:
            return False

        # Check error rate
        if service_report["error_rate"] > self.sla_config.max_error_rate:
            return False

        return True

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on performance data.

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        for service_name, metrics in self.service_metrics.items():
            if not metrics.response_times:
                continue

            avg_response_time = statistics.mean(metrics.response_times)
            cache_hit_rate = metrics.cache_hits / max(metrics.total_requests, 1)
            error_rate = metrics.error_rates[-1] if metrics.error_rates else 0

            # Response time recommendations
            if avg_response_time > 3000:
                recommendations.append(
                    {
                        "service": service_name,
                        "type": "performance",
                        "priority": "high",
                        "recommendation": "Consider optimizing model selection or increasing batch size",
                        "metric": "response_time",
                        "current_value": avg_response_time,
                    }
                )

            # Cache recommendations
            if cache_hit_rate < 0.3:
                recommendations.append(
                    {
                        "service": service_name,
                        "type": "caching",
                        "priority": "medium",
                        "recommendation": "Increase cache TTL or cache size to improve hit rate",
                        "metric": "cache_hit_rate",
                        "current_value": cache_hit_rate,
                    }
                )

            # Error rate recommendations
            if error_rate > 0.02:
                recommendations.append(
                    {
                        "service": service_name,
                        "type": "reliability",
                        "priority": "high",
                        "recommendation": "Investigate error patterns and improve fallback mechanisms",
                        "metric": "error_rate",
                        "current_value": error_rate,
                    }
                )

        return recommendations


# Global instance for easy access
performance_monitor = EmbeddingPerformanceMonitor()


def monitor_embedding_request(
    service_name: str, response_time_ms: float, success: bool, is_pharmaceutical: bool = False, cache_hit: bool = False
):
    """
    Convenience function to record embedding request performance.

    Args:
        service_name: Name of the embedding service
        response_time_ms: Response time in milliseconds
        success: Whether the request was successful
        is_pharmaceutical: Whether this was pharmaceutical content
        cache_hit: Whether this was a cache hit
    """
    performance_monitor.record_request(
        service_name=service_name,
        response_time_ms=response_time_ms,
        success=success,
        is_pharmaceutical=is_pharmaceutical,
        cache_hit=cache_hit,
    )


def get_optimal_embedding_service(primary_service: str, is_pharmaceutical: bool = False) -> Tuple[str, str]:
    """
    Get the optimal embedding service considering health and fallbacks.

    Args:
        primary_service: Preferred primary service
        is_pharmaceutical: Whether this is pharmaceutical content

    Returns:
        Tuple of (selected_service, selection_reason)
    """
    return performance_monitor.get_optimal_service(primary_service=primary_service, is_pharmaceutical=is_pharmaceutical)
