"""
Alert Manager for Pharmaceutical Research

Implements daily credit burn rate alerts with pharmaceutical-specific metrics.
Integrates with centralized alerts configuration for maintainable thresholds.

Features:
- Daily/weekly/monthly burn rate monitoring
- Pharmaceutical query pattern analysis
- Research project budget alerts
- Cost optimization recommendations
- Configurable notification channels

Integration:
- Uses config/alerts.yaml for threshold management
- Integrates with PharmaceuticalCreditTracker
- Provides actionable pharmaceutical insights
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import yaml

try:
    from .credit_tracker import PharmaceuticalCreditTracker
except ImportError:
    from src.monitoring.credit_tracker import PharmaceuticalCreditTracker

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


@dataclass
class Alert:
    """Represents a system alert."""

    id: str
    severity: AlertSeverity
    category: str
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


class PharmaceuticalAlertManager:
    """
    Alert manager for pharmaceutical research with daily burn rate monitoring.

    Provides intelligent alerting based on pharmaceutical usage patterns,
    cost optimization opportunities, and research workflow efficiency.
    """

    def __init__(
        self, tracker: PharmaceuticalCreditTracker, config_path: Optional[str] = None, enable_notifications: bool = True
    ):
        """
        Initialize pharmaceutical alert manager.

        Args:
            tracker: Pharmaceutical credit tracker instance
            config_path: Path to alerts configuration file
            enable_notifications: Enable alert notifications
        """
        self.tracker = tracker
        self.enable_notifications = enable_notifications

        # Load alerts configuration
        self.config_path = Path(config_path or "config/alerts.yaml")
        self.config = self._load_alerts_config()

        # Alert state management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_check = {}

        # Notification channels
        self.notification_channels = []
        self._initialize_notification_channels()

        logger.info("Pharmaceutical alert manager initialized")

    def check_daily_burn_rate(self) -> List[Alert]:
        """
        Check daily credit burn rate and generate alerts if needed.

        Returns:
            List of alerts generated for daily burn rate issues
        """
        alerts = []

        try:
            # Get current usage analytics
            analytics = self.tracker.get_pharmaceutical_analytics()
            today_usage = analytics["time_period_analysis"]["today"]

            # Get thresholds from configuration
            monthly_limit = self.config["nvidia_build"]["monthly_free_requests"]
            daily_threshold = monthly_limit * self.config["nvidia_build"]["usage_alerts"]["daily_burn_rate"]

            # Check daily burn rate
            if today_usage >= daily_threshold:
                alert = Alert(
                    id=f"daily_burn_rate_{datetime.now().strftime('%Y%m%d')}",
                    severity=AlertSeverity.WARNING,
                    category="cost_monitoring",
                    title="Daily Credit Burn Rate Alert",
                    message=f"Daily usage ({today_usage} requests) exceeds threshold ({daily_threshold:.0f} requests). "
                    f"Current pace projects to {today_usage * 30} requests/month.",
                    timestamp=datetime.now(),
                    data={
                        "today_usage": today_usage,
                        "daily_threshold": daily_threshold,
                        "monthly_projection": today_usage * 30,
                        "monthly_limit": monthly_limit,
                        "pharmaceutical_analytics": analytics,
                    },
                )
                alerts.append(alert)

                # Add pharmaceutical-specific insights
                pharma_insights = self._generate_pharmaceutical_burn_rate_insights(analytics, today_usage)
                if pharma_insights:
                    alert.data["pharmaceutical_insights"] = pharma_insights

        except Exception as e:
            logger.error(f"Error checking daily burn rate: {str(e)}")

        return alerts

    def check_pharmaceutical_project_budgets(self) -> List[Alert]:
        """
        Check research project budgets and generate alerts.

        Returns:
            List of alerts for project budget issues
        """
        alerts = []

        try:
            warning_threshold = self.config["pharmaceutical"]["project_budget"]["warning_threshold"]
            critical_threshold = self.config["pharmaceutical"]["project_budget"]["critical_threshold"]

            for project_id, project in self.tracker.research_projects.items():
                usage_percentage = project.queries_used / project.budget_limit

                # Warning threshold alert
                if usage_percentage >= warning_threshold and usage_percentage < critical_threshold:
                    alert = Alert(
                        id=f"project_budget_warning_{project_id}",
                        severity=AlertSeverity.WARNING,
                        category="budget_management",
                        title=f"Project Budget Warning: {project.name}",
                        message=f"Project '{project.name}' has used {usage_percentage:.1%} of its budget "
                        f"({project.queries_used}/{project.budget_limit} requests).",
                        timestamp=datetime.now(),
                        data={
                            "project_id": project_id,
                            "project_name": project.name,
                            "usage_percentage": usage_percentage,
                            "queries_used": project.queries_used,
                            "budget_limit": project.budget_limit,
                            "remaining_budget": project.budget_limit - project.queries_used,
                        },
                    )
                    alerts.append(alert)

                # Critical threshold alert
                elif usage_percentage >= critical_threshold:
                    alert = Alert(
                        id=f"project_budget_critical_{project_id}",
                        severity=AlertSeverity.CRITICAL,
                        category="budget_management",
                        title=f"Project Budget Critical: {project.name}",
                        message=f"Project '{project.name}' has used {usage_percentage:.1%} of its budget. "
                        f"Immediate attention required to avoid budget overrun.",
                        timestamp=datetime.now(),
                        data={
                            "project_id": project_id,
                            "project_name": project.name,
                            "usage_percentage": usage_percentage,
                            "queries_used": project.queries_used,
                            "budget_limit": project.budget_limit,
                            "remaining_budget": project.budget_limit - project.queries_used,
                            "recommended_actions": [
                                "Review remaining research priorities",
                                "Consider increasing project budget",
                                "Optimize query efficiency",
                                "Implement batch processing",
                            ],
                        },
                    )
                    alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking project budgets: {str(e)}")

        return alerts

    def check_pharmaceutical_query_efficiency(self) -> List[Alert]:
        """
        Check pharmaceutical query efficiency and generate optimization alerts.

        Returns:
            List of alerts for query efficiency issues
        """
        alerts = []

        try:
            analytics = self.tracker.get_pharmaceutical_analytics()

            # Check response time efficiency
            avg_response_time = analytics["performance_metrics"]["avg_response_time_ms"]
            max_acceptable_time = self.config["pharmaceutical"]["query_performance"]["max_acceptable_response_time_ms"]

            if avg_response_time > max_acceptable_time:
                alert = Alert(
                    id="query_response_time_inefficiency",
                    severity=AlertSeverity.INFO,
                    category="performance_optimization",
                    title="Query Response Time Optimization Opportunity",
                    message=f"Average response time ({avg_response_time:.0f}ms) exceeds optimal threshold "
                    f"({max_acceptable_time}ms). Consider optimization strategies.",
                    timestamp=datetime.now(),
                    data={
                        "avg_response_time_ms": avg_response_time,
                        "threshold_ms": max_acceptable_time,
                        "optimization_suggestions": [
                            "Enable batch processing for similar queries",
                            "Use cloud-first endpoints for better performance",
                            "Optimize query length and complexity",
                            "Consider caching for frequent pharmaceutical queries",
                        ],
                    },
                )
                alerts.append(alert)

            # Check token usage efficiency
            avg_tokens = analytics["performance_metrics"]["avg_tokens_per_query"]
            token_warning_threshold = self.config["pharmaceutical"]["query_performance"]["avg_tokens_warning"]

            if avg_tokens > token_warning_threshold:
                alert = Alert(
                    id="query_token_efficiency",
                    severity=AlertSeverity.INFO,
                    category="efficiency_optimization",
                    title="Query Token Usage Optimization",
                    message=f"Average tokens per query ({avg_tokens:.0f}) is higher than recommended "
                    f"({token_warning_threshold}). Consider more concise pharmaceutical queries.",
                    timestamp=datetime.now(),
                    data={
                        "avg_tokens_per_query": avg_tokens,
                        "recommended_threshold": token_warning_threshold,
                        "pharmaceutical_optimization_tips": [
                            "Use specific pharmaceutical terminology",
                            "Focus on single drug/interaction per query",
                            "Utilize domain-specific abbreviations",
                            "Structure queries with clear medical context",
                        ],
                    },
                )
                alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking query efficiency: {str(e)}")

        return alerts

    def check_cost_optimization_opportunities(self) -> List[Alert]:
        """
        Check for cost optimization opportunities with pharmaceutical focus.

        Returns:
            List of alerts for cost optimization opportunities
        """
        alerts = []

        try:
            analytics = self.tracker.get_pharmaceutical_analytics()
            recommendations = self.tracker.get_cost_optimization_recommendations()

            # Check free tier utilization
            free_tier_percentage = analytics["cost_analysis"]["cost_optimization_percentage"]
            min_free_tier = self.config["pharmaceutical"]["workflow_efficiency"]["min_free_tier_usage_percentage"]

            if free_tier_percentage < min_free_tier:
                high_priority_recs = [r for r in recommendations if r.get("priority") == "high"]

                alert = Alert(
                    id="cost_optimization_opportunity",
                    severity=AlertSeverity.INFO,
                    category="cost_optimization",
                    title="Free Tier Utilization Opportunity",
                    message=f"Currently utilizing free tier for {free_tier_percentage:.1f}% of queries. "
                    f"Target: {min_free_tier}% for optimal cost efficiency.",
                    timestamp=datetime.now(),
                    data={
                        "current_free_tier_percentage": free_tier_percentage,
                        "target_percentage": min_free_tier,
                        "optimization_recommendations": recommendations,
                        "immediate_actions": [r["action"] for r in high_priority_recs],
                        "potential_monthly_savings": self._calculate_potential_savings(analytics, min_free_tier),
                    },
                )
                alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking cost optimization: {str(e)}")

        return alerts

    def run_comprehensive_alert_check(self) -> List[Alert]:
        """
        Run comprehensive alert check across all monitoring categories.

        Returns:
            List of all active alerts
        """
        all_alerts = []

        # Check different alert categories
        alert_checks = [
            self.check_daily_burn_rate,
            self.check_pharmaceutical_project_budgets,
            self.check_pharmaceutical_query_efficiency,
            self.check_cost_optimization_opportunities,
        ]

        for check_func in alert_checks:
            try:
                alerts = check_func()
                all_alerts.extend(alerts)
            except Exception as e:
                logger.error(f"Error in alert check {check_func.__name__}: {str(e)}")

        # Process and notify for new alerts
        new_alerts = self._process_alerts(all_alerts)

        if new_alerts and self.enable_notifications:
            self._send_notifications(new_alerts)

        return all_alerts

    def generate_daily_summary(self) -> Dict[str, Any]:
        """
        Generate daily summary report with pharmaceutical insights.

        Returns:
            Daily summary with alerts and recommendations
        """
        analytics = self.tracker.get_pharmaceutical_analytics()
        active_alerts = self.run_comprehensive_alert_check()

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "pharmaceutical_analytics": analytics,
            "alert_summary": {
                "total_alerts": len(active_alerts),
                "by_severity": {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
                "by_category": self._group_alerts_by_category(active_alerts),
            },
            "cost_optimization": {
                "recommendations": self.tracker.get_cost_optimization_recommendations(),
                "potential_savings": self._calculate_optimization_potential(analytics),
            },
            "pharmaceutical_insights": self._generate_daily_pharmaceutical_insights(analytics),
            "action_items": self._generate_daily_action_items(active_alerts, analytics),
        }

    def _load_alerts_config(self) -> Dict[str, Any]:
        """Load alerts configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Alerts config not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading alerts config: {str(e)}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file not available."""
        return {
            "nvidia_build": {"monthly_free_requests": 10000, "usage_alerts": {"daily_burn_rate": 0.05}},
            "pharmaceutical": {
                "project_budget": {"warning_threshold": 0.75, "critical_threshold": 0.90},
                "query_performance": {"max_acceptable_response_time_ms": 3000, "avg_tokens_warning": 500},
                "workflow_efficiency": {"min_free_tier_usage_percentage": 80},
            },
        }

    def _initialize_notification_channels(self) -> None:
        """Initialize notification channels based on configuration."""
        if not self.enable_notifications:
            return

        config = self.config.get("alert_delivery", {})
        channels = config.get("channels", {})

        # Console logging channel (always enabled for basic alerts)
        if channels.get("console_logging", True):
            self.notification_channels.append(self._console_notification)

        # File logging channel
        if channels.get("file_logging", True):
            self.notification_channels.append(self._file_notification)

    def _process_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Process alerts and filter for new/updated alerts."""
        new_alerts = []

        for alert in alerts:
            existing_alert = self.active_alerts.get(alert.id)

            if not existing_alert:
                # New alert
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                new_alerts.append(alert)
                logger.info(f"New alert: {alert.title} ({alert.severity.value})")

            elif not existing_alert.acknowledged:
                # Update existing unacknowledged alert
                existing_alert.message = alert.message
                existing_alert.data = alert.data
                existing_alert.timestamp = alert.timestamp

        return new_alerts

    def _send_notifications(self, alerts: List[Alert]) -> None:
        """Send notifications for new alerts."""
        for alert in alerts:
            for channel in self.notification_channels:
                try:
                    channel(alert)
                except Exception as e:
                    logger.error(f"Error sending notification: {str(e)}")

    def _console_notification(self, alert: Alert) -> None:
        """Send console notification for alert."""
        severity_icons = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.URGENT: "🔥",
        }

        icon = severity_icons.get(alert.severity, "📢")
        logger.warning(f"{icon} [{alert.severity.value.upper()}] {alert.title}: {alert.message}")

    def _file_notification(self, alert: Alert) -> None:
        """Send file-based notification for alert."""
        try:
            alerts_dir = Path("logs/alerts")
            alerts_dir.mkdir(parents=True, exist_ok=True)

            alert_file = alerts_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"

            with open(alert_file, "a") as f:
                f.write(f"[{alert.timestamp.isoformat()}] {alert.severity.value.upper()} - {alert.title}\n")
                f.write(f"Message: {alert.message}\n")
                f.write(f"Category: {alert.category}\n")
                if alert.data:
                    f.write(f"Data: {alert.data}\n")
                f.write("-" * 80 + "\n")

        except Exception as e:
            logger.error(f"Error writing alert to file: {str(e)}")

    def _generate_pharmaceutical_burn_rate_insights(self, analytics: Dict[str, Any], today_usage: int) -> List[str]:
        """Generate pharmaceutical-specific burn rate insights."""
        insights = []

        # Query type analysis
        query_types = analytics.get("query_type_distribution", {})
        if query_types:
            most_common = max(query_types, key=query_types.get)
            insights.append(f"Most frequent query type today: {most_common} ({query_types[most_common]} queries)")

        # Cost tier analysis
        cost_analysis = analytics.get("cost_analysis", {})
        free_tier_percentage = cost_analysis.get("cost_optimization_percentage", 0)
        insights.append(f"Free tier utilization: {free_tier_percentage:.1f}% (target: 80%+)")

        return insights

    def _calculate_potential_savings(self, analytics: Dict[str, Any], target_percentage: float) -> Dict[str, Any]:
        """Calculate potential cost savings from optimization."""
        current_percentage = analytics["cost_analysis"]["cost_optimization_percentage"]
        monthly_queries = analytics["time_period_analysis"]["this_month"]

        if current_percentage < target_percentage:
            queries_to_optimize = monthly_queries * (target_percentage - current_percentage) / 100
            return {
                "queries_optimizable": int(queries_to_optimize),
                "percentage_improvement": target_percentage - current_percentage,
                "monthly_optimization_potential": f"{queries_to_optimize:.0f} additional free tier queries",
            }

        return {"optimization_potential": "Already optimized"}

    def _group_alerts_by_category(self, alerts: List[Alert]) -> Dict[str, int]:
        """Group alerts by category."""
        categories = {}
        for alert in alerts:
            categories[alert.category] = categories.get(alert.category, 0) + 1
        return categories

    def _calculate_optimization_potential(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall optimization potential."""
        return {
            "free_tier_optimization": analytics["cost_analysis"]["cost_optimization_percentage"],
            "performance_optimization": 100 - (analytics["performance_metrics"]["avg_response_time_ms"] / 3000 * 100),
            "cost_efficiency_score": analytics["cost_analysis"]["cost_optimization_percentage"],
        }

    def _generate_daily_pharmaceutical_insights(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate daily pharmaceutical insights."""
        insights = []

        # Usage patterns
        today_usage = analytics["time_period_analysis"]["today"]
        insights.append(f"Today's pharmaceutical queries: {today_usage}")

        # Query efficiency
        avg_response_time = analytics["performance_metrics"]["avg_response_time_ms"]
        insights.append(f"Average response time: {avg_response_time:.0f}ms")

        return insights

    def _generate_daily_action_items(self, alerts: List[Alert], analytics: Dict[str, Any]) -> List[str]:
        """Generate daily action items based on alerts and analytics."""
        action_items = []

        # High priority alerts
        high_priority_alerts = [a for a in alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.URGENT]]
        if high_priority_alerts:
            action_items.append(f"Address {len(high_priority_alerts)} high-priority alerts")

        # Cost optimization
        free_tier_percentage = analytics["cost_analysis"]["cost_optimization_percentage"]
        if free_tier_percentage < 80:
            action_items.append(f"Optimize free tier utilization (current: {free_tier_percentage:.1f}%)")

        # Performance optimization
        avg_response_time = analytics["performance_metrics"]["avg_response_time_ms"]
        if avg_response_time > 2000:
            action_items.append("Optimize query response times")

        return action_items


# Convenience function for pharmaceutical alert management
def create_pharmaceutical_alert_manager(tracker: PharmaceuticalCreditTracker) -> PharmaceuticalAlertManager:
    """
    Create pharmaceutical alert manager with default configuration.

    Args:
        tracker: Pharmaceutical credit tracker instance

    Returns:
        Configured PharmaceuticalAlertManager instance
    """
    return PharmaceuticalAlertManager(tracker)


if __name__ == "__main__":
    # Quick test of alert management
    from src.monitoring.credit_tracker import create_pharmaceutical_tracker

    tracker = create_pharmaceutical_tracker()
    alert_manager = create_pharmaceutical_alert_manager(tracker)

    # Run alert check
    alerts = alert_manager.run_comprehensive_alert_check()
    print(f"Generated {len(alerts)} alerts")

    # Generate daily summary
    summary = alert_manager.generate_daily_summary()
    print("Daily Summary Generated")
