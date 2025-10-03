"""
Drug Safety Alert Integration with Monitoring System

Integrates pharmaceutical query classification with the alert management system
to provide immediate drug safety alerts, regulatory compliance monitoring,
and automated escalation for critical pharmaceutical research queries.

Features:
- Real-time drug safety alert generation
- Integration with pharmaceutical query classifier
- Automatic escalation for critical safety queries
- Regulatory compliance monitoring
- Patient population safety warnings
- Drug interaction alert system

This system ensures critical pharmaceutical safety information is prioritized
and routed through the cloud-first architecture with appropriate urgency.
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

try:
    from ..enhanced_config import EnhancedRAGConfig
    from ..monitoring.alert_manager import PharmaceuticalAlertManager
    from .query_classifier import (
        PharmaceuticalContext,
        PharmaceuticalDomain,
        PharmaceuticalQueryClassifier,
        SafetyUrgency,
    )
except ImportError:
    from src.enhanced_config import EnhancedRAGConfig
    from src.monitoring.alert_manager import PharmaceuticalAlertManager
    from src.pharmaceutical.query_classifier import (
        PharmaceuticalContext,
        PharmaceuticalDomain,
        PharmaceuticalQueryClassifier,
        SafetyUrgency,
    )

logger = logging.getLogger(__name__)


class SafetyAlertType(Enum):
    """Types of drug safety alerts."""

    CRITICAL_SAFETY = "critical_safety"
    DRUG_INTERACTION = "drug_interaction"
    ADVERSE_REACTION = "adverse_reaction"
    CONTRAINDICATION = "contraindication"
    DOSAGE_SAFETY = "dosage_safety"
    PATIENT_POPULATION = "patient_population"
    REGULATORY_WARNING = "regulatory_warning"
    MONITORING_REQUIREMENT = "monitoring_requirement"


class AlertUrgency(Enum):
    """Alert urgency levels for drug safety."""

    IMMEDIATE = "immediate"  # Requires immediate attention
    URGENT = "urgent"  # Requires attention within hours
    ROUTINE = "routine"  # Standard safety information
    INFORMATIONAL = "informational"  # Background safety data


@dataclass
class DrugSafetyAlert:
    """Drug safety alert with comprehensive context."""

    alert_id: str
    alert_type: SafetyAlertType
    urgency: AlertUrgency
    query_text: str
    pharmaceutical_context: PharmaceuticalContext
    safety_message: str
    drug_names: List[str]
    patient_populations: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledgment_timestamp: Optional[datetime] = None


class DrugSafetyAlertIntegration:
    """
    Integrated drug safety alert system with pharmaceutical query classification.

    Monitors pharmaceutical queries for safety implications and automatically
    generates appropriate alerts with routing optimization.
    """

    def __init__(
        self, config: Optional[EnhancedRAGConfig] = None, alert_manager: Optional[PharmaceuticalAlertManager] = None
    ):
        """
        Initialize drug safety alert integration.

        Args:
            config: Enhanced RAG configuration
            alert_manager: Pharmaceutical alert manager
        """
        self.config = config or EnhancedRAGConfig.from_env()
        self.alert_manager = alert_manager or PharmaceuticalAlertManager()
        self.query_classifier = PharmaceuticalQueryClassifier()

        # Safety alert tracking
        self.active_safety_alerts: Dict[str, DrugSafetyAlert] = {}
        self.alert_history: List[DrugSafetyAlert] = []

        # Drug safety databases (simplified for demonstration)
        self.drug_safety_database = {
            "contraindications": {
                "metformin": ["severe kidney disease", "acute heart failure", "metabolic acidosis"],
                "warfarin": ["pregnancy", "active bleeding", "recent surgery"],
                "ace_inhibitors": ["pregnancy", "bilateral renal artery stenosis", "hyperkalemia"],
            },
            "critical_interactions": {
                "warfarin": ["aspirin", "nsaids", "antibiotics"],
                "metformin": ["contrast agents", "alcohol"],
                "digoxin": ["diuretics", "calcium channel blockers"],
            },
            "monitoring_requirements": {
                "warfarin": ["inr", "bleeding signs"],
                "metformin": ["kidney function", "lactic acid"],
                "statins": ["liver enzymes", "muscle symptoms"],
            },
        }

        # Patient population safety warnings
        self.population_warnings = {
            "elderly": ["increased sensitivity", "reduced clearance", "polypharmacy"],
            "pediatric": ["dosing adjustments", "safety profile", "development effects"],
            "pregnancy": ["teratogenic risk", "category classification", "fetal effects"],
            "kidney_disease": ["dosing adjustment", "accumulation risk", "toxicity"],
            "liver_disease": ["metabolism impairment", "hepatotoxicity", "clearance"],
        }

        # Safety alert callbacks
        self.alert_callbacks: List[Callable] = []

        logger.info("DrugSafetyAlertIntegration initialized with comprehensive safety monitoring")

    async def process_pharmaceutical_query(
        self, query_text: str, user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[PharmaceuticalContext, List[DrugSafetyAlert]]:
        """
        Process pharmaceutical query with safety alert generation.

        Args:
            query_text: The pharmaceutical research query
            user_context: Additional user context (research type, urgency, etc.)

        Returns:
            Tuple of pharmaceutical context and generated safety alerts
        """
        # Classify the pharmaceutical query
        pharmaceutical_context = self.query_classifier.classify_query(query_text)

        # Generate safety alerts based on classification
        safety_alerts = await self._generate_safety_alerts(query_text, pharmaceutical_context, user_context)

        # Process and route alerts
        await self._process_safety_alerts(safety_alerts)

        logger.info(f"Processed pharmaceutical query: {len(safety_alerts)} safety alerts generated")

        return pharmaceutical_context, safety_alerts

    async def _generate_safety_alerts(
        self, query_text: str, context: PharmaceuticalContext, user_context: Optional[Dict[str, Any]]
    ) -> List[DrugSafetyAlert]:
        """Generate appropriate safety alerts based on query analysis."""
        alerts = []

        # Critical safety urgency generates immediate alerts
        if context.safety_urgency == SafetyUrgency.CRITICAL:
            alert = await self._create_critical_safety_alert(query_text, context)
            alerts.append(alert)

        # Drug interaction alerts
        if context.domain == PharmaceuticalDomain.DRUG_INTERACTIONS:
            interaction_alerts = await self._generate_drug_interaction_alerts(query_text, context)
            alerts.extend(interaction_alerts)

        # Adverse reaction monitoring
        if context.domain == PharmaceuticalDomain.ADVERSE_REACTIONS:
            adverse_alert = await self._create_adverse_reaction_alert(query_text, context)
            alerts.append(adverse_alert)

        # Patient population safety warnings
        if context.patient_population:
            population_alert = await self._generate_population_safety_alert(query_text, context)
            if population_alert:
                alerts.append(population_alert)

        # Drug-specific safety alerts
        for drug_name in context.drug_names:
            drug_alerts = await self._generate_drug_specific_alerts(query_text, drug_name, context)
            alerts.extend(drug_alerts)

        # Regulatory compliance alerts
        if context.domain == PharmaceuticalDomain.REGULATORY_COMPLIANCE:
            regulatory_alert = await self._create_regulatory_compliance_alert(query_text, context)
            if regulatory_alert:
                alerts.append(regulatory_alert)

        return alerts

    async def _create_critical_safety_alert(self, query_text: str, context: PharmaceuticalContext) -> DrugSafetyAlert:
        """Create critical drug safety alert."""
        alert_id = self._generate_alert_id("critical_safety", query_text)

        recommendations = [
            "IMMEDIATE ATTENTION REQUIRED - Patient safety implications",
            "Consult drug safety database for complete information",
            "Consider immediate medical consultation if patient-related",
            "Document safety query and response for compliance",
        ]

        if context.drug_names:
            recommendations.append(f"Review complete safety profile for: {', '.join(context.drug_names)}")

        return DrugSafetyAlert(
            alert_id=alert_id,
            alert_type=SafetyAlertType.CRITICAL_SAFETY,
            urgency=AlertUrgency.IMMEDIATE,
            query_text=query_text,
            pharmaceutical_context=context,
            safety_message="CRITICAL DRUG SAFETY QUERY DETECTED - Immediate review required",
            drug_names=context.drug_names,
            patient_populations=[context.patient_population] if context.patient_population else [],
            recommendations=recommendations,
        )

    async def _generate_drug_interaction_alerts(
        self, query_text: str, context: PharmaceuticalContext
    ) -> List[DrugSafetyAlert]:
        """Generate drug interaction safety alerts."""
        alerts = []

        for drug_name in context.drug_names:
            drug_lower = drug_name.lower()

            # Check for known critical interactions
            if drug_lower in self.drug_safety_database["critical_interactions"]:
                interacting_drugs = self.drug_safety_database["critical_interactions"][drug_lower]

                # Check if query mentions any interacting drugs
                query_lower = query_text.lower()
                mentioned_interactions = [drug for drug in interacting_drugs if drug in query_lower]

                if mentioned_interactions:
                    alert_id = self._generate_alert_id("drug_interaction", f"{drug_name}_{mentioned_interactions[0]}")

                    alert = DrugSafetyAlert(
                        alert_id=alert_id,
                        alert_type=SafetyAlertType.DRUG_INTERACTION,
                        urgency=AlertUrgency.URGENT,
                        query_text=query_text,
                        pharmaceutical_context=context,
                        safety_message=f"DRUG INTERACTION ALERT: {drug_name} with {', '.join(mentioned_interactions)}",
                        drug_names=[drug_name] + mentioned_interactions,
                        patient_populations=[context.patient_population] if context.patient_population else [],
                        recommendations=[
                            "Review complete drug interaction profile",
                            "Consider alternative medications if appropriate",
                            "Monitor for interaction-related adverse effects",
                            "Adjust dosing if interaction is unavoidable",
                        ],
                    )
                    alerts.append(alert)

        return alerts

    async def _create_adverse_reaction_alert(self, query_text: str, context: PharmaceuticalContext) -> DrugSafetyAlert:
        """Create adverse reaction monitoring alert."""
        alert_id = self._generate_alert_id("adverse_reaction", query_text)

        return DrugSafetyAlert(
            alert_id=alert_id,
            alert_type=SafetyAlertType.ADVERSE_REACTION,
            urgency=AlertUrgency.URGENT,
            query_text=query_text,
            pharmaceutical_context=context,
            safety_message="ADVERSE REACTION QUERY - Enhanced monitoring recommended",
            drug_names=context.drug_names,
            patient_populations=[context.patient_population] if context.patient_population else [],
            recommendations=[
                "Document adverse reaction details thoroughly",
                "Consider pharmacovigilance reporting if applicable",
                "Review patient risk factors for adverse events",
                "Monitor patient closely for additional reactions",
            ],
        )

    async def _generate_population_safety_alert(
        self, query_text: str, context: PharmaceuticalContext
    ) -> Optional[DrugSafetyAlert]:
        """Generate patient population-specific safety alert."""
        population = context.patient_population
        if not population or population not in self.population_warnings:
            return None

        population_risks = self.population_warnings[population]
        alert_id = self._generate_alert_id(
            "patient_population", f"{population}_{context.drug_names[0] if context.drug_names else 'general'}"
        )

        recommendations = [
            f"Special considerations for {population} population required",
            f"Review {population}-specific dosing guidelines",
            f"Monitor for {population}-specific adverse effects",
        ]

        # Add population-specific recommendations
        if "dosing" in population_risks[0].lower():
            recommendations.append("Dose adjustment may be necessary")
        if "monitoring" in " ".join(population_risks).lower():
            recommendations.append("Enhanced monitoring protocols recommended")

        return DrugSafetyAlert(
            alert_id=alert_id,
            alert_type=SafetyAlertType.PATIENT_POPULATION,
            urgency=AlertUrgency.ROUTINE,
            query_text=query_text,
            pharmaceutical_context=context,
            safety_message=f"PATIENT POPULATION ALERT: {population} considerations for {', '.join(context.drug_names) if context.drug_names else 'medication'}",
            drug_names=context.drug_names,
            patient_populations=[population],
            recommendations=recommendations,
        )

    async def _generate_drug_specific_alerts(
        self, query_text: str, drug_name: str, context: PharmaceuticalContext
    ) -> List[DrugSafetyAlert]:
        """Generate drug-specific safety alerts."""
        alerts = []
        drug_lower = drug_name.lower()

        # Contraindication alerts
        if drug_lower in self.drug_safety_database["contraindications"]:
            contraindications = self.drug_safety_database["contraindications"][drug_lower]

            # Check if query mentions contraindications
            query_lower = query_text.lower()
            relevant_contraindications = [
                contra for contra in contraindications if any(word in query_lower for word in contra.split())
            ]

            if relevant_contraindications:
                alert_id = self._generate_alert_id("contraindication", f"{drug_name}_contraindication")

                alert = DrugSafetyAlert(
                    alert_id=alert_id,
                    alert_type=SafetyAlertType.CONTRAINDICATION,
                    urgency=AlertUrgency.URGENT,
                    query_text=query_text,
                    pharmaceutical_context=context,
                    safety_message=f"CONTRAINDICATION ALERT: {drug_name} contraindicated in {', '.join(relevant_contraindications)}",
                    drug_names=[drug_name],
                    patient_populations=[context.patient_population] if context.patient_population else [],
                    recommendations=[
                        f"Verify patient does not have {', '.join(relevant_contraindications)}",
                        "Consider alternative medication if contraindication present",
                        "Document contraindication assessment",
                    ],
                )
                alerts.append(alert)

        # Monitoring requirement alerts
        if drug_lower in self.drug_safety_database["monitoring_requirements"]:
            monitoring_reqs = self.drug_safety_database["monitoring_requirements"][drug_lower]

            alert_id = self._generate_alert_id("monitoring", f"{drug_name}_monitoring")

            alert = DrugSafetyAlert(
                alert_id=alert_id,
                alert_type=SafetyAlertType.MONITORING_REQUIREMENT,
                urgency=AlertUrgency.ROUTINE,
                query_text=query_text,
                pharmaceutical_context=context,
                safety_message=f"MONITORING REQUIRED: {drug_name} requires monitoring of {', '.join(monitoring_reqs)}",
                drug_names=[drug_name],
                patient_populations=[context.patient_population] if context.patient_population else [],
                recommendations=[
                    f"Establish baseline monitoring for: {', '.join(monitoring_reqs)}",
                    "Schedule regular monitoring intervals",
                    "Educate patient on monitoring importance",
                    "Document monitoring plan and results",
                ],
            )
            alerts.append(alert)

        return alerts

    async def _create_regulatory_compliance_alert(
        self, query_text: str, context: PharmaceuticalContext
    ) -> Optional[DrugSafetyAlert]:
        """Create regulatory compliance monitoring alert."""
        if not context.regulatory_context:
            return None

        alert_id = self._generate_alert_id("regulatory", context.regulatory_context)

        return DrugSafetyAlert(
            alert_id=alert_id,
            alert_type=SafetyAlertType.REGULATORY_WARNING,
            urgency=AlertUrgency.ROUTINE,
            query_text=query_text,
            pharmaceutical_context=context,
            safety_message=f"REGULATORY COMPLIANCE: {context.regulatory_context} considerations required",
            drug_names=context.drug_names,
            patient_populations=[context.patient_population] if context.patient_population else [],
            recommendations=[
                "Review current regulatory guidelines",
                "Ensure compliance with prescribing requirements",
                "Document regulatory compliance considerations",
            ],
        )

    async def _process_safety_alerts(self, alerts: List[DrugSafetyAlert]) -> None:
        """Process and route safety alerts through monitoring system."""
        for alert in alerts:
            # Add to active alerts
            self.active_safety_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)

            # Route through pharmaceutical alert manager
            await self._route_safety_alert(alert)

            # Execute callbacks
            await self._execute_alert_callbacks(alert)

            logger.info(f"Processed safety alert: {alert.alert_type.value} - {alert.urgency.value}")

    async def _route_safety_alert(self, alert: DrugSafetyAlert) -> None:
        """Route safety alert through the pharmaceutical alert manager."""
        try:
            # Create alert in the monitoring system
            alert_data = {
                "alert_type": "drug_safety",
                "severity": self._map_urgency_to_severity(alert.urgency),
                "message": alert.safety_message,
                "drug_names": alert.drug_names,
                "patient_populations": alert.patient_populations,
                "recommendations": alert.recommendations,
                "pharmaceutical_context": {
                    "domain": alert.pharmaceutical_context.domain.value,
                    "safety_urgency": alert.pharmaceutical_context.safety_urgency.name,
                    "research_priority": alert.pharmaceutical_context.research_priority.name,
                },
            }

            # Route through alert manager
            await self.alert_manager.create_alert(
                alert_type="pharmaceutical_safety", message=alert.safety_message, metadata=alert_data
            )

        except Exception as e:
            logger.error(f"Failed to route safety alert {alert.alert_id}: {str(e)}")

    def _map_urgency_to_severity(self, urgency: AlertUrgency) -> str:
        """Map alert urgency to alert manager severity."""
        mapping = {
            AlertUrgency.IMMEDIATE: "critical",
            AlertUrgency.URGENT: "high",
            AlertUrgency.ROUTINE: "medium",
            AlertUrgency.INFORMATIONAL: "low",
        }
        return mapping.get(urgency, "medium")

    async def _execute_alert_callbacks(self, alert: DrugSafetyAlert) -> None:
        """Execute registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback execution failed: {str(e)}")

    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback function for safety alerts."""
        self.alert_callbacks.append(callback)
        logger.info("Safety alert callback registered")

    def _generate_alert_id(self, alert_type: str, context: str) -> str:
        """Generate unique alert ID."""
        timestamp = int(time.time())
        content = f"{alert_type}_{context}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_active_safety_alerts(self) -> List[Dict[str, Any]]:
        """Get all active drug safety alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "urgency": alert.urgency.value,
                "safety_message": alert.safety_message,
                "drug_names": alert.drug_names,
                "patient_populations": alert.patient_populations,
                "timestamp": alert.timestamp.isoformat(),
                "acknowledged": alert.acknowledged,
            }
            for alert in self.active_safety_alerts.values()
            if not alert.acknowledged
        ]

    def acknowledge_safety_alert(self, alert_id: str) -> bool:
        """Acknowledge a safety alert."""
        if alert_id in self.active_safety_alerts:
            alert = self.active_safety_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledgment_timestamp = datetime.now()
            logger.info(f"Safety alert {alert_id} acknowledged")
            return True
        return False

    async def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive drug safety dashboard."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)

        recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= last_24h]

        return {
            "dashboard_timestamp": now.isoformat(),
            "active_alerts": len([a for a in self.active_safety_alerts.values() if not a.acknowledged]),
            "alerts_last_24h": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.urgency == AlertUrgency.IMMEDIATE]),
            "alert_breakdown": {
                "critical_safety": len([a for a in recent_alerts if a.alert_type == SafetyAlertType.CRITICAL_SAFETY]),
                "drug_interactions": len(
                    [a for a in recent_alerts if a.alert_type == SafetyAlertType.DRUG_INTERACTION]
                ),
                "adverse_reactions": len(
                    [a for a in recent_alerts if a.alert_type == SafetyAlertType.ADVERSE_REACTION]
                ),
                "contraindications": len(
                    [a for a in recent_alerts if a.alert_type == SafetyAlertType.CONTRAINDICATION]
                ),
            },
            "top_drugs_alerted": self._get_top_drugs_with_alerts(recent_alerts),
            "safety_monitoring_active": True,
            "pharmaceutical_optimization": True,
        }

    def _get_top_drugs_with_alerts(self, alerts: List[DrugSafetyAlert]) -> List[Dict[str, Any]]:
        """Get top drugs with safety alerts."""
        drug_counts = {}
        for alert in alerts:
            for drug in alert.drug_names:
                drug_counts[drug] = drug_counts.get(drug, 0) + 1

        return [
            {"drug_name": drug, "alert_count": count}
            for drug, count in sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]


# Convenience functions for drug safety integration
async def process_pharmaceutical_query_with_safety(
    query_text: str, user_context: Optional[Dict[str, Any]] = None
) -> Tuple[PharmaceuticalContext, List[DrugSafetyAlert]]:
    """
    Process pharmaceutical query with integrated safety monitoring.

    Args:
        query_text: Pharmaceutical research query
        user_context: Additional user context

    Returns:
        Pharmaceutical context and generated safety alerts
    """
    safety_integration = DrugSafetyAlertIntegration()
    return await safety_integration.process_pharmaceutical_query(query_text, user_context)


def create_drug_safety_monitor() -> DrugSafetyAlertIntegration:
    """
    Create drug safety monitoring system with optimal configuration.

    Returns:
        Configured drug safety alert integration
    """
    config = EnhancedRAGConfig.from_env()
    alert_manager = PharmaceuticalAlertManager()

    return DrugSafetyAlertIntegration(config=config, alert_manager=alert_manager)


if __name__ == "__main__":
    # Test drug safety alert integration
    async def main():
        safety_monitor = create_drug_safety_monitor()

        test_queries = [
            "Urgent: Patient on warfarin experiencing bleeding after taking aspirin - drug interaction concern",
            "Metformin contraindications in elderly patient with severe kidney disease",
            "ACE inhibitor mechanism of action in cardiovascular treatment",
            "Adverse reactions to new oncology drug in phase III clinical trial",
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Processing: {query[:50]}...")

            context, alerts = await safety_monitor.process_pharmaceutical_query(query)

            print(f"Domain: {context.domain.value}")
            print(f"Safety Urgency: {context.safety_urgency.name}")
            print(f"Generated Alerts: {len(alerts)}")

            for alert in alerts:
                print(f"  - {alert.alert_type.value} ({alert.urgency.value})")
                print(f"    {alert.safety_message}")

        # Get safety dashboard
        dashboard = await safety_monitor.get_safety_dashboard()
        print(f"\n{'='*60}")
        print("Safety Dashboard:")
        print(json.dumps(dashboard, indent=2))

    asyncio.run(main())
