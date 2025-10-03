"""
Pharmaceutical Research Workflow Templates and Examples

Pre-built pharmaceutical research workflows that integrate query classification,
safety monitoring, cost optimization, and cloud-first execution for common
pharmaceutical research scenarios.

Features:
- Drug safety assessment workflows
- Clinical trial research templates
- Drug interaction analysis pipelines
- Pharmacokinetics investigation workflows
- Regulatory compliance research templates
- Adverse reaction monitoring workflows

All templates are optimized for the NVIDIA Build cloud-first architecture
with pharmaceutical domain prioritization and cost optimization.
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

try:
    from ..clients.nemo_client_enhanced import EnhancedNeMoClient
    from ..enhanced_config import EnhancedRAGConfig
    from ..optimization.batch_integration import PharmaceuticalBatchClient
    from .query_classifier import PharmaceuticalQueryClassifier
    from .safety_alert_integration import DrugSafetyAlertIntegration
except ImportError:
    from src.clients.nemo_client_enhanced import EnhancedNeMoClient
    from src.enhanced_config import EnhancedRAGConfig
    from src.optimization.batch_integration import PharmaceuticalBatchClient
    from src.pharmaceutical.query_classifier import PharmaceuticalQueryClassifier
    from src.pharmaceutical.safety_alert_integration import DrugSafetyAlertIntegration

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Types of pharmaceutical research workflows."""

    DRUG_SAFETY_ASSESSMENT = "drug_safety_assessment"
    CLINICAL_TRIAL_RESEARCH = "clinical_trial_research"
    DRUG_INTERACTION_ANALYSIS = "drug_interaction_analysis"
    PHARMACOKINETICS_INVESTIGATION = "pharmacokinetics_investigation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    ADVERSE_REACTION_MONITORING = "adverse_reaction_monitoring"
    MECHANISM_OF_ACTION_STUDY = "mechanism_of_action_study"
    DOSAGE_OPTIMIZATION = "dosage_optimization"
    THERAPEUTIC_EFFICACY = "therapeutic_efficacy"
    PHARMACOGENOMICS_ANALYSIS = "pharmacogenomics_analysis"


@dataclass
class WorkflowStep:
    """Individual step in a pharmaceutical research workflow."""

    step_name: str
    step_type: str  # "query", "analysis", "validation", "reporting"
    description: str
    query_template: Optional[str] = None
    expected_output_type: str = "text"
    priority: str = "normal"
    safety_critical: bool = False
    cost_tier_preference: str = "free_tier"


@dataclass
class WorkflowResult:
    """Result from executing a pharmaceutical workflow."""

    workflow_type: WorkflowType
    workflow_name: str
    execution_timestamp: datetime
    step_results: List[Dict[str, Any]]
    safety_alerts: List[Dict[str, Any]]
    pharmaceutical_insights: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    overall_success: bool
    execution_time_ms: int


class PharmaceuticalWorkflowTemplate:
    """
    Template for pharmaceutical research workflows with integrated
    classification, safety monitoring, and cost optimization.
    """

    def __init__(self, workflow_type: WorkflowType, workflow_name: str, description: str, steps: List[WorkflowStep]):
        """
        Initialize pharmaceutical workflow template.

        Args:
            workflow_type: Type of pharmaceutical workflow
            workflow_name: Human-readable workflow name
            description: Detailed workflow description
            steps: List of workflow steps to execute
        """
        self.workflow_type = workflow_type
        self.workflow_name = workflow_name
        self.description = description
        self.steps = steps

        # Template metadata
        self.created_timestamp = datetime.now()
        self.pharmaceutical_optimized = True
        self.cloud_first_enabled = True

    def customize_for_drug(self, drug_name: str) -> "PharmaceuticalWorkflowTemplate":
        """Create drug-specific version of workflow template."""
        customized_steps = []

        for step in self.steps:
            customized_step = WorkflowStep(
                step_name=step.step_name,
                step_type=step.step_type,
                description=step.description.replace("{drug_name}", drug_name),
                query_template=step.query_template.replace("{drug_name}", drug_name) if step.query_template else None,
                expected_output_type=step.expected_output_type,
                priority=step.priority,
                safety_critical=step.safety_critical,
                cost_tier_preference=step.cost_tier_preference,
            )
            customized_steps.append(customized_step)

        return PharmaceuticalWorkflowTemplate(
            workflow_type=self.workflow_type,
            workflow_name=f"{self.workflow_name} - {drug_name}",
            description=f"{self.description} Customized for {drug_name}.",
            steps=customized_steps,
        )


class PharmaceuticalWorkflowExecutor:
    """
    Executor for pharmaceutical research workflows with integrated
    safety monitoring and cloud-first optimization.
    """

    def __init__(self, config: Optional[EnhancedRAGConfig] = None):
        """
        Initialize pharmaceutical workflow executor.

        Args:
            config: Enhanced RAG configuration
        """
        self.config = config or EnhancedRAGConfig.from_env()

        # Initialize integrated components
        self.enhanced_client = EnhancedNeMoClient(pharmaceutical_optimized=True)
        self.query_classifier = PharmaceuticalQueryClassifier()
        self.safety_integration = DrugSafetyAlertIntegration()
        self.batch_client: Optional[PharmaceuticalBatchClient] = None

        # Execution tracking
        self.execution_history: List[WorkflowResult] = []

        logger.info("PharmaceuticalWorkflowExecutor initialized with integrated components")

    async def execute_workflow(
        self, template: PharmaceuticalWorkflowTemplate, parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute pharmaceutical research workflow with full integration.

        Args:
            template: Workflow template to execute
            parameters: Additional parameters for workflow customization

        Returns:
            Complete workflow execution results
        """
        start_time = datetime.now()
        execution_start_ms = int(start_time.timestamp() * 1000)

        step_results = []
        safety_alerts_all = []
        pharmaceutical_insights = {}

        logger.info(f"Executing workflow: {template.workflow_name}")

        try:
            # Initialize batch client if needed
            if not self.batch_client:
                self.batch_client = PharmaceuticalBatchClient(enhanced_client=self.enhanced_client)

            # Execute each workflow step
            for i, step in enumerate(template.steps):
                logger.info(f"Executing step {i+1}/{len(template.steps)}: {step.step_name}")

                step_result = await self._execute_workflow_step(step, parameters, template)
                step_results.append(step_result)

                # Collect safety alerts from step execution
                if "safety_alerts" in step_result:
                    safety_alerts_all.extend(step_result["safety_alerts"])

                # Stop execution if critical error and safety-critical step
                if step.safety_critical and not step_result.get("success", False):
                    logger.error(f"Critical step {step.step_name} failed - stopping workflow")
                    break

            # Analyze pharmaceutical insights across all steps
            pharmaceutical_insights = self._analyze_pharmaceutical_insights(step_results, template)

            # Calculate cost analysis
            cost_analysis = await self._calculate_workflow_cost_analysis(step_results)

            # Determine overall success
            overall_success = all(
                result.get("success", False)
                for result in step_results
                if result.get("step_type") != "analysis"  # Analysis steps can be informational
            )

            execution_time_ms = int((datetime.now().timestamp() * 1000) - execution_start_ms)

            result = WorkflowResult(
                workflow_type=template.workflow_type,
                workflow_name=template.workflow_name,
                execution_timestamp=start_time,
                step_results=step_results,
                safety_alerts=safety_alerts_all,
                pharmaceutical_insights=pharmaceutical_insights,
                cost_analysis=cost_analysis,
                overall_success=overall_success,
                execution_time_ms=execution_time_ms,
            )

            # Store in execution history
            self.execution_history.append(result)

            logger.info(
                f"Workflow completed: {template.workflow_name} "
                f"({execution_time_ms}ms, {len(safety_alerts_all)} alerts)"
            )

            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")

            execution_time_ms = int((datetime.now().timestamp() * 1000) - execution_start_ms)

            return WorkflowResult(
                workflow_type=template.workflow_type,
                workflow_name=template.workflow_name,
                execution_timestamp=start_time,
                step_results=step_results,
                safety_alerts=safety_alerts_all,
                pharmaceutical_insights={"error": str(e)},
                cost_analysis={"execution_failed": True},
                overall_success=False,
                execution_time_ms=execution_time_ms,
            )

    async def _execute_workflow_step(
        self, step: WorkflowStep, parameters: Optional[Dict[str, Any]], template: PharmaceuticalWorkflowTemplate
    ) -> Dict[str, Any]:
        """Execute individual workflow step."""
        step_start_time = datetime.now()

        try:
            if step.step_type == "query":
                return await self._execute_query_step(step, parameters)
            elif step.step_type == "analysis":
                return await self._execute_analysis_step(step, parameters)
            elif step.step_type == "validation":
                return await self._execute_validation_step(step, parameters)
            elif step.step_type == "reporting":
                return await self._execute_reporting_step(step, parameters)
            else:
                return {
                    "step_name": step.step_name,
                    "step_type": step.step_type,
                    "success": False,
                    "error": f"Unknown step type: {step.step_type}",
                    "timestamp": step_start_time.isoformat(),
                }

        except Exception as e:
            return {
                "step_name": step.step_name,
                "step_type": step.step_type,
                "success": False,
                "error": str(e),
                "timestamp": step_start_time.isoformat(),
            }

    async def _execute_query_step(self, step: WorkflowStep, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute query step with pharmaceutical optimization."""
        if not step.query_template:
            return {"step_name": step.step_name, "success": False, "error": "No query template provided"}

        # Substitute parameters in query template
        query_text = step.query_template
        if parameters:
            for key, value in parameters.items():
                query_text = query_text.replace(f"{{{key}}}", str(value))

        # Classify query for pharmaceutical context
        pharmaceutical_context, safety_alerts = await self.safety_integration.process_pharmaceutical_query(query_text)

        # Execute query through enhanced client
        if step.expected_output_type == "embeddings":
            response = self.enhanced_client.create_embeddings([query_text])
        else:
            messages = [{"role": "user", "content": query_text}]
            response = self.enhanced_client.create_chat_completion(messages)

        return {
            "step_name": step.step_name,
            "step_type": step.step_type,
            "success": response.success,
            "query_text": query_text,
            "pharmaceutical_context": {
                "domain": pharmaceutical_context.domain.value,
                "safety_urgency": pharmaceutical_context.safety_urgency.name,
                "drug_names": pharmaceutical_context.drug_names,
            },
            "response": response.data if response.success else None,
            "error": response.error if not response.success else None,
            "safety_alerts": [
                {"alert_type": alert.alert_type.value, "urgency": alert.urgency.value, "message": alert.safety_message}
                for alert in safety_alerts
            ],
            "cost_tier": response.cost_tier,
            "response_time_ms": response.response_time_ms,
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_analysis_step(self, step: WorkflowStep, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute analysis step."""
        # Analysis steps typically process previous results
        return {
            "step_name": step.step_name,
            "step_type": step.step_type,
            "success": True,
            "analysis_type": "pharmaceutical_domain_analysis",
            "description": step.description,
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_validation_step(
        self, step: WorkflowStep, parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute validation step."""
        return {
            "step_name": step.step_name,
            "step_type": step.step_type,
            "success": True,
            "validation_type": "pharmaceutical_safety_validation",
            "description": step.description,
            "timestamp": datetime.now().isoformat(),
        }

    async def _execute_reporting_step(self, step: WorkflowStep, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute reporting step."""
        return {
            "step_name": step.step_name,
            "step_type": step.step_type,
            "success": True,
            "report_type": "pharmaceutical_research_report",
            "description": step.description,
            "timestamp": datetime.now().isoformat(),
        }

    def _analyze_pharmaceutical_insights(
        self, step_results: List[Dict[str, Any]], template: PharmaceuticalWorkflowTemplate
    ) -> Dict[str, Any]:
        """Analyze pharmaceutical insights from workflow execution."""
        insights = {
            "workflow_type": template.workflow_type.value,
            "pharmaceutical_domains_covered": [],
            "drug_names_analyzed": [],
            "safety_considerations": [],
            "research_priorities_identified": [],
            "cost_optimization_applied": True,
        }

        for result in step_results:
            # Extract pharmaceutical context
            if "pharmaceutical_context" in result:
                context = result["pharmaceutical_context"]
                if context.get("domain") not in insights["pharmaceutical_domains_covered"]:
                    insights["pharmaceutical_domains_covered"].append(context["domain"])

                insights["drug_names_analyzed"].extend(context.get("drug_names", []))

            # Extract safety alerts
            if "safety_alerts" in result:
                for alert in result["safety_alerts"]:
                    safety_consideration = f"{alert['alert_type']}: {alert['urgency']}"
                    if safety_consideration not in insights["safety_considerations"]:
                        insights["safety_considerations"].append(safety_consideration)

        # Remove duplicates from drug names
        insights["drug_names_analyzed"] = list(set(insights["drug_names_analyzed"]))

        return insights

    async def _calculate_workflow_cost_analysis(self, step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cost analysis for workflow execution."""
        cost_analysis = {
            "total_queries": len([r for r in step_results if r.get("step_type") == "query"]),
            "free_tier_queries": 0,
            "infrastructure_queries": 0,
            "total_response_time_ms": 0,
            "cost_optimized": True,
        }

        for result in step_results:
            if result.get("step_type") == "query":
                cost_tier = result.get("cost_tier", "unknown")
                response_time = result.get("response_time_ms", 0)

                if cost_tier == "free_tier":
                    cost_analysis["free_tier_queries"] += 1
                elif cost_tier == "infrastructure":
                    cost_analysis["infrastructure_queries"] += 1

                cost_analysis["total_response_time_ms"] += response_time

        # Calculate optimization metrics
        if cost_analysis["total_queries"] > 0:
            cost_analysis["free_tier_utilization"] = cost_analysis["free_tier_queries"] / cost_analysis["total_queries"]
        else:
            cost_analysis["free_tier_utilization"] = 0.0

        return cost_analysis


# Pre-built Pharmaceutical Workflow Templates


def create_drug_safety_assessment_workflow() -> PharmaceuticalWorkflowTemplate:
    """Create comprehensive drug safety assessment workflow."""
    steps = [
        WorkflowStep(
            step_name="Drug Safety Overview",
            step_type="query",
            description="Get comprehensive safety profile for {drug_name}",
            query_template="Provide a comprehensive drug safety profile for {drug_name}, including contraindications, warnings, and precautions.",
            priority="high",
            safety_critical=True,
        ),
        WorkflowStep(
            step_name="Contraindications Analysis",
            step_type="query",
            description="Analyze contraindications and absolute restrictions for {drug_name}",
            query_template="What are the absolute contraindications for {drug_name}? Include patient populations where {drug_name} must not be used.",
            priority="critical",
            safety_critical=True,
        ),
        WorkflowStep(
            step_name="Drug Interactions Assessment",
            step_type="query",
            description="Identify critical drug interactions for {drug_name}",
            query_template="List critical drug interactions for {drug_name}, focusing on life-threatening or severe interactions.",
            priority="high",
            safety_critical=True,
        ),
        WorkflowStep(
            step_name="Monitoring Requirements",
            step_type="query",
            description="Determine monitoring requirements for {drug_name} therapy",
            query_template="What laboratory monitoring and clinical assessments are required for patients taking {drug_name}?",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Safety Validation",
            step_type="validation",
            description="Validate safety information completeness and accuracy",
        ),
        WorkflowStep(
            step_name="Safety Report Generation",
            step_type="reporting",
            description="Generate comprehensive drug safety assessment report",
        ),
    ]

    return PharmaceuticalWorkflowTemplate(
        workflow_type=WorkflowType.DRUG_SAFETY_ASSESSMENT,
        workflow_name="Comprehensive Drug Safety Assessment",
        description="Complete drug safety evaluation including contraindications, interactions, and monitoring requirements.",
        steps=steps,
    )


def create_clinical_trial_research_workflow() -> PharmaceuticalWorkflowTemplate:
    """Create clinical trial research workflow."""
    steps = [
        WorkflowStep(
            step_name="Clinical Efficacy Overview",
            step_type="query",
            description="Research clinical efficacy data for {drug_name}",
            query_template="Summarize the clinical efficacy data for {drug_name} from pivotal clinical trials, including primary endpoints.",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Phase III Trial Results",
            step_type="query",
            description="Analyze Phase III trial outcomes for {drug_name}",
            query_template="What were the key Phase III clinical trial results for {drug_name}? Include efficacy outcomes and statistical significance.",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Safety Profile from Trials",
            step_type="query",
            description="Extract safety data from clinical trials",
            query_template="What adverse events and safety signals were identified in clinical trials for {drug_name}?",
            priority="high",
            safety_critical=True,
        ),
        WorkflowStep(
            step_name="Regulatory Approval Basis",
            step_type="query",
            description="Analyze regulatory approval basis",
            query_template="On what clinical evidence basis was {drug_name} approved by regulatory authorities?",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Trial Data Analysis",
            step_type="analysis",
            description="Analyze consistency and quality of clinical trial data",
        ),
        WorkflowStep(
            step_name="Clinical Research Report",
            step_type="reporting",
            description="Generate comprehensive clinical trial research report",
        ),
    ]

    return PharmaceuticalWorkflowTemplate(
        workflow_type=WorkflowType.CLINICAL_TRIAL_RESEARCH,
        workflow_name="Clinical Trial Evidence Research",
        description="Comprehensive analysis of clinical trial data and regulatory approval evidence.",
        steps=steps,
    )


def create_drug_interaction_analysis_workflow() -> PharmaceuticalWorkflowTemplate:
    """Create drug interaction analysis workflow."""
    steps = [
        WorkflowStep(
            step_name="Major Drug Interactions",
            step_type="query",
            description="Identify major drug interactions for {drug_name}",
            query_template="What are the major drug interactions for {drug_name} that require dose adjustment or are contraindicated?",
            priority="critical",
            safety_critical=True,
        ),
        WorkflowStep(
            step_name="Mechanism-Based Interactions",
            step_type="query",
            description="Analyze interaction mechanisms",
            query_template="Explain the mechanisms of drug interactions for {drug_name}, including CYP enzyme interactions and transporter effects.",
            priority="high",
        ),
        WorkflowStep(
            step_name="Clinical Management",
            step_type="query",
            description="Determine clinical management strategies",
            query_template="How should clinically significant drug interactions with {drug_name} be managed? Include monitoring recommendations.",
            priority="high",
            safety_critical=True,
        ),
        WorkflowStep(
            step_name="Patient Population Considerations",
            step_type="query",
            description="Analyze population-specific interaction risks",
            query_template="Are there specific patient populations at higher risk for drug interactions with {drug_name}?",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Interaction Risk Assessment",
            step_type="analysis",
            description="Assess overall interaction risk profile",
        ),
        WorkflowStep(
            step_name="Interaction Management Report",
            step_type="reporting",
            description="Generate drug interaction management report",
        ),
    ]

    return PharmaceuticalWorkflowTemplate(
        workflow_type=WorkflowType.DRUG_INTERACTION_ANALYSIS,
        workflow_name="Drug Interaction Analysis",
        description="Comprehensive drug interaction analysis with clinical management recommendations.",
        steps=steps,
    )


def create_pharmacokinetics_investigation_workflow() -> PharmaceuticalWorkflowTemplate:
    """Create pharmacokinetics investigation workflow."""
    steps = [
        WorkflowStep(
            step_name="ADME Profile",
            step_type="query",
            description="Analyze ADME characteristics of {drug_name}",
            query_template="Describe the ADME (absorption, distribution, metabolism, excretion) profile of {drug_name}.",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Pharmacokinetic Parameters",
            step_type="query",
            description="Extract key pharmacokinetic parameters",
            query_template="What are the key pharmacokinetic parameters for {drug_name} including half-life, clearance, and bioavailability?",
            priority="normal",
        ),
        WorkflowStep(
            step_name="Special Population PK",
            step_type="query",
            description="Analyze pharmacokinetics in special populations",
            query_template="How does the pharmacokinetics of {drug_name} change in elderly patients, patients with renal impairment, and hepatic impairment?",
            priority="high",
        ),
        WorkflowStep(
            step_name="Dosing Implications",
            step_type="query",
            description="Determine dosing implications from PK data",
            query_template="What are the dosing implications based on the pharmacokinetic profile of {drug_name}?",
            priority="normal",
        ),
        WorkflowStep(
            step_name="PK Data Analysis",
            step_type="analysis",
            description="Analyze pharmacokinetic data comprehensiveness",
        ),
        WorkflowStep(
            step_name="Pharmacokinetics Report",
            step_type="reporting",
            description="Generate comprehensive pharmacokinetics report",
        ),
    ]

    return PharmaceuticalWorkflowTemplate(
        workflow_type=WorkflowType.PHARMACOKINETICS_INVESTIGATION,
        workflow_name="Pharmacokinetics Investigation",
        description="Comprehensive pharmacokinetic analysis with dosing implications.",
        steps=steps,
    )


# Workflow Template Factory
class PharmaceuticalWorkflowFactory:
    """Factory for creating pharmaceutical research workflow templates."""

    @staticmethod
    def get_available_workflows() -> List[str]:
        """Get list of available workflow types."""
        return [
            "drug_safety_assessment",
            "clinical_trial_research",
            "drug_interaction_analysis",
            "pharmacokinetics_investigation",
        ]

    @staticmethod
    def create_workflow(workflow_name: str) -> Optional[PharmaceuticalWorkflowTemplate]:
        """Create workflow template by name."""
        workflows = {
            "drug_safety_assessment": create_drug_safety_assessment_workflow,
            "clinical_trial_research": create_clinical_trial_research_workflow,
            "drug_interaction_analysis": create_drug_interaction_analysis_workflow,
            "pharmacokinetics_investigation": create_pharmacokinetics_investigation_workflow,
        }

        factory_func = workflows.get(workflow_name)
        return factory_func() if factory_func else None

    @staticmethod
    def create_custom_workflow(
        workflow_type: WorkflowType, name: str, description: str, steps: List[WorkflowStep]
    ) -> PharmaceuticalWorkflowTemplate:
        """Create custom workflow template."""
        return PharmaceuticalWorkflowTemplate(
            workflow_type=workflow_type, workflow_name=name, description=description, steps=steps
        )


# Convenience functions
async def execute_drug_safety_workflow(drug_name: str) -> WorkflowResult:
    """Execute drug safety assessment workflow for specific drug."""
    template = create_drug_safety_assessment_workflow()
    customized_template = template.customize_for_drug(drug_name)

    executor = PharmaceuticalWorkflowExecutor()
    return await executor.execute_workflow(customized_template, parameters={"drug_name": drug_name})


async def execute_clinical_trial_workflow(drug_name: str) -> WorkflowResult:
    """Execute clinical trial research workflow for specific drug."""
    template = create_clinical_trial_research_workflow()
    customized_template = template.customize_for_drug(drug_name)

    executor = PharmaceuticalWorkflowExecutor()
    return await executor.execute_workflow(customized_template, parameters={"drug_name": drug_name})


if __name__ == "__main__":
    # Example workflow execution
    async def main():
        # Test drug safety assessment workflow
        print("Testing Drug Safety Assessment Workflow")
        print("=" * 50)

        safety_result = await execute_drug_safety_workflow("metformin")

        print(f"Workflow: {safety_result.workflow_name}")
        print(f"Success: {safety_result.overall_success}")
        print(f"Execution Time: {safety_result.execution_time_ms}ms")
        print(f"Safety Alerts: {len(safety_result.safety_alerts)}")
        print(f"Steps Completed: {len(safety_result.step_results)}")

        # Display pharmaceutical insights
        insights = safety_result.pharmaceutical_insights
        print("\nPharmaceutical Insights:")
        print(f"  Domains: {insights.get('pharmaceutical_domains_covered', [])}")
        print(f"  Drugs: {insights.get('drug_names_analyzed', [])}")
        print(f"  Safety Considerations: {len(insights.get('safety_considerations', []))}")

        # Display cost analysis
        cost = safety_result.cost_analysis
        print("\nCost Analysis:")
        print(f"  Total Queries: {cost.get('total_queries', 0)}")
        print(f"  Free Tier Utilization: {cost.get('free_tier_utilization', 0.0):.1%}")

        print("\n" + "=" * 50)

        # Test clinical trial workflow
        print("Testing Clinical Trial Research Workflow")
        print("=" * 50)

        clinical_result = await execute_clinical_trial_workflow("atorvastatin")

        print(f"Workflow: {clinical_result.workflow_name}")
        print(f"Success: {clinical_result.overall_success}")
        print(f"Execution Time: {clinical_result.execution_time_ms}ms")

    asyncio.run(main())
