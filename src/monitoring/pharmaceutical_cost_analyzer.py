"""
Pharmaceutical Cost Analyzer for NVIDIA Build Free Tier Optimization

Provides sophisticated cost analysis and budgeting specifically for pharmaceutical
research workflows, with intelligent categorization of query types and ROI tracking.

Features:
- Cost-per-query analysis by pharmaceutical domain
- Research project budgeting and tracking
- ROI calculation for pharmaceutical insights
- Free tier optimization with domain-specific strategies
- Predictive cost modeling for research planning

Integration:
- Extends existing credit tracking with pharmaceutical analytics
- Integrates with batch processing for cost optimization
- Provides actionable insights for research budget management
"""

import logging
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict

try:
    from .credit_tracker import PharmaceuticalCreditTracker
    from ..enhanced_config import EnhancedRAGConfig
except ImportError:
    from src.monitoring.credit_tracker import PharmaceuticalCreditTracker
    from src.enhanced_config import EnhancedRAGConfig

logger = logging.getLogger(__name__)

class PharmaceuticalQueryType(Enum):
    """Pharmaceutical research query categories for cost analysis."""
    DRUG_SAFETY = "drug_safety"  # Highest value - safety critical
    CLINICAL_TRIALS = "clinical_trials"  # High value - research critical
    DRUG_INTERACTIONS = "drug_interactions"  # High value - safety critical
    MECHANISM_OF_ACTION = "mechanism_of_action"  # Medium value - research
    PHARMACOKINETICS = "pharmacokinetics"  # Medium value - research
    DOSAGE_GUIDELINES = "dosage_guidelines"  # Medium value - clinical
    GENERAL_RESEARCH = "general_research"  # Standard value
    EXPLORATORY = "exploratory"  # Lower value - discovery

@dataclass
class PharmaceuticalQueryRecord:
    """Record of a pharmaceutical research query with cost analysis."""
    query_id: str
    query_text: str
    query_type: PharmaceuticalQueryType
    timestamp: datetime
    cost_tier: str  # "free_tier" or "infrastructure"
    estimated_tokens: int
    estimated_cost_usd: float
    response_quality_score: Optional[float] = None
    research_value_score: Optional[float] = None
    project_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

class ResearchProjectBudget:
    """Budget tracking for pharmaceutical research projects."""

    def __init__(self,
                 project_id: str,
                 project_name: str,
                 monthly_budget_usd: float,
                 priority_level: int = 3):
        self.project_id = project_id
        self.project_name = project_name
        self.monthly_budget_usd = monthly_budget_usd
        self.priority_level = priority_level  # 1=critical, 5=low

        self.spent_this_month = 0.0
        self.queries_this_month = 0
        self.high_value_queries = 0
        self.created_date = datetime.now()
        self.last_activity = datetime.now()

    def add_query_cost(self, cost: float, query_type: PharmaceuticalQueryType) -> None:
        """Add query cost to project budget tracking."""
        self.spent_this_month += cost
        self.queries_this_month += 1
        self.last_activity = datetime.now()

        # Track high-value queries
        high_value_types = {
            PharmaceuticalQueryType.DRUG_SAFETY,
            PharmaceuticalQueryType.CLINICAL_TRIALS,
            PharmaceuticalQueryType.DRUG_INTERACTIONS
        }
        if query_type in high_value_types:
            self.high_value_queries += 1

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and projections."""
        utilization = (self.spent_this_month / self.monthly_budget_usd) if self.monthly_budget_usd > 0 else 0
        remaining_budget = max(0, self.monthly_budget_usd - self.spent_this_month)

        # Calculate average cost per query
        avg_cost = (self.spent_this_month / self.queries_this_month) if self.queries_this_month > 0 else 0

        # Estimate remaining queries possible
        remaining_queries = int(remaining_budget / avg_cost) if avg_cost > 0 else 0

        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "monthly_budget_usd": self.monthly_budget_usd,
            "spent_this_month": round(self.spent_this_month, 4),
            "remaining_budget": round(remaining_budget, 4),
            "budget_utilization": round(utilization, 3),
            "queries_this_month": self.queries_this_month,
            "high_value_queries": self.high_value_queries,
            "avg_cost_per_query": round(avg_cost, 4),
            "estimated_remaining_queries": remaining_queries,
            "priority_level": self.priority_level,
            "last_activity": self.last_activity.isoformat()
        }

class PharmaceuticalCostAnalyzer:
    """
    Advanced cost analyzer for pharmaceutical research workflows.

    Provides detailed cost breakdown, ROI analysis, and optimization
    recommendations specifically for pharmaceutical research applications.
    """

    def __init__(self,
                 credit_tracker: Optional[PharmaceuticalCreditTracker] = None,
                 config: Optional[EnhancedRAGConfig] = None):
        """
        Initialize pharmaceutical cost analyzer.

        Args:
            credit_tracker: Pharmaceutical credit tracker for integration
            config: Enhanced RAG configuration
        """
        self.credit_tracker = credit_tracker or PharmaceuticalCreditTracker()
        self.config = config or EnhancedRAGConfig.from_env()

        # Query records and analysis
        self.query_records: List[PharmaceuticalQueryRecord] = []
        self.projects: Dict[str, ResearchProjectBudget] = {}

        # Cost analysis cache
        self._analysis_cache = {}
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minute cache

        # Pharmaceutical domain cost mappings
        self.query_type_values = {
            PharmaceuticalQueryType.DRUG_SAFETY: 5.0,  # Highest value
            PharmaceuticalQueryType.CLINICAL_TRIALS: 4.5,
            PharmaceuticalQueryType.DRUG_INTERACTIONS: 4.5,
            PharmaceuticalQueryType.MECHANISM_OF_ACTION: 3.5,
            PharmaceuticalQueryType.PHARMACOKINETICS: 3.5,
            PharmaceuticalQueryType.DOSAGE_GUIDELINES: 3.0,
            PharmaceuticalQueryType.GENERAL_RESEARCH: 2.0,
            PharmaceuticalQueryType.EXPLORATORY: 1.0
        }

        logger.info("PharmaceuticalCostAnalyzer initialized with domain-specific optimization")

    def record_pharmaceutical_query(self,
                                  query_id: str,
                                  query_text: str,
                                  cost_tier: str,
                                  estimated_tokens: int,
                                  project_id: Optional[str] = None,
                                  tags: Optional[List[str]] = None) -> PharmaceuticalQueryRecord:
        """
        Record a pharmaceutical query for cost analysis.

        Args:
            query_id: Unique query identifier
            query_text: The research query text
            cost_tier: "free_tier" or "infrastructure"
            estimated_tokens: Estimated token usage
            project_id: Research project identifier
            tags: Additional tags for categorization

        Returns:
            PharmaceuticalQueryRecord with analysis
        """
        # Classify query type based on content
        query_type = self._classify_pharmaceutical_query(query_text)

        # Calculate estimated cost
        estimated_cost = self._calculate_query_cost(estimated_tokens, cost_tier, query_type)

        # Create query record
        record = PharmaceuticalQueryRecord(
            query_id=query_id,
            query_text=query_text,
            query_type=query_type,
            timestamp=datetime.now(),
            cost_tier=cost_tier,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            project_id=project_id,
            tags=tags or []
        )

        # Calculate research value score
        record.research_value_score = self._calculate_research_value_score(record)

        # Add to records
        self.query_records.append(record)

        # Update project budget if applicable
        if project_id and project_id in self.projects:
            self.projects[project_id].add_query_cost(estimated_cost, query_type)

        # Invalidate cache
        self._invalidate_cache()

        logger.debug(f"Recorded pharmaceutical query {query_id}: {query_type.value} "
                    f"(${estimated_cost:.4f}, {cost_tier})")

        return record

    def _classify_pharmaceutical_query(self, query_text: str) -> PharmaceuticalQueryType:
        """Classify pharmaceutical query based on content analysis."""
        query_lower = query_text.lower()

        # Drug safety indicators (highest priority)
        safety_keywords = [
            "adverse", "side effect", "toxicity", "contraindication", "warning",
            "danger", "risk", "safety", "overdose", "poisoning", "fatal"
        ]
        if any(keyword in query_lower for keyword in safety_keywords):
            return PharmaceuticalQueryType.DRUG_SAFETY

        # Drug interaction indicators
        interaction_keywords = [
            "interaction", "combined", "together", "concurrent", "simultaneous",
            "contraindicated", "incompatible"
        ]
        if any(keyword in query_lower for keyword in interaction_keywords):
            return PharmaceuticalQueryType.DRUG_INTERACTIONS

        # Clinical trials indicators
        trial_keywords = [
            "clinical trial", "study", "efficacy", "effectiveness", "trial data",
            "clinical evidence", "phase", "randomized", "placebo"
        ]
        if any(keyword in query_lower for keyword in trial_keywords):
            return PharmaceuticalQueryType.CLINICAL_TRIALS

        # Pharmacokinetics indicators
        pk_keywords = [
            "pharmacokinetics", "absorption", "metabolism", "distribution",
            "excretion", "clearance", "half-life", "bioavailability"
        ]
        if any(keyword in query_lower for keyword in pk_keywords):
            return PharmaceuticalQueryType.PHARMACOKINETICS

        # Mechanism of action indicators
        mechanism_keywords = [
            "mechanism", "action", "pathway", "target", "receptor", "enzyme",
            "how does", "how works", "mode of action"
        ]
        if any(keyword in query_lower for keyword in mechanism_keywords):
            return PharmaceuticalQueryType.MECHANISM_OF_ACTION

        # Dosage guidelines
        dosage_keywords = [
            "dose", "dosage", "dosing", "administration", "frequency",
            "how much", "how often", "regimen"
        ]
        if any(keyword in query_lower for keyword in dosage_keywords):
            return PharmaceuticalQueryType.DOSAGE_GUIDELINES

        # Check for general pharmaceutical research terms
        research_keywords = [
            "drug", "medication", "pharmaceutical", "treatment", "therapy",
            "medicine", "compound", "molecule"
        ]
        if any(keyword in query_lower for keyword in research_keywords):
            return PharmaceuticalQueryType.GENERAL_RESEARCH

        # Default to exploratory
        return PharmaceuticalQueryType.EXPLORATORY

    def _calculate_query_cost(self,
                            tokens: int,
                            cost_tier: str,
                            query_type: PharmaceuticalQueryType) -> float:
        """Calculate estimated cost for a pharmaceutical query."""
        if cost_tier == "free_tier":
            # Free tier has opportunity cost rather than direct cost
            # Value based on free tier allocation efficiency
            base_cost = 0.0
            opportunity_cost = tokens * 0.0001  # Small opportunity cost
            return opportunity_cost

        # Infrastructure tier cost estimation
        # Assuming $0.002 per 1K tokens (conservative estimate)
        base_cost = (tokens / 1000.0) * 0.002

        # Apply pharmaceutical query type multiplier for value assessment
        type_multiplier = {
            PharmaceuticalQueryType.DRUG_SAFETY: 0.8,  # Lower cost tolerance for critical queries
            PharmaceuticalQueryType.CLINICAL_TRIALS: 0.9,
            PharmaceuticalQueryType.DRUG_INTERACTIONS: 0.8,
            PharmaceuticalQueryType.MECHANISM_OF_ACTION: 1.0,
            PharmaceuticalQueryType.PHARMACOKINETICS: 1.0,
            PharmaceuticalQueryType.DOSAGE_GUIDELINES: 1.0,
            PharmaceuticalQueryType.GENERAL_RESEARCH: 1.1,
            PharmaceuticalQueryType.EXPLORATORY: 1.2
        }.get(query_type, 1.0)

        return base_cost * type_multiplier

    def _calculate_research_value_score(self, record: PharmaceuticalQueryRecord) -> float:
        """Calculate research value score for pharmaceutical queries."""
        base_value = self.query_type_values.get(record.query_type, 1.0)

        # Adjust based on cost efficiency
        cost_efficiency = 1.0
        if record.cost_tier == "free_tier":
            cost_efficiency = 2.0  # Higher value for free tier usage

        # Adjust based on query complexity (token usage)
        complexity_factor = min(2.0, record.estimated_tokens / 500.0)  # Cap at 2x

        return base_value * cost_efficiency * complexity_factor

    def create_research_project(self,
                              project_id: str,
                              project_name: str,
                              monthly_budget_usd: float,
                              priority_level: int = 3) -> ResearchProjectBudget:
        """Create a new research project budget."""
        if project_id in self.projects:
            logger.warning(f"Project {project_id} already exists, updating budget")

        project = ResearchProjectBudget(
            project_id=project_id,
            project_name=project_name,
            monthly_budget_usd=monthly_budget_usd,
            priority_level=priority_level
        )

        self.projects[project_id] = project
        logger.info(f"Created research project {project_id}: ${monthly_budget_usd}/month")

        return project

    def get_cost_analysis(self,
                         days_back: int = 30,
                         force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive cost analysis for pharmaceutical research.

        Args:
            days_back: Number of days to analyze
            force_refresh: Force cache refresh

        Returns:
            Comprehensive cost analysis report
        """
        # Check cache validity
        if (not force_refresh and
            self._cache_timestamp and
            self._analysis_cache and
            (datetime.now() - self._cache_timestamp).seconds < self._cache_ttl_seconds):
            return self._analysis_cache

        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_queries = [
            record for record in self.query_records
            if record.timestamp >= cutoff_date
        ]

        analysis = {
            "analysis_period": {
                "days_back": days_back,
                "cutoff_date": cutoff_date.isoformat(),
                "total_queries_analyzed": len(recent_queries)
            },
            "cost_breakdown": self._analyze_cost_breakdown(recent_queries),
            "pharmaceutical_insights": self._analyze_pharmaceutical_value(recent_queries),
            "free_tier_optimization": self._analyze_free_tier_usage(recent_queries),
            "project_analysis": self._analyze_project_costs(),
            "recommendations": self._generate_cost_recommendations(recent_queries)
        }

        # Update cache
        self._analysis_cache = analysis
        self._cache_timestamp = datetime.now()

        return analysis

    def _analyze_cost_breakdown(self, queries: List[PharmaceuticalQueryRecord]) -> Dict[str, Any]:
        """Analyze cost breakdown by various dimensions."""
        if not queries:
            return {"total_cost": 0, "query_count": 0}

        total_cost = sum(q.estimated_cost_usd for q in queries)
        free_tier_cost = sum(q.estimated_cost_usd for q in queries if q.cost_tier == "free_tier")
        infrastructure_cost = sum(q.estimated_cost_usd for q in queries if q.cost_tier == "infrastructure")

        # Cost by query type
        cost_by_type = defaultdict(float)
        queries_by_type = defaultdict(int)

        for query in queries:
            cost_by_type[query.query_type.value] += query.estimated_cost_usd
            queries_by_type[query.query_type.value] += 1

        return {
            "total_cost_usd": round(total_cost, 4),
            "free_tier_cost_usd": round(free_tier_cost, 4),
            "infrastructure_cost_usd": round(infrastructure_cost, 4),
            "cost_per_query_avg": round(total_cost / len(queries), 4),
            "cost_by_query_type": {
                qtype: {
                    "total_cost": round(cost, 4),
                    "query_count": queries_by_type[qtype],
                    "avg_cost": round(cost / queries_by_type[qtype], 4) if queries_by_type[qtype] > 0 else 0
                }
                for qtype, cost in cost_by_type.items()
            },
            "tier_distribution": {
                "free_tier_queries": len([q for q in queries if q.cost_tier == "free_tier"]),
                "infrastructure_queries": len([q for q in queries if q.cost_tier == "infrastructure"]),
                "free_tier_percentage": round(
                    len([q for q in queries if q.cost_tier == "free_tier"]) / len(queries) * 100, 1
                )
            }
        }

    def _analyze_pharmaceutical_value(self, queries: List[PharmaceuticalQueryRecord]) -> Dict[str, Any]:
        """Analyze pharmaceutical research value and ROI."""
        if not queries:
            return {}

        # Calculate total research value
        total_value = sum(q.research_value_score or 0 for q in queries)
        total_cost = sum(q.estimated_cost_usd for q in queries)

        # Value by query type
        value_by_type = defaultdict(list)
        for query in queries:
            if query.research_value_score:
                value_by_type[query.query_type.value].append(query.research_value_score)

        # High-value query analysis
        high_value_queries = [q for q in queries if (q.research_value_score or 0) >= 4.0]

        return {
            "total_research_value": round(total_value, 2),
            "roi_ratio": round(total_value / total_cost, 2) if total_cost > 0 else float('inf'),
            "high_value_queries": len(high_value_queries),
            "high_value_percentage": round(len(high_value_queries) / len(queries) * 100, 1),
            "avg_value_by_type": {
                qtype: {
                    "avg_value": round(statistics.mean(values), 2),
                    "max_value": round(max(values), 2),
                    "query_count": len(values)
                }
                for qtype, values in value_by_type.items() if values
            },
            "pharmaceutical_focus": {
                "safety_queries": len([q for q in queries if q.query_type == PharmaceuticalQueryType.DRUG_SAFETY]),
                "clinical_queries": len([q for q in queries if q.query_type == PharmaceuticalQueryType.CLINICAL_TRIALS]),
                "interaction_queries": len([q for q in queries if q.query_type == PharmaceuticalQueryType.DRUG_INTERACTIONS])
            }
        }

    def _analyze_free_tier_usage(self, queries: List[PharmaceuticalQueryRecord]) -> Dict[str, Any]:
        """Analyze free tier usage efficiency."""
        free_tier_queries = [q for q in queries if q.cost_tier == "free_tier"]

        if not queries:
            return {"efficiency_score": 0}

        efficiency_score = len(free_tier_queries) / len(queries)

        # Analyze free tier value extraction
        free_tier_value = sum(q.research_value_score or 0 for q in free_tier_queries)
        total_value = sum(q.research_value_score or 0 for q in queries)

        return {
            "efficiency_score": round(efficiency_score, 3),
            "free_tier_queries": len(free_tier_queries),
            "total_queries": len(queries),
            "free_tier_value_extraction": round(
                free_tier_value / total_value, 3
            ) if total_value > 0 else 0,
            "optimization_status": (
                "excellent" if efficiency_score >= 0.8 else
                "good" if efficiency_score >= 0.6 else
                "needs_improvement"
            )
        }

    def _analyze_project_costs(self) -> Dict[str, Any]:
        """Analyze costs across research projects."""
        if not self.projects:
            return {"project_count": 0}

        project_summaries = {}
        total_monthly_budgets = 0
        total_spent = 0

        for project_id, project in self.projects.items():
            status = project.get_budget_status()
            project_summaries[project_id] = status
            total_monthly_budgets += project.monthly_budget_usd
            total_spent += project.spent_this_month

        # Find high-utilization projects
        high_utilization_projects = [
            p for p in project_summaries.values()
            if p["budget_utilization"] > 0.8
        ]

        return {
            "project_count": len(self.projects),
            "total_monthly_budgets": round(total_monthly_budgets, 2),
            "total_spent_this_month": round(total_spent, 4),
            "overall_utilization": round(total_spent / total_monthly_budgets, 3) if total_monthly_budgets > 0 else 0,
            "high_utilization_projects": len(high_utilization_projects),
            "project_details": project_summaries
        }

    def _generate_cost_recommendations(self, queries: List[PharmaceuticalQueryRecord]) -> List[Dict[str, Any]]:
        """Generate actionable cost optimization recommendations."""
        recommendations = []

        if not queries:
            return recommendations

        # Free tier usage recommendation
        free_tier_queries = len([q for q in queries if q.cost_tier == "free_tier"])
        total_queries = len(queries)
        free_tier_pct = free_tier_queries / total_queries if total_queries > 0 else 0

        if free_tier_pct < 0.7:
            recommendations.append({
                "type": "free_tier_optimization",
                "priority": "high",
                "title": "Increase Free Tier Utilization",
                "description": f"Only {free_tier_pct*100:.1f}% of queries use free tier. Target 70%+ for cost optimization.",
                "action": "Review query batching and cloud-first configuration settings",
                "potential_savings_usd": round((0.7 - free_tier_pct) * total_queries * 0.01, 3)
            })

        # Query type optimization
        exploratory_queries = len([q for q in queries if q.query_type == PharmaceuticalQueryType.EXPLORATORY])
        if exploratory_queries > total_queries * 0.3:
            recommendations.append({
                "type": "query_optimization",
                "priority": "medium",
                "title": "Focus on High-Value Pharmaceutical Queries",
                "description": f"{exploratory_queries} exploratory queries detected. Focus on drug safety and clinical research.",
                "action": "Prioritize drug safety, clinical trials, and drug interaction queries",
                "research_value_impact": "high"
            })

        # Project budget recommendations
        for project_id, project in self.projects.items():
            status = project.get_budget_status()
            if status["budget_utilization"] > 0.9:
                recommendations.append({
                    "type": "budget_alert",
                    "priority": "critical",
                    "title": f"Project Budget Alert: {status['project_name']}",
                    "description": f"Project {project_id} has used {status['budget_utilization']*100:.1f}% of monthly budget",
                    "action": "Review project queries or increase budget allocation",
                    "project_id": project_id
                })

        return recommendations

    def _invalidate_cache(self) -> None:
        """Invalidate analysis cache."""
        self._analysis_cache = {}
        self._cache_timestamp = None

    def export_cost_report(self,
                          filepath: Optional[str] = None,
                          days_back: int = 30) -> str:
        """Export detailed cost analysis report to JSON file."""
        analysis = self.get_cost_analysis(days_back=days_back, force_refresh=True)

        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"pharmaceutical_cost_report_{timestamp}.json"

        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_period_days": days_back,
                "total_queries_analyzed": len(self.query_records),
                "projects_tracked": len(self.projects)
            },
            "cost_analysis": analysis,
            "raw_query_data": [
                {
                    "query_id": record.query_id,
                    "query_type": record.query_type.value,
                    "timestamp": record.timestamp.isoformat(),
                    "cost_tier": record.cost_tier,
                    "estimated_cost_usd": record.estimated_cost_usd,
                    "research_value_score": record.research_value_score,
                    "project_id": record.project_id
                }
                for record in self.query_records[-500:]  # Last 500 queries
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Cost analysis report exported to {filepath}")
        return filepath

# Convenience function for pharmaceutical cost tracking
def create_pharmaceutical_cost_tracker(
    enable_project_budgeting: bool = True
) -> PharmaceuticalCostAnalyzer:
    """
    Create pharmaceutical cost analyzer with optimal configuration.

    Args:
        enable_project_budgeting: Enable project-based budget tracking

    Returns:
        Configured PharmaceuticalCostAnalyzer
    """
    credit_tracker = PharmaceuticalCreditTracker()
    config = EnhancedRAGConfig.from_env()

    analyzer = PharmaceuticalCostAnalyzer(
        credit_tracker=credit_tracker,
        config=config
    )

    if enable_project_budgeting:
        # Create default pharmaceutical research project
        analyzer.create_research_project(
            project_id="default_pharma_research",
            project_name="General Pharmaceutical Research",
            monthly_budget_usd=50.0,  # Conservative default budget
            priority_level=3
        )

    return analyzer

if __name__ == "__main__":
    # Test pharmaceutical cost analyzer
    analyzer = create_pharmaceutical_cost_tracker()

    # Record sample queries
    analyzer.record_pharmaceutical_query(
        query_id="test_001",
        query_text="What are the contraindications for metformin in elderly patients with kidney disease?",
        cost_tier="free_tier",
        estimated_tokens=150,
        project_id="default_pharma_research",
        tags=["drug_safety", "elderly", "kidney"]
    )

    analyzer.record_pharmaceutical_query(
        query_id="test_002",
        query_text="Mechanism of action of ACE inhibitors in hypertension treatment",
        cost_tier="free_tier",
        estimated_tokens=200,
        project_id="default_pharma_research",
        tags=["mechanism", "cardiovascular"]
    )

    # Generate analysis
    analysis = analyzer.get_cost_analysis(days_back=7)
    print("Pharmaceutical Cost Analysis:")
    print(json.dumps(analysis, indent=2))