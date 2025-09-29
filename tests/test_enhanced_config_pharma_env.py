import os

from src.enhanced_config import EnhancedRAGConfig


def test_from_env_parses_pharma_vars_isolated():
    env = {
        # Core pharma budgeting + tracking
        "PHARMA_PROJECT_ID": "proj-abc",
        "PHARMA_BUDGET_LIMIT_USD": "12.34",
        "PHARMA_RESEARCH_PROJECT_BUDGETING": "true",
        "PHARMA_COST_PER_QUERY_TRACKING": "false",
        # Feature flags
        "PHARMACEUTICAL_FEATURE_DRUG_INTERACTION_ANALYSIS": "false",
        "PHARMACEUTICAL_FEATURE_CLINICAL_TRIAL_PROCESSING": "true",
        "PHARMACEUTICAL_FEATURE_PHARMACOKINETICS_OPTIMIZATION": "false",
        # Global pharma mode
        "PHARMACEUTICAL_RESEARCH_MODE": "false",
        # Ensure deterministic unrelated toggles
        "ENABLE_NVIDIA_BUILD_FALLBACK": "false",
    }

    cfg = EnhancedRAGConfig.from_env(env)

    # Core budgeting / tracking
    assert cfg.pharma_project_id == "proj-abc"
    assert cfg.research_project_budget_limit_usd == 12.34
    assert cfg.research_project_budgeting is True
    assert cfg.cost_per_query_tracking is False

    # Pharma feature flags
    assert cfg.enable_drug_interaction_analysis is False
    assert cfg.enable_clinical_trial_processing is True
    assert cfg.enable_pharmacokinetics_optimization is False

    # Global pharma mode flag
    assert getattr(cfg, "pharmaceutical_research_mode", True) is False

    # Feature flags map reflects the above
    flags = cfg.get_feature_flags()
    assert flags.get("pharmaceutical_research_mode") is False
    assert flags.get("drug_interaction_analysis") is False
    assert flags.get("clinical_trial_processing") is True
    assert flags.get("pharmacokinetics_optimization") is False


