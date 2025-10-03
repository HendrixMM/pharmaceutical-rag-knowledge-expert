from src.monitoring.credit_tracker import PharmaceuticalCreditTracker
from src.nemo_retriever_client import NVIDIABuildCreditsMonitor


def test_daily_burn_rate_increments_and_reports():
    mon = NVIDIABuildCreditsMonitor(api_key="k")
    # Log a few calls
    mon.log_api_call("embedding", tokens_used=1)
    mon.log_api_call("reranking", tokens_used=2)

    burn = mon.daily_burn_rate()
    assert isinstance(burn, dict)
    # used_today should reflect two calls
    assert burn.get("used_today") >= 2
    # burn_rate computed against ~333/day budget
    assert burn.get("burn_rate") >= 0


def test_tracker_attaches_to_monitor_and_logs_without_recursion():
    mon = NVIDIABuildCreditsMonitor(api_key="k")
    tracker = PharmaceuticalCreditTracker(base_monitor=mon)

    # Initially zero credits used
    assert mon.credits_used == 0

    tracker.track_pharmaceutical_query(
        query_type="drug_interaction",
        model_used="embedding",
        tokens_consumed=10,
        response_time_ms=100,
        research_context="ACE inhibitor interaction",
    )

    # Credits reflect exactly one request logged (no recursion)
    assert mon.credits_used == 1

    # Query type breakdown present via monitor summary
    summary = mon.get_usage_summary()
    assert summary.get("by_query_type", {}).get("drug_interaction", 0) >= 1
