def test_fetch_credit_burn_snapshot_handles_missing(monkeypatch):
    import streamlit_app

    # Monkeypatch shared tracker provider to return a dummy with empty analytics
    class DummyTracker:
        def get_pharmaceutical_analytics(self):
            return {}

    monkeypatch.setattr(streamlit_app, "get_shared_credit_tracker", lambda: DummyTracker())

    snap = streamlit_app.fetch_credit_burn_snapshot()
    assert isinstance(snap, dict)
    assert snap == {}


def test_fetch_credit_burn_snapshot_values(monkeypatch):
    import streamlit_app

    class DummyTracker:
        def get_pharmaceutical_analytics(self):
            return {
                "base_monitor_summary": {"by_service": {"embedding": 3, "reranking": 2}},
                "daily_burn": {"used_today": 5, "burn_rate": 0.1},
                "performance_metrics": {"total_tokens_consumed": 1500, "avg_tokens_per_query": 300},
            }

    monkeypatch.setattr(streamlit_app, "get_shared_credit_tracker", lambda: DummyTracker())

    snap = streamlit_app.fetch_credit_burn_snapshot()
    assert snap["used_today"] == 5
    assert snap["burn_rate"] == 0.1
    assert snap["service_counts"]["embedding"] == 3
    assert snap["tokens_month"] == 1500
    assert snap["avg_tokens_per_query"] == 300
