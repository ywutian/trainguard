import pytest


def test_interval_respects_rollback_budget():
    from trainguard.policy import suggest_interval
    result = suggest_interval(overhead_seconds=1, mtbf_seconds=100, commit_lag_seconds=2,
                              rto_seconds=3, rollback_budget_seconds=10, upload_seconds=4)
    assert result['feasible']
    assert result['interval_seconds'] == 8
    assert result['estimated_waste_fraction'] == pytest.approx(0.215)


def test_upload_exceeding_budget_is_reported():
    from trainguard.policy import suggest_interval
    result = suggest_interval(overhead_seconds=1, mtbf_seconds=100, commit_lag_seconds=2,
                              rto_seconds=3, rollback_budget_seconds=10, upload_seconds=12)
    assert not result['feasible']
    assert result['interval_seconds'] is None
