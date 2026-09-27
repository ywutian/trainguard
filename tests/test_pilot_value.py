import importlib.util
from pathlib import Path

import pytest


def _calculator():
    path = Path(__file__).parents[1] / "scripts" / "calculate_pilot_value.py"
    spec = importlib.util.spec_from_file_location("calculate_pilot_value", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate


def _ledger():
    return {
        "schema_version": 1,
        "currency": "USD",
        "gpu_hour_rate": "4",
        "engineer_hour_rate": "100",
        "paired_injection_scenarios": [
            {"scenario_id": "one", "baseline_lost_gpu_hours": "10",
             "trial_lost_gpu_hours": "3", "baseline_engineer_hours": "2",
             "trial_engineer_hours": "1"}
        ],
        "trial_overhead": {"checkpoint_gpu_hours": "1", "additional_engineer_hours": "0",
                           "storage_cost": "5", "exercise_cost": "10", "service_fee": "20"},
    }


def test_paired_trial_value_subtracts_all_explicit_costs() -> None:
    result = _calculator()(_ledger())
    assert result["gross_delta_cost"] == "128"
    assert result["trial_overhead_cost"] == "39"
    assert result["test_basket_net_cost_delta"] == "89"
    assert result["annualized"] is False
    assert result["realized_savings_claim_eligible"] is False
    assert result["invoice_eligible"] is False


def test_duplicate_scenarios_and_negative_costs_are_rejected() -> None:
    ledger = _ledger()
    ledger["paired_injection_scenarios"].append(ledger["paired_injection_scenarios"][0])
    with pytest.raises(ValueError, match="unique"):
        _calculator()(ledger)
    ledger = _ledger()
    ledger["trial_overhead"]["storage_cost"] = "-1"
    with pytest.raises(ValueError, match="nonnegative"):
        _calculator()(ledger)
