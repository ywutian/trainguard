"""Reconcile paired trial costs without presenting injected savings as realized revenue."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _amount(record: dict, key: str) -> Decimal:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise TypeError(f"{key} must be a nonnegative number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{key} must be a nonnegative number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{key} must be a nonnegative finite number")
    return amount


def calculate(ledger: dict) -> dict:
    if ledger.get("schema_version") != 1:
        raise ValueError("unsupported ledger schema")
    currency = ledger.get("currency")
    if not isinstance(currency, str) or len(currency) != 3 or not currency.isalpha():
        raise ValueError("currency must be a three-letter code")
    gpu_rate = _amount(ledger, "gpu_hour_rate")
    engineer_rate = _amount(ledger, "engineer_hour_rate")
    pairs = ledger.get("paired_injection_scenarios")
    overhead = ledger.get("trial_overhead")
    if not isinstance(pairs, list) or not pairs or not isinstance(overhead, dict):
        raise ValueError("paired injected scenarios and trial overhead are required")
    ids: set[str] = set()
    baseline_gpu = trial_gpu = baseline_engineer = trial_engineer = Decimal(0)
    for pair in pairs:
        if not isinstance(pair, dict):
            raise TypeError("scenario must be a mapping")
        identity = pair.get("scenario_id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("scenario IDs must be unique and nonempty")
        ids.add(identity)
        baseline_gpu += _amount(pair, "baseline_lost_gpu_hours")
        trial_gpu += _amount(pair, "trial_lost_gpu_hours")
        baseline_engineer += _amount(pair, "baseline_engineer_hours")
        trial_engineer += _amount(pair, "trial_engineer_hours")
    overhead_cost = (
        _amount(overhead, "checkpoint_gpu_hours") * gpu_rate
        + _amount(overhead, "additional_engineer_hours") * engineer_rate
        + _amount(overhead, "storage_cost")
        + _amount(overhead, "exercise_cost")
        + _amount(overhead, "service_fee")
    )
    gross = (baseline_gpu - trial_gpu) * gpu_rate + (
        baseline_engineer - trial_engineer
    ) * engineer_rate
    return {
        "schema_version": 1,
        "currency": currency.upper(),
        "paired_scenarios": len(pairs),
        "measurement_type": "paired_fault_injection_test_basket",
        "gross_delta_cost": str(gross),
        "trial_overhead_cost": str(overhead_cost),
        "test_basket_net_cost_delta": str(gross - overhead_cost),
        "annualized": False,
        "realized_savings_claim_eligible": False,
        "invoice_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"), parse_float=Decimal)
    result = calculate(ledger)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
