import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "calculate_pilot_value.py"
    spec = importlib.util.spec_from_file_location("calculate_pilot_value", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, name, record):
    content = (json.dumps(record, sort_keys=True) + "\n").encode()
    path = root / name
    path.write_bytes(content)
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}


def _record(scenario, arm, number, *, day="01"):
    if arm == "baseline":
        gpu = ("00:00:00Z", "10:00:00Z")
        engineer = ("00:00:00Z", "02:00:00Z")
    else:
        gpu = ("11:00:00Z", "14:00:00Z")
        engineer = ("11:00:00Z", "12:00:00Z")
    prefix = f"2026-09-{day}T"
    return {
        "schema_version": 1,
        "record_type": "paired_arm",
        "scenario_id": scenario["scenario_id"],
        "exercise_id": scenario["exercise_id"],
        "comparison_spec_sha256": scenario["comparison_spec_sha256"],
        "fault_spec_sha256": scenario["fault_spec_sha256"],
        "arm": arm,
        "event_id": f"{arm}-event-{number}",
        "run_id": f"{arm}-run-{number}",
        "gpu_loss_intervals": [{"resource_id": "gpu-1", "start_utc": prefix + gpu[0],
                                "end_utc": prefix + gpu[1]}],
        "engineer_intervals": [{"person_id": "engineer-1",
                                 "start_utc": prefix + engineer[0],
                                 "end_utc": prefix + engineer[1]}],
    }


def _add_scenario(root, ledger, number, *, day="01"):
    scenario = {
        "scenario_id": f"scenario-{number}",
        "exercise_id": f"exercise-{number}",
        "comparison_spec_sha256": "a" * 64,
        "fault_spec_sha256": "b" * 64,
    }
    scenario["baseline_record"] = _write(
        root, f"baseline-{number}.json", _record(scenario, "baseline", number, day=day)
    )
    scenario["trial_record"] = _write(
        root, f"trial-{number}.json", _record(scenario, "trial", number, day=day)
    )
    ledger["paired_injection_scenarios"].append(scenario)
    overhead = json.loads((root / "overhead.json").read_text())
    overhead["exercise_ids"].append(scenario["exercise_id"])
    ledger["trial_overhead"]["record"] = _write(root, "overhead.json", overhead)
    return scenario


def _ledger(root):
    overhead = {
        "schema_version": 1,
        "record_type": "trial_overhead",
        "record_id": "overhead-event",
        "exercise_ids": [],
        "gpu_overhead_intervals": [{"resource_id": "gpu-1",
                                    "start_utc": "2026-09-01T15:00:00Z",
                                    "end_utc": "2026-09-01T16:00:00Z"}],
        "engineer_overhead_intervals": [],
    }
    ledger = {
        "schema_version": 2,
        "currency": "USD",
        "gpu_hour_rate": "4",
        "engineer_hour_rate": "100",
        "paired_injection_scenarios": [],
        "trial_overhead": {
            "record": _write(root, "overhead.json", overhead),
            "storage_cost": "5", "exercise_cost": "10", "service_fee": "20",
        },
    }
    _add_scenario(root, ledger, 1)
    return ledger


def _calculate(root, ledger):
    return _module().calculate(ledger, evidence_root=root)


def test_paired_trial_value_uses_bound_raw_intervals_and_subtracts_all_costs(tmp_path):
    result = _calculate(tmp_path, _ledger(tmp_path))
    assert result["gross_delta_cost"] == "128"
    assert result["trial_overhead_cost"] == "39"
    assert result["test_basket_net_cost_delta"] == "89"
    assert result["paired_scenarios"] == 1
    assert result["evidence_file_integrity_checked"] is True
    assert result["evidence_authenticity_verified"] is False
    assert result["annualized"] is False
    assert result["realized_savings_claim_eligible"] is False
    assert result["invoice_eligible"] is False


def test_two_nonoverlapping_paired_exercises_can_be_counted(tmp_path):
    ledger = _ledger(tmp_path)
    _add_scenario(tmp_path, ledger, 2, day="02")
    result = _calculate(tmp_path, ledger)
    assert result["paired_scenarios"] == 2
    assert result["gross_delta_cost"] == "256"
    assert result["trial_overhead_cost"] == "39"


def test_different_scenario_ids_cannot_double_count_same_gpu_interval(tmp_path):
    ledger = _ledger(tmp_path)
    _add_scenario(tmp_path, ledger, 2)
    with pytest.raises(ValueError, match="overlap"):
        _calculate(tmp_path, ledger)


def test_different_scenario_ids_cannot_double_count_same_person_interval(tmp_path):
    ledger = _ledger(tmp_path)
    _add_scenario(tmp_path, ledger, 2, day="02")
    record_path = tmp_path / "baseline-2.json"
    record = json.loads(record_path.read_text())
    record["engineer_intervals"][0].update(
        start_utc="2026-09-01T01:00:00Z", end_utc="2026-09-01T02:00:00Z"
    )
    ledger["paired_injection_scenarios"][1]["baseline_record"] = _write(
        tmp_path, "baseline-2.json", record
    )
    with pytest.raises(ValueError, match="overlap"):
        _calculate(tmp_path, ledger)


def test_overhead_cannot_overlap_a_trial_measurement(tmp_path):
    ledger = _ledger(tmp_path)
    overhead = json.loads((tmp_path / "overhead.json").read_text())
    overhead["gpu_overhead_intervals"][0].update(
        start_utc="2026-09-01T13:00:00Z", end_utc="2026-09-01T14:00:00Z"
    )
    ledger["trial_overhead"]["record"] = _write(tmp_path, "overhead.json", overhead)
    with pytest.raises(ValueError, match="overlap"):
        _calculate(tmp_path, ledger)


def test_pair_requires_same_fault_and_comparison_identity(tmp_path):
    ledger = _ledger(tmp_path)
    record = json.loads((tmp_path / "trial-1.json").read_text())
    record["fault_spec_sha256"] = "c" * 64
    ledger["paired_injection_scenarios"][0]["trial_record"] = _write(
        tmp_path, "trial-1.json", record
    )
    with pytest.raises(ValueError, match="fault identity"):
        _calculate(tmp_path, ledger)


def test_distinct_raw_event_and_run_ids_are_required(tmp_path):
    ledger = _ledger(tmp_path)
    record = json.loads((tmp_path / "trial-1.json").read_text())
    record["event_id"] = "baseline-event-1"
    ledger["paired_injection_scenarios"][0]["trial_record"] = _write(
        tmp_path, "trial-1.json", record
    )
    with pytest.raises(ValueError, match="reused"):
        _calculate(tmp_path, ledger)


def test_raw_evidence_digest_and_overhead_basket_are_required(tmp_path):
    ledger = _ledger(tmp_path)
    (tmp_path / "trial-1.json").write_text("{}\n")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        _calculate(tmp_path, ledger)
    ledger = _ledger(tmp_path)
    overhead = json.loads((tmp_path / "overhead.json").read_text())
    overhead["exercise_ids"] = []
    ledger["trial_overhead"]["record"] = _write(tmp_path, "overhead.json", overhead)
    with pytest.raises(ValueError, match="full exercise basket"):
        _calculate(tmp_path, ledger)


def test_non_utc_or_reversed_intervals_fail_closed(tmp_path):
    ledger = _ledger(tmp_path)
    record = json.loads((tmp_path / "baseline-1.json").read_text())
    record["gpu_loss_intervals"][0]["start_utc"] = "2026-09-01T00:00:00+00:00"
    ledger["paired_injection_scenarios"][0]["baseline_record"] = _write(
        tmp_path, "baseline-1.json", record
    )
    with pytest.raises(ValueError, match="canonical UTC"):
        _calculate(tmp_path, ledger)
    record["gpu_loss_intervals"][0].update(
        start_utc="2026-09-01T10:00:00Z", end_utc="2026-09-01T00:00:00Z"
    )
    ledger["paired_injection_scenarios"][0]["baseline_record"] = _write(
        tmp_path, "baseline-1.json", record
    )
    with pytest.raises(ValueError, match="positive duration"):
        _calculate(tmp_path, ledger)


def test_legacy_ledger_and_missing_evidence_cannot_produce_an_output(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    ledger["schema_version"] = 1
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["calculate_pilot_value.py", str(ledger_path),
                                      "--output", str(output)])
    with pytest.raises(ValueError, match="legacy hour totals"):
        _module().main()
    assert not output.exists()
    ledger["schema_version"] = 2
    ledger["paired_injection_scenarios"][0].pop("baseline_record")
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="incomplete"):
        _module().main()
    assert not output.exists()


def test_negative_cost_and_duplicate_scenario_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    ledger["trial_overhead"]["storage_cost"] = "-1"
    with pytest.raises(ValueError, match="nonnegative"):
        _calculate(tmp_path, ledger)
    ledger = _ledger(tmp_path)
    ledger["paired_injection_scenarios"].append(ledger["paired_injection_scenarios"][0])
    with pytest.raises(ValueError, match="unique"):
        _calculate(tmp_path, ledger)


def test_duplicate_raw_file_bytes_and_linked_evidence_are_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    content = (tmp_path / "baseline-1.json").read_bytes()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(content)
    ledger["paired_injection_scenarios"][0]["trial_record"] = {
        "path": duplicate.name, "sha256": hashlib.sha256(content).hexdigest(),
    }
    with pytest.raises(ValueError, match="counted twice"):
        _calculate(tmp_path, ledger)
    ledger = _ledger(tmp_path)
    linked = tmp_path / "linked.json"
    linked.symlink_to(tmp_path / "trial-1.json")
    ledger["paired_injection_scenarios"][0]["trial_record"] = {
        "path": linked.name,
        "sha256": ledger["paired_injection_scenarios"][0]["trial_record"]["sha256"],
    }
    with pytest.raises(ValueError, match="symbolic link"):
        _calculate(tmp_path, ledger)


def test_existing_result_file_is_never_overwritten(tmp_path, monkeypatch):
    ledger = _ledger(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    output = tmp_path / "result.json"
    output.write_text("previous result\n")
    monkeypatch.setattr(sys, "argv", ["calculate_pilot_value.py", str(ledger_path),
                                      "--output", str(output)])
    with pytest.raises(FileExistsError, match="fresh output path"):
        _module().main()
    assert output.read_text() == "previous result\n"


def test_cli_report_binds_ledger_bytes_without_leaking_evidence_paths(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.json"
    content = (json.dumps(_ledger(tmp_path), sort_keys=True) + "\n").encode()
    ledger_path.write_bytes(content)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["calculate_pilot_value.py", str(ledger_path),
                                      "--output", str(output)])
    _module().main()
    report = json.loads(output.read_text())
    assert report["ledger_sha256"] == hashlib.sha256(content).hexdigest()
    assert report["evidence_authenticity_verified"] is False
    assert report["invoice_eligible"] is False
    assert str(tmp_path) not in output.read_text()
    assert "gpu-1" not in output.read_text()
    assert "engineer-1" not in output.read_text()
