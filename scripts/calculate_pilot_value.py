"""Reconcile a paired fault-injection basket from bounded, nonoverlapping evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path, PurePosixPath

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
UTC_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
MAX_EVIDENCE_BYTES = 1024 * 1024
MAX_SCENARIOS = 1000
MAX_INTERVALS = 10000


def _fields(record: object, names: set[str], label: str) -> dict:
    if not isinstance(record, dict) or set(record) != names:
        raise ValueError(f"{label} fields are incomplete or unsupported")
    return record


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a stable nonempty identifier")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


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


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    found = {}
    for key, value in pairs:
        if key in found:
            raise ValueError("evidence contains duplicate JSON keys")
        found[key] = value
    return found


def _evidence(reference: object, root: Path, seen_digests: set[str]) -> dict:
    ref = _fields(reference, {"path", "sha256"}, "evidence reference")
    name = ref["path"]
    digest = _sha256(ref["sha256"], "evidence digest")
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("evidence path must be relative and canonical")
    relative = PurePosixPath(name)
    if (
        not name or relative.is_absolute() or relative.as_posix() != name
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise ValueError("evidence path must be relative and canonical")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("evidence path contains a symbolic link")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("evidence file is missing or outside the ledger directory")
    try:
        if path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ValueError("evidence file exceeds the size limit")
        content = path.read_bytes()
        if len(content) > MAX_EVIDENCE_BYTES:
            raise ValueError("evidence file exceeds the size limit")
        data = json.loads(content.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence file is unreadable") from exc
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("evidence file SHA-256 differs")
    if digest in seen_digests:
        raise ValueError("the same evidence file was counted twice")
    seen_digests.add(digest)
    if not isinstance(data, dict):
        raise TypeError("evidence record is not a JSON object")
    return data


def _instant(value: object) -> datetime:
    if not isinstance(value, str) or not UTC_INSTANT.fullmatch(value):
        raise ValueError("measurement times must use canonical UTC with Z")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("measurement time is invalid") from exc


def _hours(rows: object, identity_key: str, kind: str,
           intervals: dict[tuple[str, str], list[tuple[datetime, datetime]]],
           interval_counter: list[int]) -> Decimal:
    if not isinstance(rows, list):
        raise TypeError(f"{kind} intervals must be a list")
    seconds = Decimal(0)
    for row in rows:
        item = _fields(row, {identity_key, "start_utc", "end_utc"}, f"{kind} interval")
        identity = _identifier(item[identity_key], f"{kind} identity")
        start = _instant(item["start_utc"])
        end = _instant(item["end_utc"])
        if end <= start:
            raise ValueError(f"{kind} interval must have positive duration")
        interval_counter[0] += 1
        if interval_counter[0] > MAX_INTERVALS:
            raise ValueError("too many measurement intervals")
        intervals.setdefault((kind, identity), []).append((start, end))
        span = end - start
        seconds += Decimal(span.days * 86400 + span.seconds) + (
            Decimal(span.microseconds) / Decimal(1_000_000)
        )
    return seconds / Decimal(3600)


def _arm(record: dict, scenario: dict, arm: str, event_ids: set[str],
         run_ids: set[str], intervals: dict, interval_counter: list[int]) -> tuple[Decimal, Decimal]:
    _fields(record, {
        "schema_version", "record_type", "scenario_id", "exercise_id",
        "comparison_spec_sha256", "fault_spec_sha256", "arm", "event_id", "run_id",
        "gpu_loss_intervals", "engineer_intervals",
    }, f"{arm} event")
    if (
        type(record["schema_version"]) is not int or record["schema_version"] != 1
        or record["record_type"] != "paired_arm"
        or record["arm"] != arm
        or any(record[key] != scenario[key] for key in (
            "scenario_id", "exercise_id", "comparison_spec_sha256", "fault_spec_sha256"
        ))
    ):
        raise ValueError("paired raw events differ in exercise or fault identity")
    event_id = _identifier(record["event_id"], "event ID")
    run_id = _identifier(record["run_id"], "run ID")
    if event_id in event_ids or run_id in run_ids:
        raise ValueError("raw event or run is reused")
    event_ids.add(event_id)
    run_ids.add(run_id)
    return (
        _hours(record["gpu_loss_intervals"], "resource_id", "gpu", intervals,
               interval_counter),
        _hours(record["engineer_intervals"], "person_id", "person", intervals,
               interval_counter),
    )


def _check_nonoverlap(intervals: dict) -> None:
    for periods in intervals.values():
        ordered = sorted(periods)
        if any(current[0] < previous[1] for previous, current in pairwise(ordered)):
            raise ValueError("measurement intervals overlap for the same resource or person")


def calculate(ledger: dict, *, evidence_root: Path | None = None) -> dict:
    """Check source-file digests and event identities; authenticity still needs review."""
    _fields(ledger, {
        "schema_version", "currency", "gpu_hour_rate", "engineer_hour_rate",
        "paired_injection_scenarios", "trial_overhead",
    }, "ledger")
    if type(ledger["schema_version"]) is not int or ledger["schema_version"] != 2:
        raise ValueError("unsupported ledger schema; legacy hour totals cannot be verified")
    if evidence_root is None:
        raise ValueError("the ledger evidence directory is required")
    root = evidence_root.resolve(strict=True)
    currency = ledger["currency"]
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("currency must be an uppercase three-letter code")
    gpu_rate = _amount(ledger, "gpu_hour_rate")
    engineer_rate = _amount(ledger, "engineer_hour_rate")
    pairs = ledger["paired_injection_scenarios"]
    overhead = _fields(ledger["trial_overhead"], {
        "record", "storage_cost", "exercise_cost", "service_fee",
    }, "trial overhead")
    if not isinstance(pairs, list) or not pairs or len(pairs) > MAX_SCENARIOS:
        raise ValueError("a bounded list of paired scenarios is required")
    scenario_ids: set[str] = set()
    exercise_ids: set[str] = set()
    event_ids: set[str] = set()
    run_ids: set[str] = set()
    evidence_digests: set[str] = set()
    intervals: dict[tuple[str, str], list[tuple[datetime, datetime]]] = {}
    interval_counter = [0]
    baseline_gpu = trial_gpu = baseline_engineer = trial_engineer = Decimal(0)
    for scenario in pairs:
        pair = _fields(scenario, {
            "scenario_id", "exercise_id", "comparison_spec_sha256", "fault_spec_sha256",
            "baseline_record", "trial_record",
        }, "paired scenario")
        scenario_id = _identifier(pair["scenario_id"], "scenario ID")
        exercise_id = _identifier(pair["exercise_id"], "exercise ID")
        _sha256(pair["comparison_spec_sha256"], "comparison specification")
        _sha256(pair["fault_spec_sha256"], "fault specification")
        if scenario_id in scenario_ids or exercise_id in exercise_ids:
            raise ValueError("scenario and exercise IDs must be unique")
        scenario_ids.add(scenario_id)
        exercise_ids.add(exercise_id)
        baseline = _evidence(pair["baseline_record"], root, evidence_digests)
        trial = _evidence(pair["trial_record"], root, evidence_digests)
        baseline_gpu_hours, baseline_person_hours = _arm(
            baseline, pair, "baseline", event_ids, run_ids, intervals, interval_counter
        )
        trial_gpu_hours, trial_person_hours = _arm(
            trial, pair, "trial", event_ids, run_ids, intervals, interval_counter
        )
        baseline_gpu += baseline_gpu_hours
        trial_gpu += trial_gpu_hours
        baseline_engineer += baseline_person_hours
        trial_engineer += trial_person_hours
    overhead_record = _evidence(overhead["record"], root, evidence_digests)
    _fields(overhead_record, {
        "schema_version", "record_type", "record_id", "exercise_ids",
        "gpu_overhead_intervals", "engineer_overhead_intervals",
    }, "overhead event")
    if (
        type(overhead_record["schema_version"]) is not int
        or overhead_record["schema_version"] != 1
        or overhead_record["record_type"] != "trial_overhead"
        or not isinstance(overhead_record["exercise_ids"], list)
        or any(not isinstance(value, str) or not IDENTIFIER.fullmatch(value)
               for value in overhead_record["exercise_ids"])
        or len(overhead_record["exercise_ids"]) != len(exercise_ids)
        or set(overhead_record["exercise_ids"]) != exercise_ids
        or _identifier(overhead_record["record_id"], "overhead record ID") in event_ids
    ):
        raise ValueError("overhead record does not bind the full exercise basket")
    overhead_gpu = _hours(
        overhead_record["gpu_overhead_intervals"], "resource_id", "gpu", intervals,
        interval_counter,
    )
    overhead_engineer = _hours(
        overhead_record["engineer_overhead_intervals"], "person_id", "person", intervals,
        interval_counter,
    )
    _check_nonoverlap(intervals)
    overhead_cost = (
        overhead_gpu * gpu_rate + overhead_engineer * engineer_rate
        + _amount(overhead, "storage_cost") + _amount(overhead, "exercise_cost")
        + _amount(overhead, "service_fee")
    )
    gross = (baseline_gpu - trial_gpu) * gpu_rate + (
        baseline_engineer - trial_engineer
    ) * engineer_rate
    return {
        "schema_version": 2,
        "currency": currency,
        "paired_scenarios": len(pairs),
        "measurement_type": "paired_fault_injection_test_basket",
        "evidence_file_integrity_checked": True,
        "evidence_authenticity_verified": False,
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
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("choose a fresh output path for each ledger calculation")
    ledger_bytes = args.ledger.read_bytes()
    ledger = json.loads(ledger_bytes.decode("utf-8"), parse_float=Decimal,
                        object_pairs_hook=_no_duplicate_keys)
    result = calculate(ledger, evidence_root=args.ledger.parent)
    result["ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
