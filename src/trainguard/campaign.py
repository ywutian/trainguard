"""Persistent acceptance campaigns with positive and negative recovery evidence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from trainguard.benchmark import _run_metrics
from trainguard.config import ProjectConfig, load_config
from trainguard.controller import RunActiveError, _controller_lock, resume, run
from trainguard.environment import environment_snapshot
from trainguard.events import utc_now, write_json_atomic
from trainguard.strategy import preflight
from trainguard.validation import validate_runs


def _case_config(base: ProjectConfig, mode: str, fault: str, omitted="none") -> ProjectConfig:
    raw = base.model_dump()
    raw["checkpoint"]["mode"] = mode
    raw["checkpoint"]["interval_steps"] = 1
    raw["checkpoint"]["keep_last_k"] = None
    raw["checkpoint"]["max_retained_bytes"] = None
    raw["fault"] = {"kind": fault, "step": 3 if fault != "none" else None, "rank": 0}
    raw["recovery"]["omit_state"] = omitted
    if fault == "hang":
        raw["recovery"]["progress_timeout_seconds"] = 15 if base.run.device == "cpu" else 60
    if mode == "async" and fault == "worker_exit":
        raw["fault"]["require_committed_step"] = 1
    return ProjectConfig.model_validate(raw)


def _persist(directory, result):
    write_json_atomic(directory / "acceptance.json", result)
    lines = [
        "# Recovery acceptance",
        "",
        f"Status: {result['status']}",
        "",
        f"Source: `{result['environment']['source_sha256']}`",
        "",
        "| Case | Result | Recovery count | Exact comparison |",
        "| --- | --- | ---: | --- |",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['name']} | {case['status']} | {case.get('recovery_count', 0)} | {case.get('validation', {}).get('passed', 'pending')} |"
        )
    lines.extend(
        [
            "",
            "Negative controls pass only when training recovers and the exact comparison detects omitted state.",
            "",
            "## Environment gates",
            "",
            f"Tested target: {result['config']['run']['device']} / {result['config']['run']['strategy']} / fixed {result['config']['run']['world_size']} ranks.",
            "",
            "CUDA and FSDP2 require their own successful campaign on actual CUDA devices. Multi-node scheduling, object storage transactions and power-loss durability require separate infrastructure acceptance.",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def _reconcile_or_retry(
    directory, result, owner, config_copy, root,
    run_key="run_dir", history_key="history",
):
    existing = Path(owner[run_key]) if owner.get(run_key) else None
    if existing is None:
        candidates = sorted(root.glob("*/run.json"), key=lambda path: path.stat().st_mtime)
        existing = candidates[-1].parent if candidates else None
    if existing is not None:
        if load_config(existing / "config.json").fingerprint() != load_config(config_copy).fingerprint():
            raise ValueError("acceptance configuration differs")
        owner[run_key] = str(existing)
        _persist(directory, result)
        original_status = json.loads((existing / "run.json").read_text())["status"]
        succeeded = resume(existing)  # A live worker group must prevent a new launch.
        if succeeded:
            return existing, True
        if original_status == "SUCCEEDED":
            raise ValueError("completed acceptance evidence is invalid")
        owner.setdefault(history_key, []).append({
            "run_dir": str(existing), "time": utc_now(),
            "reason": "incomplete training; workers reconciled before retry",
        })
    # A subsequent coordinator must discover any new run even if launch never returns.
    owner.pop(run_key, None)
    _persist(directory, result)
    return run(config_copy, root, allow_experiment=True)


def _case_evidence(reference, run_dir, case, expected):
    integrity = validate_runs(run_dir, run_dir)
    if not integrity["passed"]:
        raise ValueError(f"acceptance evidence is invalid: {integrity['differences']}")
    if load_config(run_dir / "config.json").fingerprint() != expected.fingerprint():
        raise ValueError("acceptance case configuration differs")
    status = json.loads((run_dir / "run.json").read_text())
    with sqlite3.connect((run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as database:
        count = database.execute(
            "SELECT COUNT(*) FROM recoveries WHERE run_id=?", (status["run_id"],)
        ).fetchone()[0]
        attempts = database.execute(
            "SELECT status FROM attempts WHERE run_id=? ORDER BY number", (status["run_id"],)
        ).fetchall()
    fault_log = run_dir / "attempts" / "attempt-001" / f"rank-{expected.fault.rank}.jsonl"
    injections = (
        [
            json.loads(line)
            for line in fault_log.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event_type") == "fault_injected"
        ]
        if fault_log.is_file()
        else []
    )
    fault_attributed = (
        len(injections) == 1
        and injections[0].get("fault_kind") == case["fault"]
        and injections[0].get("global_step") == expected.fault.step
    )
    validation = validate_runs(reference, run_dir)
    differences = set(validation["differences"])
    expected_difference = validation["passed"] == case["expected_exact"]
    if not case["expected_exact"]:
        allowed = {"final model_sha256 differs", "final optimizer_sha256 differs"}
        required = {"final model_sha256 differs"}
        if case["omit_state"] == "cursor":
            sequences = {
                f"rank {rank} {name} differs"
                for rank in range(expected.run.world_size)
                for name in ("effective sample sequence", "consumed batch sequence")
            }
            allowed |= sequences
            required |= sequences
        expected_difference = required <= differences <= allowed
    case.update(
        validation=validation,
        recovery_count=count,
        fault_attributed=fault_attributed,
        metrics=_run_metrics(run_dir),
    )
    return (
        count == 1
        and [row[0] for row in attempts] == ["FAILED", "SUCCEEDED"]
        and fault_attributed
        and expected_difference
    )


def _execute_cases(directory: Path, result: dict) -> Path:
    base = ProjectConfig.model_validate(result["config"])
    try:
        preflight(base)
    except (RuntimeError, ValueError, OSError) as exc:
        result.update(status="BLOCKED", reason=str(exc))
        _persist(directory, result)
        return directory
    result.update(status="RUNNING", reason=None)
    _persist(directory, result)
    reference = Path(result["reference_dir"]) if result.get("reference_dir") else None
    if reference is not None and result.get("reference_status") == "VALIDATED":
        if not validate_runs(reference, reference)["passed"]:
            raise ValueError("acceptance reference evidence is invalid")
    else:
        path = directory / "reference-config.json"
        write_json_atomic(path, _case_config(base, "none", "none").model_dump())
        try:
            reference, succeeded = _reconcile_or_retry(
                directory, result, result, path, directory / "reference",
                run_key="reference_dir", history_key="reference_history",
            )
        except RunActiveError as exc:
            result.update(status="BLOCKED", reason=str(exc))
            _persist(directory, result)
            raise
        result["reference_dir"] = str(reference)
        if not succeeded or not validate_runs(reference, reference)["passed"]:
            result.update(status="FAILED", reason="reference failed")
            _persist(directory, result)
            return directory
        result["reference_status"] = "VALIDATED"
        _persist(directory, result)
    for case in result["cases"]:
        expected = _case_config(base, case["mode"], case["fault"], case["omit_state"])
        if case["status"] == "PASSED":
            if not _case_evidence(reference, Path(case["run_dir"]), case, expected):
                raise ValueError(f"completed acceptance evidence changed: {case['name']}")
            continue
        case["status"] = "RUNNING"
        _persist(directory, result)
        config_copy = directory / f"{case['name']}-config.json"
        write_json_atomic(config_copy, expected.model_dump())
        try:
            run_dir, succeeded = _reconcile_or_retry(
                directory, result, case, config_copy, directory / case["name"]
            )
            case["run_dir"] = str(run_dir)
            _persist(directory, result)
            case["status"] = (
                "PASSED"
                if succeeded and _case_evidence(reference, run_dir, case, expected)
                else "FAILED"
            )
            case["reason"] = (
                None if case["status"] == "PASSED" else "recovery or comparison expectation failed"
            )
        except RunActiveError as exc:
            case.update(status="RUNNING", reason=str(exc))
            result.update(status="BLOCKED", reason=str(exc))
            _persist(directory, result)
            raise
        except BaseException as exc:
            case.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            result.update(status="FAILED", reason=case["reason"])
            _persist(directory, result)
            raise
        _persist(directory, result)
    result["status"] = (
        "SUCCEEDED" if all(case["status"] == "PASSED" for case in result["cases"]) else "FAILED"
    )
    result["finished_at"] = utc_now()
    _persist(directory, result)
    return directory


def _execute(directory: Path, result: dict) -> Path:
    try:
        return _execute_cases(directory, result)
    except RunActiveError:
        raise
    except BaseException as exc:
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        _persist(directory, result)
        raise


def run_campaign(config_path: Path, output_root: Path) -> Path:
    base = load_config(config_path.resolve())
    if base.training.total_steps < 4:
        raise ValueError("acceptance requires at least four optimizer updates")
    raw = base.model_dump()
    raw["model"]["dropout"] = max(raw["model"]["dropout"], 0.2)
    base = ProjectConfig.model_validate(raw)
    directory = (output_root / f"acceptance-{uuid.uuid4().hex[:12]}").resolve()
    directory.mkdir(parents=True)
    cases = [
        {
            "name": f"{mode}-{fault}",
            "mode": mode,
            "fault": fault,
            "omit_state": "none",
            "expected_exact": True,
            "status": "PENDING",
        }
        for mode, fault in [
            ("sync", "worker_exit"),
            ("async", "worker_exit"),
            ("sync", "save_interrupt"),
            ("async", "save_interrupt"),
            ("sync", "corrupt"), ("async", "corrupt"), ("sync", "hang"),
        ]
    ]
    cases.extend(
        {
            "name": f"omit-{omitted}",
            "mode": "sync",
            "fault": "worker_exit",
            "omit_state": omitted,
            "expected_exact": False,
            "status": "PENDING",
        }
        for omitted in ("rng", "optimizer", "cursor")
    )
    result = {
        "schema_version": 1,
        "config": base.model_dump(),
        "created_at": utc_now(),
        "status": "PENDING",
        "reference_dir": None,
        "cases": cases,
        "environment": environment_snapshot(base.run.world_size, base.run.device, directory),
    }
    with _controller_lock(directory):
        return _execute(directory, result)


def resume_campaign(directory: Path) -> Path:
    directory = directory.resolve()
    with _controller_lock(directory):
        result = json.loads((directory / "acceptance.json").read_text())
        if result.get("schema_version") != 1:
            raise ValueError("acceptance schema is not resumable")
        base = ProjectConfig.model_validate(result["config"])
        current = environment_snapshot(base.run.world_size, base.run.device, directory)
        for field in (
            "source_sha256", "torch", "python", "versions", "installed_distributions"
        ):
            if field not in result["environment"]:
                raise ValueError(f"acceptance runtime {field} identity is missing")
            if result["environment"][field] != current[field]:
                raise ValueError(f"acceptance source or runtime {field} differs")
        return _execute(directory, result)
