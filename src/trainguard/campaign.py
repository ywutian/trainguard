"""Persistent acceptance campaigns with positive and negative recovery evidence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from trainguard.benchmark import _run_metrics
from trainguard.config import ProjectConfig, load_config
from trainguard.controller import _controller_lock, resume, run
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


def _execute(directory: Path, result: dict) -> Path:
    base = ProjectConfig.model_validate(result["config"])
    try:
        preflight(base)
    except (RuntimeError, ValueError, OSError) as exc:
        result.update(status="BLOCKED", reason=str(exc))
        _persist(directory, result)
        return directory
    result["status"] = "RUNNING"
    _persist(directory, result)
    reference = Path(result["reference_dir"]) if result.get("reference_dir") else None
    if reference is not None:
        if not validate_runs(reference, reference)["passed"]:
            raise ValueError("acceptance reference evidence is invalid")
    else:
        path = directory / "reference-config.json"
        write_json_atomic(path, _case_config(base, "none", "none").model_dump())
        candidates = list((directory / "reference").glob("*/run.json"))
        if candidates:
            reference = candidates[-1].parent
            succeeded = resume(reference)
        else:
            reference, succeeded = run(path, directory / "reference")
        result["reference_dir"] = str(reference)
        if not succeeded or not validate_runs(reference, reference)["passed"]:
            result.update(status="FAILED", reason="reference failed")
            _persist(directory, result)
            return directory
        _persist(directory, result)
    for case in result["cases"]:
        if case["status"] == "PASSED":
            validation = validate_runs(reference, Path(case["run_dir"]))
            if validation["passed"] != case["expected_exact"]:
                raise ValueError(f"completed acceptance evidence changed: {case['name']}")
            continue
        case["status"] = "RUNNING"
        _persist(directory, result)
        config_copy = directory / f"{case['name']}-config.json"
        expected = _case_config(base, case["mode"], case["fault"], case["omit_state"])
        write_json_atomic(config_copy, expected.model_dump())
        try:
            existing = Path(case["run_dir"]) if case.get("run_dir") else None
            if existing is None:
                candidates = list((directory / case["name"]).glob("*/run.json"))
                existing = candidates[-1].parent if candidates else None
            if existing:
                if load_config(existing / "config.json").fingerprint() != expected.fingerprint():
                    raise ValueError("case configuration differs")
                run_dir, succeeded = existing, resume(existing)
            else:
                run_dir, succeeded = run(config_copy, directory / case["name"])
            case["run_dir"] = str(run_dir)
            _persist(directory, result)
            validation = (
                validate_runs(reference, run_dir)
                if succeeded
                else {"passed": False, "differences": ["training did not succeed"]}
            )
            with sqlite3.connect(run_dir / "run.sqlite3") as database:
                count = database.execute("SELECT COUNT(*) FROM recoveries").fetchone()[0]
            case.update(validation=validation, recovery_count=count, metrics=_run_metrics(run_dir))
            case["status"] = (
                "PASSED"
                if succeeded and count >= 1 and validation["passed"] == case["expected_exact"]
                else "FAILED"
            )
            case["reason"] = (
                None if case["status"] == "PASSED" else "recovery or comparison expectation failed"
            )
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
        for field in ("source_sha256", "torch", "python", "versions"):
            if result["environment"][field] != current[field]:
                raise ValueError(f"acceptance source or runtime {field} differs")
        return _execute(directory, result)
