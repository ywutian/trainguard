"""Compare recovered training with an uninterrupted fixed-workload reference."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from trainguard.config import ProjectConfig, load_config
from trainguard.records import parse_event, summary_errors


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} is not a mapping")
    return value


def _effective_samples(
    run_dir: Path, run_id: str, config: ProjectConfig
) -> tuple[dict[int, list[tuple[int, list[int]]]], list[str]]:
    database_path = run_dir / "run.sqlite3"
    world_size = config.run.world_size
    errors: list[str] = []
    effective: dict[int, dict[int, list[int]]] = {rank: {} for rank in range(world_size)}
    try:
        with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            attempts = database.execute(
                "SELECT attempt_id, resume_step, status FROM attempts WHERE run_id=? ORDER BY number",
                (run_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        return {rank: [] for rank in range(world_size)}, [f"attempt index is unreadable: {exc}"]
    if not attempts:
        errors.append("attempt index contains no attempts")
    for attempt_id, resume_step, status in attempts:
        if type(resume_step) is not int or not 0 <= resume_step <= config.training.total_steps:
            errors.append(f"{attempt_id}: invalid resume step")
            continue
        for rank in range(world_size):
            effective[rank] = {
                step: ids for step, ids in effective[rank].items() if step <= resume_step
            }
            path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
            if not path.is_file():
                if status == "SUCCEEDED":
                    errors.append(f"{attempt_id} rank {rank}: missing rank log")
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            except (OSError, UnicodeError) as exc:
                errors.append(f"{attempt_id} rank {rank}: rank log is unreadable: {exc}")
                continue
            previous_step = resume_step
            seen: set[int] = set()
            for number, line in enumerate(lines, 1):
                location = f"{attempt_id} rank {rank} line {number}"
                if not line.endswith("\n") and status != "SUCCEEDED" and number == len(lines):
                    continue
                try:
                    event = parse_event(line, run_id, attempt_id, rank)
                except ValueError as exc:
                    errors.append(f"{location}: {exc}")
                    continue
                if event is None:
                    continue
                if event.get("event_type") != "step_completed":
                    continue
                step = event.get("global_step")
                ids = event.get("sample_ids")
                if type(step) is not int or not resume_step < step <= config.training.total_steps:
                    errors.append(f"{location}: invalid step")
                    continue
                if step in seen:
                    errors.append(f"{location}: duplicate step {step}")
                    continue
                seen.add(step)
                if step != previous_step + 1:
                    errors.append(f"{location}: invalid step order; expected {previous_step + 1}")
                previous_step = step
                if (
                    not isinstance(ids, list)
                    or not config.training.gradient_accumulation_steps
                    <= len(ids)
                    <= (
                        config.training.batch_size_per_rank
                        * config.training.gradient_accumulation_steps
                    )
                    or (
                        config.data.kind == "synthetic"
                        and len(ids)
                        != config.training.batch_size_per_rank
                        * config.training.gradient_accumulation_steps
                    )
                    or any(type(sample) is not int or sample < 0 for sample in ids)
                    or (config.data.kind == "synthetic" and len(set(ids)) != len(ids))
                ):
                    errors.append(f"{location}: invalid sample IDs")
                    continue
                effective[rank][step] = ids
    return {rank: sorted(steps.items()) for rank, steps in effective.items()}, errors


def _sequence_digest(sequence: list[tuple[int, list[int]]]) -> str:
    payload = json.dumps(sequence, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_runs(reference_dir: Path, recovered_dir: Path) -> dict[str, Any]:
    reference_dir = reference_dir.resolve()
    recovered_dir = recovered_dir.resolve()
    try:
        reference_status = _read_json(reference_dir / "run.json")
        recovered_status = _read_json(recovered_dir / "run.json")
        reference_config = load_config(reference_dir / "config.json")
        recovered_config = load_config(recovered_dir / "config.json")
        reference_summary = _read_json(reference_dir / "summary.json")
        recovered_summary = _read_json(recovered_dir / "summary.json")
        for status in (reference_status, recovered_status):
            if not isinstance(status.get("run_id"), str) or not isinstance(
                status.get("status"), str
            ):
                raise TypeError("run identity or status is invalid")
    except (OSError, ValueError, UnicodeError, TypeError) as exc:
        return {
            "passed": False,
            "differences": [f"run metadata or summary is unreadable: {exc}"],
            "reference_run_id": None,
            "recovered_run_id": None,
        }
    differences = []
    if reference_status["status"] != "SUCCEEDED":
        differences.append("reference run did not succeed")
    if recovered_status["status"] != "SUCCEEDED":
        differences.append("recovered run did not succeed")
    if reference_config.workload_fingerprint() != recovered_config.workload_fingerprint():
        differences.append("workload fingerprint differs")
    for name, summary, config in (
        ("reference", reference_summary, reference_config),
        ("recovered", recovered_summary, recovered_config),
    ):
        differences.extend(f"{name} {error}" for error in summary_errors(summary, config))
    for directory, status, config, summary, name in (
        (reference_dir, reference_status, reference_config, reference_summary, "reference"),
        (recovered_dir, recovered_status, recovered_config, recovered_summary, "recovered"),
    ):
        if summary.get("state_schema_version") == 2:
            attempt_id = status.get("attempt_id")
            if not isinstance(attempt_id, str):
                differences.append(f"{name} final attempt identity is invalid")
            else:
                differences.extend(
                    f"{name} {error}"
                    for error in completion_errors(
                        directory, attempt_id, config, status["run_id"], summary
                    )
                )
    for field in (
        "model_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "global_step",
        "scaler_sha256",
        "consumed_batches",
        "optimizer_updates",
    ):
        if reference_summary.get(field) != recovered_summary.get(field):
            differences.append(f"final {field} differs")

    reference_samples, reference_errors = _effective_samples(
        reference_dir, reference_status["run_id"], reference_config
    )
    recovered_samples, recovered_errors = _effective_samples(
        recovered_dir, recovered_status["run_id"], recovered_config
    )
    differences.extend(f"reference {error}" for error in reference_errors)
    differences.extend(f"recovered {error}" for error in recovered_errors)
    sample_counts = {}
    for rank in range(reference_config.run.world_size):
        expected = reference_samples[rank]
        actual = recovered_samples.get(rank, [])
        if [step for step, _ in expected] != list(
            range(1, reference_config.training.total_steps + 1)
        ):
            differences.append(f"reference rank {rank} effective sample steps are incomplete")
        steps = [step for step, _ in actual]
        if steps != list(range(1, recovered_config.training.total_steps + 1)):
            differences.append(f"rank {rank} effective sample steps are incomplete")
        if actual != expected:
            differences.append(f"rank {rank} effective sample sequence differs")
        sample_counts[str(rank)] = {
            "reference_steps": len(expected),
            "recovered_steps": len(actual),
            "reference_sha256": _sequence_digest(expected),
            "recovered_sha256": _sequence_digest(actual),
        }
    batch_comparison = {}
    if (
        reference_summary.get("state_schema_version") == 2
        or recovered_summary.get("state_schema_version") == 2
    ):
        reference_batches, errors = _effective_batches(
            reference_dir, reference_status["run_id"], reference_config
        )
        differences.extend(f"reference {error}" for error in errors)
        recovered_batches, errors = _effective_batches(
            recovered_dir, recovered_status["run_id"], recovered_config
        )
        differences.extend(f"recovered {error}" for error in errors)
        for rank in range(reference_config.run.world_size):
            if reference_batches[rank] != recovered_batches.get(rank):
                differences.append(f"rank {rank} consumed batch sequence differs")
            batch_comparison[str(rank)] = {
                "reference_batches": len(reference_batches[rank]),
                "recovered_batches": len(recovered_batches.get(rank, [])),
            }
    return {
        "passed": not differences,
        "differences": differences,
        "reference_run_id": reference_status["run_id"],
        "recovered_run_id": recovered_status["run_id"],
        "comparison": {
            "tensor_atol": 0.0,
            "tensor_rtol": 0.0,
            "method": "exact SHA-256 of rank states and effective sample/batch IDs",
        },
        "effective_samples": sample_counts,
        "effective_batches": batch_comparison,
    }


def completion_errors(
    run_dir: Path, attempt_id: str, config: ProjectConfig, run_id: str, summary: Any
) -> list[str]:
    errors = summary_errors(summary, config, run_id, attempt_id)
    if errors:
        return errors
    samples, audit = _effective_samples(run_dir, run_id, config)
    errors.extend(audit)
    expected_steps = list(range(1, config.training.total_steps + 1))
    if summary.get("state_schema_version") == 2:
        batches, batch_errors = _effective_batches(run_dir, run_id, config)
        errors.extend(batch_errors)
        for rank in range(config.run.world_size):
            if [index for index, _ in batches[rank]] != list(
                range(1, summary["consumed_batches"] + 1)
            ):
                errors.append(f"rank {rank}: consumed batch evidence is incomplete")
    for rank in range(config.run.world_size):
        if [step for step, _ in samples[rank]] != expected_steps:
            errors.append(f"rank {rank}: effective sample steps are incomplete")
        path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
        completed = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = parse_event(line, run_id, attempt_id, rank)
                if event and event["event_type"] == "training_completed":
                    completed.append(event["global_step"])
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append(f"rank {rank}: completion log invalid: {exc}")
        if completed != [config.training.total_steps]:
            errors.append(f"rank {rank}: final completion missing or invalid")
    return errors


def _effective_batches(run_dir: Path, run_id: str, config: ProjectConfig):
    effective = {rank: {} for rank in range(config.run.world_size)}
    errors = []
    try:
        with sqlite3.connect(
            (run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        ) as database:
            attempts = database.execute(
                "SELECT attempt_id, status, resume_consumed_batches FROM attempts WHERE run_id=? ORDER BY number",
                (run_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        return {rank: [] for rank in effective}, [f"attempt index is unreadable: {exc}"]
    for attempt_id, status, cursor in attempts:
        if type(cursor) is not int or cursor < 0:
            errors.append(f"{attempt_id}: invalid consumed batch cursor")
            continue
        for rank, prior in effective.items():
            effective[rank] = {index: ids for index, ids in prior.items() if index <= cursor}
            path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
            if not path.exists():
                continue
            previous = cursor
            try:
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            except (OSError, UnicodeError) as exc:
                errors.append(f"{attempt_id} rank {rank}: rank log is unreadable: {exc}")
                continue
            for line in lines:
                if not line.endswith("\n") and status != "SUCCEEDED":
                    continue
                try:
                    event = parse_event(line, run_id, attempt_id, rank)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if not event or event["event_type"] != "batch_consumed":
                    continue
                index, ids = event.get("consumed_batches"), event.get("sample_ids")
                if type(index) is not int or index != previous + 1:
                    errors.append(f"{attempt_id} rank {rank}: consumed batch order is invalid")
                    continue
                if (
                    not isinstance(ids, list)
                    or not 1 <= len(ids) <= config.training.batch_size_per_rank
                    or any(type(item) is not int or item < 0 for item in ids)
                ):
                    errors.append(f"{attempt_id} rank {rank}: invalid batch sample IDs")
                    continue
                effective[rank][index] = ids
                previous = index
    return {rank: sorted(rows.items()) for rank, rows in effective.items()}, errors
