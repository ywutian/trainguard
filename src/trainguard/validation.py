"""Compare recovered training with an uninterrupted fixed-workload reference."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from trainguard.config import ProjectConfig, load_config
from trainguard.data import sample_ids_for_step
from trainguard.external_workload import (
    frozen_workload_path,
    read_verified_source,
    verify_v2_inputs,
)
from trainguard.privacy import key_for_run, verified_sample_event, verify_expected_sample_ids
from trainguard.records import parse_event, summary_errors
from trainguard.run_evidence import (
    RUNTIME_IDENTITY_FIELDS,
    completed_index_errors,
    indexed_schema_version,
    saved_completed_metadata_errors,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} is not a mapping")
    return value


def _verify_synthetic_sample_event(
    event: dict[str, Any], config: ProjectConfig, rank: int, key: bytes
) -> None:
    if config.external_workload is not None or config.data.kind != "synthetic":
        return
    cursor = event.get("consumed_batches")
    if type(cursor) is not int or cursor < 1:
        raise ValueError("synthetic sample cursor is invalid")
    batch_count = (
        1 if event["event_type"] == "batch_consumed"
        else config.training.gradient_accumulation_steps
    )
    if cursor < batch_count:
        raise ValueError("synthetic sample cursor precedes its update")
    expected_ids = [
        sample_id
        for batch in range(cursor - batch_count, cursor)
        for sample_id in sample_ids_for_step(
            batch, rank, config.run.world_size, config.training.batch_size_per_rank
        )
    ]
    verify_expected_sample_ids(event, expected_ids, key)


def _effective_samples(
    run_dir: Path, run_id: str, config: ProjectConfig
) -> tuple[dict[int, list[tuple[int, Any]]], list[str]]:
    database_path = run_dir / "run.sqlite3"
    world_size = config.run.world_size
    errors: list[str] = []
    effective: dict[int, dict[int, Any]] = {rank: {} for rank in range(world_size)}
    try:
        sample_key = key_for_run(run_dir) if config.run.profile == "guarded" else None
    except ValueError as exc:
        return {rank: [] for rank in range(world_size)}, [str(exc)]
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
        if not isinstance(attempt_id, str) or re.fullmatch(r"attempt-[0-9]{3,}", attempt_id) is None:
            errors.append("attempt index contains an invalid identity")
            continue
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
                sample_value = None
                if sample_key is not None and event.get("event_type") in {
                    "batch_consumed", "step_completed", "update_skipped"
                }:
                    try:
                        sample_value = verified_sample_event(event, sample_key)
                        _verify_synthetic_sample_event(event, config, rank, sample_key)
                    except ValueError as exc:
                        errors.append(f"{location}: {exc}")
                        continue
                if event.get("event_type") != "step_completed":
                    continue
                step = event.get("global_step")
                ids = sample_value if sample_key is not None else event.get("sample_ids")
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
                count = ids[0] if sample_key is not None else len(ids) if isinstance(ids, list) else 0
                if (
                    not config.training.gradient_accumulation_steps
                    <= count
                    <= config.training.batch_size_per_rank * config.training.gradient_accumulation_steps
                    or (
                        config.external_workload is None
                        and config.data.kind == "synthetic"
                        and count != config.training.batch_size_per_rank
                        * config.training.gradient_accumulation_steps
                    )
                    or (sample_key is None and (
                        not isinstance(ids, list)
                        or any(type(sample) is not int or sample < 0 for sample in ids)
                        or (config.external_workload is None and config.data.kind == "synthetic"
                            and len(set(ids)) != len(ids))
                    ))
                ):
                    errors.append(f"{location}: invalid sample IDs")
                    continue
                effective[rank][step] = ids
    return {rank: sorted(steps.items()) for rank, steps in effective.items()}, errors


def _sequence_digest(sequence: list[tuple[int, Any]]) -> str:
    payload = json.dumps(sequence, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recovery_lineage_errors(run_dir: Path, run_id: str, world_size: int) -> list[str]:
    """Bind each recovery decision to the state every progressing rank reports loading."""
    try:
        with sqlite3.connect(
            (run_dir / "run.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
        ) as database:
            attempts = database.execute(
                """SELECT attempt_id, number, status, resume_checkpoint,
                          resume_step, resume_consumed_batches
                   FROM attempts WHERE run_id=? ORDER BY number""",
                (run_id,),
            ).fetchall()
            recoveries = database.execute(
                """SELECT from_attempt, to_attempt, checkpoint_path, resume_step
                   FROM recoveries WHERE run_id=?""",
                (run_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        return [f"recovery lineage index is unreadable: {exc}"]

    errors: list[str] = []
    selections: dict[str, list[dict]] = {}
    if len(attempts) > 1:
        path = run_dir / "controller.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except (OSError, UnicodeError) as exc:
            errors.append(f"checkpoint selection evidence is unreadable: {exc}")
            lines = []
        for number, line in enumerate(lines, 1):
            if not line.endswith("\n") and number == len(lines):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                errors.append(f"checkpoint selection line {number} is invalid JSON")
                continue
            if not isinstance(event, dict):
                errors.append(f"checkpoint selection line {number} is not a mapping")
                continue
            if event.get("run_id") != run_id or event.get("event_type") != "checkpoint_selected":
                continue
            attempt_id = event.get("attempt_id")
            if (
                not isinstance(attempt_id, str)
                or not isinstance(event.get("checkpoint_path"), str)
                or not event["checkpoint_path"]
                or type(event.get("global_step")) is not int
                or not isinstance(event.get("manifest_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", event["manifest_sha256"]) is None
            ):
                errors.append(f"checkpoint selection line {number} has an invalid identity")
                continue
            selections.setdefault(attempt_id, []).append(event)
    decisions: dict[str, list[tuple]] = {}
    for recovery in recoveries:
        decisions.setdefault(recovery[1], []).append(recovery)
    if len(recoveries) != max(0, len(attempts) - 1):
        errors.append("recovery decision count differs from attempt history")

    for index, (attempt_id, number, status, checkpoint, step, cursor) in enumerate(attempts):
        if number != index + 1 or not isinstance(attempt_id, str) or (
            re.fullmatch(r"attempt-[0-9]{3,}", attempt_id) is None
        ):
            errors.append(f"attempt {index + 1}: recovery attempt order is invalid")
            continue
        if index == 0:
            if checkpoint is not None or step != 0 or cursor != 0 or attempt_id in decisions:
                errors.append(f"{attempt_id}: initial attempt has a recovery decision")
            continue
        if (
            not isinstance(checkpoint, str)
            or not checkpoint
            or type(step) is not int
            or step < 1
            or type(cursor) is not int
            or cursor < step
        ):
            errors.append(f"{attempt_id}: recovery boundary is invalid")
            continue
        matching = decisions.get(attempt_id, [])
        if len(matching) != 1 or matching[0] != (
            attempts[index - 1][0], attempt_id, checkpoint, step
        ):
            errors.append(f"{attempt_id}: recovery decision differs from selected checkpoint")
        selected = selections.get(attempt_id, [])
        selected_identities = {
            (event["checkpoint_path"], event["global_step"], event["manifest_sha256"])
            for event in selected
        }
        selected_identity = next(iter(selected_identities)) if len(selected_identities) == 1 else None
        if selected_identity is None or selected_identity[:2] != (checkpoint, step):
            errors.append(f"{attempt_id}: checkpoint selection differs from recovery decision")
            selected_digest = None
        else:
            selected_digest = selected_identity[2]

        for rank in range(world_size):
            path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            except (OSError, UnicodeError) as exc:
                errors.append(f"{attempt_id} rank {rank}: recovery log is unreadable: {exc}")
                continue
            loaded, started, progressed = [], [], False
            for line_number, line in enumerate(lines, 1):
                if not line.endswith("\n") and status != "SUCCEEDED" and line_number == len(lines):
                    continue
                try:
                    event = parse_event(line, run_id, attempt_id, rank)
                except ValueError:
                    continue  # The ordinary event audit reports malformed records.
                if event is None:
                    continue
                kind = event["event_type"]
                if kind == "state_loaded":
                    loaded.append(event)
                elif kind == "training_started":
                    started.append(event)
                elif kind == "step_completed":
                    progressed = True
            if status == "SUCCEEDED" or progressed or started:
                if len(loaded) != 1:
                    errors.append(f"{attempt_id} rank {rank}: expected one state_loaded event")
            elif len(loaded) > 1:
                errors.append(f"{attempt_id} rank {rank}: duplicate state_loaded events")
            if status == "SUCCEEDED" or progressed:
                if len(started) != 1:
                    errors.append(f"{attempt_id} rank {rank}: expected one training_started event")
            elif len(started) > 1:
                errors.append(f"{attempt_id} rank {rank}: duplicate training_started events")
            for event in loaded:
                if (
                    type(event.get("global_step")) is not int
                    or event["global_step"] != step
                    or type(event.get("consumed_batches")) is not int
                    or event["consumed_batches"] != cursor
                    or event.get("checkpoint_path") != checkpoint
                    or event.get("manifest_sha256") != selected_digest
                ):
                    errors.append(f"{attempt_id} rank {rank}: loaded state boundary differs")
            for event in started:
                if (
                    event.get("resumed_from") != checkpoint
                    or type(event.get("global_step")) is not int
                    or event["global_step"] != step
                    or type(event.get("consumed_batches")) is not int
                    or event["consumed_batches"] != cursor
                ):
                    errors.append(f"{attempt_id} rank {rank}: started state differs from recovery decision")
    return errors


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
    for name, directory, status, config in (
        ("reference", reference_dir, reference_status, reference_config),
        ("recovered", recovered_dir, recovered_status, recovered_config),
    ):
        if status.get("status") == "SUCCEEDED" and (
            status.get("run_schema_version") == 2
            or indexed_schema_version(directory, status.get("run_id")) == 2
        ):
            differences.extend(
                f"{name} {error}" for error in saved_completed_metadata_errors(status, config)
            )
            differences.extend(
                f"{name} {error}" for error in completed_index_errors(directory, status, config)
            )
    if differences:
        return {
            "passed": False, "differences": differences,
            "reference_run_id": reference_status.get("run_id"),
            "recovered_run_id": recovered_status.get("run_id"),
        }
    for name, directory, config, status in (
        ("reference", reference_dir, reference_config, reference_status),
        ("recovered", recovered_dir, recovered_config, recovered_status),
    ):
        if not config.matches_saved_config(status.get("config")):
            differences.append(f"{name} saved run configuration differs")
        if config.external_workload is not None:
            try:
                read_verified_source(config, path=frozen_workload_path(directory))
                verify_v2_inputs(config, run_dir=directory)
                if config.external_workload.version == 2:
                    read_verified_source(config)
                    verify_v2_inputs(config)
            except ValueError as exc:
                differences.append(f"{name} external workload evidence differs: {exc}")
    if reference_status["status"] != "SUCCEEDED":
        differences.append("reference run did not succeed")
    if recovered_status["status"] != "SUCCEEDED":
        differences.append("recovered run did not succeed")
    if reference_config.workload_fingerprint() != recovered_config.workload_fingerprint():
        differences.append("workload fingerprint differs")
    if reference_config.run.profile != recovered_config.run.profile:
        differences.append("run protection profile differs")
    if (
        reference_config.run.profile == "guarded"
        and reference_status.get("sample_key_id") != recovered_status.get("sample_key_id")
    ):
        differences.append("sample commitment key identity differs")
    if any(
        status.get("run_schema_version") == 2
        for status in (reference_status, recovered_status)
    ):
        reference_environment = reference_status.get("environment")
        recovered_environment = recovered_status.get("environment")
        if not isinstance(reference_environment, dict) or not isinstance(
            recovered_environment, dict
        ):
            differences.append("run environment identity is missing")
        else:
            for field in RUNTIME_IDENTITY_FIELDS:
                if (
                    field not in reference_environment
                    or field not in recovered_environment
                    or type(reference_environment[field]) is not type(recovered_environment[field])
                    or reference_environment[field] != recovered_environment[field]
                ):
                    differences.append(f"run environment {field} differs")
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
        "stream_sha256",
        "extra_sha256",
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
            "method": (
                "exact SHA-256 of rank states and keyed ordered sample/batch commitments"
                if reference_config.run.profile == "guarded"
                else "exact SHA-256 of rank states and effective sample/batch IDs"
            ),
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
    errors.extend(_recovery_lineage_errors(run_dir, run_id, config.run.world_size))
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
        sample_key = key_for_run(run_dir) if config.run.profile == "guarded" else None
    except ValueError as exc:
        return {rank: [] for rank in effective}, [str(exc)]
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
        if not isinstance(attempt_id, str) or re.fullmatch(r"attempt-[0-9]{3,}", attempt_id) is None:
            errors.append("attempt index contains an invalid identity")
            continue
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
            for number, line in enumerate(lines, 1):
                if not line.endswith("\n") and status != "SUCCEEDED":
                    continue
                try:
                    event = parse_event(line, run_id, attempt_id, rank)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if not event or event["event_type"] != "batch_consumed":
                    continue
                if sample_key is not None:
                    try:
                        ids = verified_sample_event(event, sample_key)
                        _verify_synthetic_sample_event(event, config, rank, sample_key)
                    except ValueError as exc:
                        errors.append(f"{attempt_id} rank {rank} line {number}: {exc}")
                        continue
                else:
                    ids = event.get("sample_ids")
                index = event.get("consumed_batches")
                if type(index) is not int or index != previous + 1:
                    errors.append(f"{attempt_id} rank {rank}: consumed batch order is invalid")
                    continue
                count = ids[0] if sample_key is not None else len(ids) if isinstance(ids, list) else 0
                if (
                    not 1 <= count <= config.training.batch_size_per_rank
                    or (sample_key is None and (
                        not isinstance(ids, list)
                        or any(type(item) is not int or item < 0 for item in ids)
                    ))
                ):
                    errors.append(f"{attempt_id} rank {rank}: invalid batch sample IDs")
                    continue
                effective[rank][index] = ids
                previous = index
    return {rank: sorted(rows.items()) for rank, rows in effective.items()}, errors
