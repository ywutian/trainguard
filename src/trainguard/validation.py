"""Compare recovered training with an uninterrupted fixed-workload reference."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from trainguard.config import ProjectConfig, load_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                    event = json.loads(line)
                except ValueError:
                    errors.append(f"{location}: invalid JSON")
                    continue
                if not isinstance(event, dict):
                    errors.append(f"{location}: event is not a mapping")
                    continue
                if (event.get("run_id"), event.get("attempt_id"), event.get("rank")) != (
                    run_id, attempt_id, rank
                ):
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
                if (not isinstance(ids, list) or len(ids) != config.training.batch_size_per_rank
                        or any(type(sample) is not int or sample < 0 for sample in ids)
                        or len(set(ids)) != len(ids)):
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
    reference_status = _read_json(reference_dir / "run.json")
    recovered_status = _read_json(recovered_dir / "run.json")
    reference_config = load_config(reference_dir / "config.json")
    recovered_config = load_config(recovered_dir / "config.json")
    reference_summary = _read_json(reference_dir / "summary.json")
    recovered_summary = _read_json(recovered_dir / "summary.json")
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
        for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256"):
            value = summary.get(field)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                differences.append(f"{name} final {field} is missing or invalid")
        if (type(summary.get("global_step")) is not int
                or summary["global_step"] != config.training.total_steps):
            differences.append(f"{name} final global_step differs from configured total")
    for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256", "global_step"):
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
        if [step for step, _ in expected] != list(range(1, reference_config.training.total_steps + 1)):
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
    return {
        "passed": not differences,
        "differences": differences,
        "reference_run_id": reference_status["run_id"],
        "recovered_run_id": recovered_status["run_id"],
        "comparison": {
            "tensor_atol": 0.0,
            "tensor_rtol": 0.0,
            "method": "exact SHA-256 of CPU state and effective sample IDs",
        },
        "effective_samples": sample_counts,
    }
