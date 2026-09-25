"""Compare recovered training with an uninterrupted fixed-workload reference."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from trainguard.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_samples(run_dir: Path, run_id: str, world_size: int) -> dict[int, list[tuple[int, list[int]]]]:
    database_path = run_dir / "run.sqlite3"
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
        attempts = database.execute(
            "SELECT attempt_id, resume_step FROM attempts WHERE run_id=? ORDER BY number",
            (run_id,),
        ).fetchall()
    effective: dict[int, dict[int, list[int]]] = {rank: {} for rank in range(world_size)}
    for attempt_id, resume_step in attempts:
        for rank in range(world_size):
            effective[rank] = {
                step: ids for step, ids in effective[rank].items() if step <= resume_step
            }
            path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (event.get("run_id"), event.get("attempt_id"), event.get("rank")) != (
                    run_id, attempt_id, rank
                ):
                    continue
                step = event.get("global_step")
                ids = event.get("sample_ids")
                if (event.get("event_type") == "step_completed" and isinstance(step, int)
                        and step > resume_step and isinstance(ids, list)):
                    effective[rank][step] = ids
    return {rank: sorted(steps.items()) for rank, steps in effective.items()}


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
    for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256", "global_step"):
        if reference_summary.get(field) != recovered_summary.get(field):
            differences.append(f"final {field} differs")

    reference_samples = _effective_samples(
        reference_dir, reference_status["run_id"], reference_config.run.world_size
    )
    recovered_samples = _effective_samples(
        recovered_dir, recovered_status["run_id"], recovered_config.run.world_size
    )
    sample_counts = {}
    for rank in range(reference_config.run.world_size):
        expected = reference_samples[rank]
        actual = recovered_samples.get(rank, [])
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
