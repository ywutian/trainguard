"""Repeated local CPU timing runs with raw measurements and a readable report."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from trainguard.config import ProjectConfig, load_config
from trainguard.controller import run
from trainguard.events import utc_now, write_json_atomic
from trainguard.validation import validate_runs


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        values = [row["elapsed_seconds"] for row in rows if row["mode"] == mode]
        result[mode] = {
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
    return result


def mode_order(repeat: int) -> tuple[str, str, str]:
    modes = ("none", "sync", "async")
    offset = (repeat - 1) % len(modes)
    return modes[offset:] + modes[:offset]


def _run_metrics(run_dir: Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "checkpoint_count": 0,
        "staging_seconds": 0.0,
        "writing_seconds": 0.0,
        "checksum_commit_seconds": 0.0,
        "restart_seconds": 0.0,
        "recomputed_steps": 0,
    }
    for path in (run_dir / "attempts").glob("*/rank-0.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event_type") != "checkpoint_committed":
                continue
            metrics["checkpoint_count"] += 1
            for key in ("staging_seconds", "writing_seconds", "checksum_commit_seconds"):
                metrics[key] += float(event[key])
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        rows = database.execute(
            """SELECT old.finished_at, new.started_at, recovery.recomputed_steps
            FROM recoveries AS recovery
            JOIN attempts AS old ON old.attempt_id=recovery.from_attempt
            JOIN attempts AS new ON new.attempt_id=recovery.to_attempt"""
        ).fetchall()
    for old_finished, new_started, recomputed in rows:
        metrics["recomputed_steps"] += recomputed
        if old_finished:
            metrics["restart_seconds"] += max(
                0.0,
                (datetime.fromisoformat(new_started) - datetime.fromisoformat(old_finished))
                .total_seconds(),
            )
    return metrics


def _report_text(results: dict[str, Any]) -> str:
    lines = [
        "# CPU checkpoint benchmark",
        "",
        f"Recorded: {results['created_at']}",
        f"Workload fingerprint: `{results['workload_fingerprint']}`",
        (
            f"Environment: Python {results['environment']['python']}; "
            f"PyTorch {results['environment']['torch']}; "
            f"{results['environment']['platform']}; "
            f"{results['environment']['cpu_count']} logical CPUs."
        ),
        (
            f"World size: {results['environment']['world_size']}; "
            f"worker threads: {results['environment']['worker_threads']}."
        ),
        "",
        "## Elapsed time",
        "",
        "| Mode | Median (s) | Range (s) | Runs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in ("none", "sync", "async"):
        summary = results["summary"][mode]
        count = sum(row["mode"] == mode for row in results["raw_runs"])
        lines.append(
            f"| {mode} | {summary['median_seconds']:.3f} | "
            f"{summary['min_seconds']:.3f}–{summary['max_seconds']:.3f} | {count} |"
        )
    lines.extend([
        "",
        "## Raw measurements",
        "",
        (
            "| Mode | Repeat | Elapsed (s) | Checkpoints | Staging (s) | "
            "Writing elapsed (s) | Hash + commit (s) | Restart (s) | Recomputed steps | Valid |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in results["raw_runs"]:
        lines.append(
            f"| {row['mode']} | {row['repeat']} | {row['elapsed_seconds']:.3f} | "
            f"{row['checkpoint_count']} | {row['staging_seconds']:.3f} | "
            f"{row['writing_seconds']:.3f} | {row['checksum_commit_seconds']:.3f} | "
            f"{row['restart_seconds']:.3f} | {row['recomputed_steps']} | "
            f"{'yes' if row['validation_passed'] else 'no'} |"
        )
    lines.extend([
        "",
        (
            "Each raw row points to its run directory in `results.json`. "
            "Validation uses exact hashes and effective sample IDs (atol=0, rtol=0). "
            "Mode order rotates across repetitions."
        ),
        "",
        "## Measurement limits",
        "",
        (
            "Native asynchronous save can overlap writing with training. Its writing elapsed "
            "measurement includes that overlap, so phase totals must not be added to wall time. "
            "These runs use a local CPU filesystem; they do not measure GPU, multi-node, "
            "storage-delay, disk-loss, or host-power-failure behavior. This four-step workload "
            "is short and includes launch overhead; three repetitions cannot establish a "
            "general performance advantage."
        ),
        "",
    ])
    return "\n".join(lines)


def run_benchmark(
    config_path: Path, output_root: Path, repetitions: int = 3
) -> Path:
    if repetitions < 3:
        raise ValueError("benchmark requires at least three repetitions per mode")
    base = load_config(config_path.resolve())
    benchmark_dir = (output_root / f"benchmark-{uuid.uuid4().hex[:12]}").resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    reference_dir = None
    configs = {}
    for mode in ("none", "sync", "async"):
        raw = base.model_dump()
        raw["checkpoint"]["mode"] = mode
        raw["fault"] = {"kind": "none", "step": None, "rank": 0}
        raw["recovery"]["omit_state"] = "none"
        mode_config = ProjectConfig.model_validate(raw)
        config_copy = benchmark_dir / f"{mode}-config.json"
        write_json_atomic(config_copy, mode_config.model_dump())
        configs[mode] = config_copy
    for repeat in range(1, repetitions + 1):
        for mode in mode_order(repeat):
            started = time.monotonic()
            run_dir, succeeded = run(configs[mode], benchmark_dir / "runs")
            elapsed = time.monotonic() - started
            if not succeeded:
                raise RuntimeError(f"benchmark run failed: {run_dir}")
            if reference_dir is None:
                reference_dir = run_dir
            validation = validate_runs(reference_dir, run_dir)
            row = {
                "mode": mode,
                "repeat": repeat,
                "run_dir": str(run_dir),
                "elapsed_seconds": elapsed,
                "validation_passed": validation["passed"],
                "validation_differences": validation["differences"],
                **_run_metrics(run_dir),
            }
            rows.append(row)
            if not validation["passed"]:
                raise RuntimeError(f"benchmark correctness comparison failed: {run_dir}")
    results = {
        "created_at": utc_now(),
        "workload_fingerprint": base.workload_fingerprint(),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "world_size": base.run.world_size,
            "worker_threads": 1,
            "device": "cpu",
            "storage": "local filesystem",
        },
        "raw_runs": rows,
        "summary": summarize_rows(rows),
    }
    write_json_atomic(benchmark_dir / "results.json", results)
    (benchmark_dir / "report.md").write_text(_report_text(results), encoding="utf-8")
    return benchmark_dir
