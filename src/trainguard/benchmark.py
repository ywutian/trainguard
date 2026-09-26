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


def summarize_rows(
    rows: list[dict[str, Any]], metric: str = "elapsed_seconds"
) -> dict[str, dict[str, float]]:
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        values = [row[metric] for row in rows if row["mode"] == mode]
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
        "checkpoint_bytes": 0,
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
            manifest = json.loads(
                (Path(event["checkpoint_path"]) / "manifest.json").read_text(encoding="utf-8")
            )
            metrics["checkpoint_bytes"] += sum(entry["size"] for entry in manifest["files"])
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
        f"Warm-up repetitions per mode: {results['warmups']}; excluded from statistics.",
        f"Measured repetitions per mode: {results['repetitions']}.",
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
        "", "## Worker training window", "",
        "| Mode | Median (s) | Range (s) | Runs |",
        "| --- | ---: | ---: | ---: |",
    ])
    for mode in ("none", "sync", "async"):
        summary = results["training_summary"][mode]
        count = sum(row["mode"] == mode for row in results["raw_runs"])
        lines.append(
            f"| {mode} | {summary['median_seconds']:.3f} | "
            f"{summary['min_seconds']:.3f}–{summary['max_seconds']:.3f} | {count} |"
        )
    for title, rows in (
        ("Raw measurements", results["raw_runs"]),
        ("Warm-up measurements (excluded)", results["warmup_runs"]),
    ):
        lines.extend([
            "", f"## {title}", "",
            ("| Mode | Repeat | Elapsed (s) | Training (s) | Checkpoints | Payload (MiB) | "
             "Load 1m before / after | Valid |"),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in rows:
            lines.append(
                f"| {row['mode']} | {row['repeat']} | {row['elapsed_seconds']:.3f} | "
                f"{row['training_seconds']:.3f} | {row['checkpoint_count']} | "
                f"{row['checkpoint_bytes'] / 2**20:.3f} | "
                f"{row['load_average_before'][0]:.2f} / {row['load_average_after'][0]:.2f} | "
                f"{'yes' if row['validation_passed'] else 'no'} |"
            )
    lines.extend([
        "", "## Measured save and recovery phases", "",
        ("| Mode | Repeat | Staging (s) | Completion lag (s) | Hash + commit (s) | "
         "Restart (s) | Recomputed steps |"),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in results["raw_runs"]:
        lines.append(
            f"| {row['mode']} | {row['repeat']} | {row['staging_seconds']:.3f} | "
            f"{row['writing_seconds']:.3f} | {row['checksum_commit_seconds']:.3f} | "
            f"{row['restart_seconds']:.3f} | {row['recomputed_steps']} |"
        )
    lines.extend([
        "",
        (
            "Each raw row points to its run directory in `results.json`. "
            "Validation uses exact hashes and effective sample IDs (atol=0, rtol=0). "
            "Mode order rotates across repetitions. The exact base configuration is in "
            "`results.json`, and mode configurations are copied beside this report. "
            "Payload bytes sum committed manifest files across all checkpoints in a run, "
            "excluding the manifest and commit marker. Load averages record host activity "
            "over 1, 5 and 15 minutes before and after each run."
        ),
        "",
        "## Measurement limits",
        "",
        (
            "Completion lag (the raw writing_seconds field) is elapsed time after staging until "
            "the trainer observes completion at the rank barrier. It includes rank-state writes; "
            "for asynchronous save it also includes overlapping "
            "training and may exceed actual I/O time, so phase totals must not be added to wall time. "
            "Worker training time starts after process-group and model initialization and ends "
            "after the final training barrier; it includes checkpoint work but excludes launch. "
            "These runs use a local CPU filesystem; they do not measure GPU, multi-node, "
            "storage-delay, disk-loss, or host-power-failure behavior. Short workloads and "
            "few repetitions cannot establish a general performance advantage. "
            "The host is not isolated; load snapshots do not control other workloads, "
            "filesystem cache, or thermal effects. Warm-up rounds prepare the host and cache; "
            "each measured run still launches new workers."
        ),
        "",
    ])
    return "\n".join(lines)


def run_benchmark(
    config_path: Path, output_root: Path, repetitions: int = 3, warmups: int = 1
) -> Path:
    if repetitions < 3:
        raise ValueError("benchmark requires at least three repetitions per mode")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    base = load_config(config_path.resolve())
    benchmark_dir = (output_root / f"benchmark-{uuid.uuid4().hex[:12]}").resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
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
    for phase_rows, count in ((warmup_rows, warmups), (rows, repetitions)):
        for repeat in range(1, count + 1):
            for mode in mode_order(repeat):
                load_before = list(os.getloadavg())
                started = time.monotonic()
                run_dir, succeeded = run(configs[mode], benchmark_dir / "runs")
                elapsed = time.monotonic() - started
                load_after = list(os.getloadavg())
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
                    "training_seconds": json.loads(
                        (run_dir / "summary.json").read_text(encoding="utf-8")
                    )["training_elapsed_seconds"],
                    "load_average_before": load_before,
                    "load_average_after": load_after,
                    "validation_passed": validation["passed"],
                    "validation_differences": validation["differences"],
                    **_run_metrics(run_dir),
                }
                phase_rows.append(row)
                if not validation["passed"]:
                    raise RuntimeError(f"benchmark correctness comparison failed: {run_dir}")
    results = {
        "created_at": utc_now(),
        "config": base.model_dump(),
        "repetitions": repetitions,
        "warmups": warmups,
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
        "warmup_runs": warmup_rows,
        "summary": summarize_rows(rows),
        "training_summary": summarize_rows(rows, "training_seconds"),
    }
    write_json_atomic(benchmark_dir / "results.json", results)
    (benchmark_dir / "report.md").write_text(_report_text(results), encoding="utf-8")
    return benchmark_dir
