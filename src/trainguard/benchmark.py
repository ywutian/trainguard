"""Repeated local CPU timing runs with raw measurements and a readable report."""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from trainguard.config import ProjectConfig, load_config
from trainguard.controller import RunActiveError, _controller_lock, resume, run
from trainguard.environment import environment_snapshot
from trainguard.events import utc_now, write_json_atomic
from trainguard.run_evidence import trusted_measurement
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
    orders = (
        ("none", "sync", "async"),
        ("sync", "async", "none"),
        ("async", "none", "sync"),
        ("none", "async", "sync"),
        ("async", "sync", "none"),
        ("sync", "none", "async"),
    )
    return orders[(repeat - 1) % 6]


def _run_metrics(run_dir: Path) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "checkpoint_count": 0,
        "checkpoint_bytes": 0,
        "staging_seconds": 0.0,
        "writing_seconds": 0.0,
        "checksum_commit_seconds": 0.0,
        "restart_seconds": 0.0,
        "recomputed_steps": 0,
        "preparation_seconds": 0.0,
        "upload_seconds": 0.0,
        "main_thread_wait_seconds": 0.0,
        "eligibility_lag_seconds": 0.0,
        "recovery_rto_seconds": 0.0,
    }
    for path in (run_dir / "attempts").glob("*/rank-0.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event_type") != "checkpoint_committed":
                continue
            metrics["checkpoint_count"] += 1
            if "checkpoint_bytes" in event:
                metrics["checkpoint_bytes"] += event["checkpoint_bytes"]
            else:
                manifest = json.loads(
                    (Path(event["checkpoint_path"]) / "manifest.json").read_text()
                )
                metrics["checkpoint_bytes"] += sum(entry["size"] for entry in manifest["files"])
            for key in (
                "staging_seconds",
                "writing_seconds",
                "checksum_commit_seconds",
                "preparation_seconds",
                "upload_seconds",
                "main_thread_wait_seconds",
                "eligibility_lag_seconds",
            ):
                metrics[key] += float(event.get(key, 0.0))
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        rows = database.execute(
            """SELECT old.finished_at, new.started_at, recovery.recomputed_steps,
                   recovery.from_attempt, recovery.to_attempt
            FROM recoveries AS recovery
            JOIN attempts AS old ON old.attempt_id=recovery.from_attempt
            JOIN attempts AS new ON new.attempt_id=recovery.to_attempt"""
        ).fetchall()
    controller_path = run_dir / "controller.jsonl"
    controller_events = (
        [json.loads(line) for line in controller_path.read_text().splitlines()]
        if controller_path.exists()
        else []
    )
    phases = []
    for old_finished, new_started, recomputed, from_attempt, to_attempt in rows:
        metrics["recomputed_steps"] += recomputed
        if old_finished:
            metrics["restart_seconds"] += max(
                0.0,
                (
                    datetime.fromisoformat(new_started) - datetime.fromisoformat(old_finished)
                ).total_seconds(),
            )
        observed = next(
            (
                event["time"]
                for event in controller_events
                if event["attempt_id"] == from_attempt and event["event_type"] == "fault_observed"
            ),
            None,
        )
        resumed = []
        for log in (run_dir / "attempts" / to_attempt).glob("rank-*.jsonl"):
            resumed.extend(
                event["time"]
                for event in map(json.loads, log.read_text().splitlines())
                if event.get("event_type") == "first_resumed_update"
            )
        config = (
            load_config(run_dir / "config.json") if (run_dir / "config.json").exists() else None
        )
        rto = None
        if observed and config and len(resumed) == config.run.world_size:
            rto = max(
                0.0,
                (
                    datetime.fromisoformat(max(resumed)) - datetime.fromisoformat(observed)
                ).total_seconds(),
            )
            metrics["recovery_rto_seconds"] += rto
        phases.append(
            {
                "from_attempt": from_attempt,
                "to_attempt": to_attempt,
                "fault_observed_at": observed,
                "all_ranks_first_update_at": max(resumed) if resumed else None,
                "rto_seconds": rto,
            }
        )
    metrics["attempt_gap_seconds"] = metrics["restart_seconds"]
    metrics["recovery_phases"] = phases
    return metrics


def _report_text(results: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint benchmark",
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
    lines.extend(
        [
            "",
            "## Worker training window",
            "",
            "| Mode | Median (s) | Range (s) | Runs |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
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
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                (
                    "| Mode | Repeat | Elapsed (s) | Training (s) | Checkpoints | Payload (MiB) | "
                    "Load 1m before / after | Valid |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['mode']} | {row['repeat']} | {row['elapsed_seconds']:.3f} | "
                f"{row['training_seconds']:.3f} | {row['checkpoint_count']} | "
                f"{row['checkpoint_bytes'] / 2**20:.3f} | "
                f"{row['load_average_before'][0]:.2f} / {row['load_average_after'][0]:.2f} | "
                f"{'yes' if row['validation_passed'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Measured save and recovery phases",
            "",
            (
                "| Mode | Repeat | Staging (s) | Upload (s) | Hash + commit (s) | "
                "Recovery RTO (s) | Recomputed steps |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results["raw_runs"]:
        lines.append(
            f"| {row['mode']} | {row['repeat']} | {row['staging_seconds']:.3f} | "
            f"{row.get('upload_seconds', 0):.3f} | {row['checksum_commit_seconds']:.3f} | "
            f"{row.get('recovery_rto_seconds', 0):.3f} | {row['recomputed_steps']} |"
        )
    lines.extend(
        [
            "",
            (
                "Each raw row points to its run directory in `results.json`. "
                "Validation uses exact hashes and effective sample IDs (atol=0, rtol=0). "
                "Mode order balances six permutations across repetitions. The exact base configuration is in "
                "`results.json`, and mode configurations are copied beside this report. "
                "Payload bytes sum committed manifest files across all checkpoints in a run, "
                "excluding the manifest and commit marker. Load averages record host activity "
                "over 1, 5 and 15 minutes before and after each run."
            ),
            "",
            "## Measurement limits",
            "",
            (
                "The legacy writing_seconds field is elapsed time after staging until "
                "the trainer observes completion at the rank barrier. It includes rank-state writes; "
                "for asynchronous save it also includes overlapping "
                "training and may exceed actual I/O time, so phase totals must not be added to wall time. "
                "Worker training time starts after process-group and model initialization and ends "
                "after the final training barrier; it includes checkpoint work but excludes launch. "
                "Each report applies only to its recorded device and local filesystem; it does not measure multi-node, "
                "storage-delay, disk-loss, or host-power-failure behavior. Short workloads and "
                "few repetitions cannot establish a general performance advantage. "
                "The host is not isolated; load snapshots do not control other workloads, "
                "filesystem cache, or thermal effects. Warm-up rounds prepare the host and cache; "
                "each measured run still launches new workers."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _persist(directory: Path, results: dict) -> None:
    results["raw_runs"] = [
        slot["row"]
        for slot in results["slots"]
        if slot["phase"] == "measured" and slot["status"] == "VALIDATED"
    ]
    results["warmup_runs"] = [
        slot["row"]
        for slot in results["slots"]
        if slot["phase"] == "warmup" and slot["status"] == "VALIDATED"
    ]
    results["summary"] = summarize_rows(results["raw_runs"])
    results["training_summary"] = summarize_rows(results["raw_runs"], "training_seconds")
    paired = []
    for repeat in range(1, results["repetitions"] + 1):
        batch = {row["mode"]: row for row in results["raw_runs"] if row["repeat"] == repeat}
        if len(batch) == 3:
            paired.append(
                {
                    "repeat": repeat,
                    "sync_minus_none_seconds": batch["sync"]["training_seconds"]
                    - batch["none"]["training_seconds"],
                    "async_minus_sync_seconds": batch["async"]["training_seconds"]
                    - batch["sync"]["training_seconds"],
                }
            )
    results["paired_differences"] = paired
    write_json_atomic(directory / "results.json", results)
    if results["status"] == "SUCCEEDED":
        (directory / "report.md").write_text(_report_text(results), encoding="utf-8")


def _original_measurement(run_dir: Path) -> dict | None:
    return trusted_measurement(run_dir)


def _execute_slots(directory: Path, results: dict) -> Path:
    reference = Path(results["reference_dir"]) if results.get("reference_dir") else None
    if reference is not None and not validate_runs(reference, reference)["passed"]:
        raise ValueError("benchmark reference evidence is invalid")
    results["status"] = "RUNNING"
    _persist(directory, results)
    for slot in results["slots"]:
        if slot["status"] == "VALIDATED":
            if not validate_runs(reference, Path(slot["row"]["run_dir"]))["passed"]:
                raise ValueError("completed benchmark evidence is invalid")
            continue
        mode, repeat, phase = slot["mode"], slot["repeat"], slot["phase"]
        root = directory / "runs" / f"{phase}-{repeat}-{mode}"
        slot["status"] = "RUNNING"
        slot["reason"] = None
        slot["started_at"] = utc_now()
        _persist(directory, results)
        try:
            config_copy = directory / f"{mode}-config.json"
            # A completed worker run may have outlived an interrupted experiment coordinator.
            candidates = sorted(
                root.glob("*/run.json"), key=lambda path: path.stat().st_mtime, reverse=True
            )
            reusable = (
                Path(slot["run_dir"]) if slot.get("run_dir")
                else candidates[0].parent if candidates else None
            )
            measurement = None
            if reusable is not None:
                if load_config(reusable / "config.json").fingerprint() != load_config(config_copy).fingerprint():
                    raise ValueError("benchmark slot configuration differs")
                run_dir, succeeded = reusable, resume(reusable)
                measurement = _original_measurement(run_dir) if succeeded else None
                if measurement is None:
                    slot.setdefault("history", []).append({
                        "reason": "original measurement unavailable; excluded from formal statistics"
                        if succeeded else "previous training failed; workers reconciled before retry",
                        "run_dir": str(run_dir), "time": utc_now(),
                    })
            if measurement is None:
                slot.pop("run_dir", None)
                _persist(directory, results)
                load_before = list(os.getloadavg())
                started = time.monotonic()
                run_dir, succeeded = run(config_copy, root)
                measurement = _original_measurement(run_dir) or {
                    "elapsed_seconds": time.monotonic() - started,
                    "load_average_before": load_before,
                    "load_average_after": list(os.getloadavg()),
                    "method": "experiment_monotonic",
                }
            slot["run_dir"] = str(run_dir)
            _persist(directory, results)
            if not succeeded:
                raise RuntimeError(f"benchmark run failed: {run_dir}")
            if reference is None:
                reference = run_dir
                results["reference_dir"] = str(reference)
            validation = validate_runs(reference, run_dir)
            if not validation["passed"]:
                raise RuntimeError(
                    f"benchmark correctness comparison failed: {validation['differences']}"
                )
            summary = json.loads((run_dir / "summary.json").read_text())
            rank_states = summary.get("rank_states", [])
            slot["row"] = {
                "mode": mode,
                "repeat": repeat,
                "phase": phase,
                "run_dir": str(run_dir),
                "elapsed_seconds": measurement["elapsed_seconds"],
                "elapsed_method": measurement["method"],
                "training_seconds": json.loads((run_dir / "summary.json").read_text())[
                    "training_elapsed_seconds"
                ],
                "rss_peak_bytes": max(
                    (item["rss_peak_bytes"] for item in rank_states), default=None
                ),
                "gpu_peak_allocated_bytes": max(
                    (
                        item["gpu_peak_allocated_bytes"]
                        for item in rank_states
                        if item.get("gpu_peak_allocated_bytes") is not None
                    ),
                    default=None,
                ),
                "rank_training_seconds": [item["training_seconds"] for item in rank_states],
                "load_average_before": measurement["load_average_before"],
                "load_average_after": measurement["load_average_after"],
                "validation_passed": validation["passed"],
                "validation_differences": validation["differences"],
                **_run_metrics(run_dir),
            }
            slot["status"] = "VALIDATED"
        except RunActiveError as exc:
            slot.update(status="RUNNING", reason=str(exc))
            results.update(status="BLOCKED", reason=str(exc))
            _persist(directory, results)
            raise
        except BaseException as exc:
            slot["status"] = "FAILED"
            slot["reason"] = f"{type(exc).__name__}: {exc}"
            slot.setdefault("history", []).append(
                {"reason": slot["reason"], "run_dir": slot.get("run_dir"), "time": utc_now()}
            )
            results["status"] = "FAILED"
            _persist(directory, results)
            raise
        _persist(directory, results)
    results["status"] = "SUCCEEDED"
    _persist(directory, results)
    return directory


def _execute_benchmark(directory: Path, results: dict) -> Path:
    try:
        return _execute_slots(directory, results)
    except RunActiveError:
        raise
    except BaseException as exc:
        results.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        _persist(directory, results)
        raise


def run_benchmark(
    config_path: Path, output_root: Path, repetitions: int = 3, warmups: int = 1
) -> Path:
    if repetitions < 3:
        raise ValueError("benchmark requires at least three repetitions per mode")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    base = load_config(config_path.resolve())
    directory = (output_root / f"benchmark-{uuid.uuid4().hex[:12]}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for mode in ("none", "sync", "async"):
        raw = base.model_dump()
        raw["checkpoint"]["mode"] = mode
        raw["fault"] = {"kind": "none", "step": None, "rank": 0}
        raw["recovery"]["omit_state"] = "none"
        write_json_atomic(
            directory / f"{mode}-config.json", ProjectConfig.model_validate(raw).model_dump()
        )
    slots = [
        {"phase": phase, "repeat": repeat, "mode": mode, "status": "PENDING"}
        for phase, count in (("warmup", warmups), ("measured", repetitions))
        for repeat in range(1, count + 1)
        for mode in mode_order(repeat)
    ]
    results = {
        "schema_version": 2,
        "created_at": utc_now(),
        "config": base.model_dump(),
        "repetitions": repetitions,
        "warmups": warmups,
        "workload_fingerprint": base.workload_fingerprint(),
        "environment": environment_snapshot(base.run.world_size, base.run.device, directory),
        "slots": slots,
        "status": "PENDING",
        "reference_dir": None,
    }
    with _controller_lock(directory):
        return _execute_benchmark(directory, results)


def resume_benchmark(directory: Path) -> Path:
    directory = directory.resolve()
    with _controller_lock(directory):
        results = json.loads((directory / "results.json").read_text())
        if results.get("schema_version") != 2:
            raise ValueError("benchmark schema is not resumable")
        base = ProjectConfig.model_validate(results["config"])
        current = environment_snapshot(base.run.world_size, base.run.device, directory)
        for field in (
            "source_sha256", "torch", "python", "versions", "installed_distributions",
            "environment_options", "startup_identity_sha256",
            "world_size", "device",
        ):
            if field not in results["environment"]:
                raise ValueError(f"benchmark runtime {field} identity is missing")
            if current[field] != results["environment"][field]:
                raise ValueError(f"benchmark source or runtime {field} differs")
        for mode in ("none", "sync", "async"):
            actual = load_config(directory / f"{mode}-config.json")
            expected = base.model_dump()
            expected["checkpoint"]["mode"] = mode
            expected["fault"] = {"kind": "none", "step": None, "rank": 0}
            expected["recovery"]["omit_state"] = "none"
            if actual.fingerprint() != ProjectConfig.model_validate(expected).fingerprint():
                raise ValueError("benchmark configuration differs")
        return _execute_benchmark(directory, results)
