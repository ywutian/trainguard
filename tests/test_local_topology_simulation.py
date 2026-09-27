"""Two local launch agents exercise the distributed trainer across a restart."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from trainguard.checkpoint import latest_valid_checkpoint
from trainguard.config import ProjectConfig, load_config
from trainguard.controller import run


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _launch_agents(
    root: Path, config: Path, run_id: str, attempt_id: str, checkpoint: Path | None,
    checkpoint_sha256: str | None = None,
) -> list[int]:
    (root / "attempts" / attempt_id).mkdir(parents=True)
    port = _free_port()
    processes = []
    streams = []
    try:
        for node_rank in range(2):
            command = [
                sys.executable, "-m", "torch.distributed.run", "--nnodes=2",
                "--nproc-per-node=1", f"--node-rank={node_rank}",
                "--rdzv-backend=static", "--master-addr=127.0.0.1",
                f"--master-port={port}", "--max-restarts=0",
                "-m", "trainguard.trainer", "--config", str(config),
                "--run-dir", str(root), "--run-id", run_id, "--attempt-id", attempt_id,
            ]
            if checkpoint is not None:
                assert checkpoint_sha256 is not None
                command.extend([
                    "--resume-checkpoint", str(checkpoint),
                    "--expected-checkpoint-sha256", checkpoint_sha256,
                ])
            stream = (root / f"{attempt_id}-node-{node_rank}.log").open("w")
            streams.append(stream)
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            codes = [process.poll() for process in processes]
            if all(code is not None for code in codes):
                return [int(code) for code in codes]
            if any(code not in (None, 0) for code in codes):
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                return [process.wait(timeout=5) for process in processes]
            time.sleep(0.1)
        raise TimeoutError(f"two launch agents did not finish: {[p.poll() for p in processes]}")
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
        for stream in streams:
            stream.close()


def _step_evidence(path: Path) -> list[tuple[int, int, tuple[int, ...]]]:
    return [
        (record["global_step"], record["consumed_batches"], tuple(record["sample_ids"]))
        for line in path.read_text().splitlines()
        if (record := json.loads(line)).get("event_type") == "step_completed"
    ]


def test_two_local_launch_agents_recover_exactly_after_worker_exit(tmp_path: Path) -> None:
    base = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml")
    raw = base.model_dump()
    raw["model"]["dropout"] = 0.2
    raw["checkpoint"] = {"mode": "sync", "interval_steps": 1}
    raw["fault"] = {"kind": "worker_exit", "step": 3, "rank": 0}
    config = ProjectConfig.model_validate(raw)
    run_dir = tmp_path / "two-agent"
    run_dir.mkdir()
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config.model_dump()))

    reference_raw = config.model_dump()
    reference_raw["checkpoint"]["mode"] = "none"
    reference_raw["fault"] = {"kind": "none", "step": None, "rank": 0}
    reference_config = tmp_path / "reference.json"
    reference_config.write_text(json.dumps(reference_raw))
    reference, reference_ok = run(reference_config, tmp_path / "reference-runs")
    assert reference_ok, (reference / "launcher.log").read_text()

    run_id = "two-agent-simulation"
    first = _launch_agents(run_dir, config_path, run_id, "attempt-001", None)
    assert any(code != 0 for code in first), first
    selected = latest_valid_checkpoint(run_dir, config, run_id)
    assert selected is not None and selected.global_step == 2
    second = _launch_agents(
        run_dir, config_path, run_id, "attempt-002", selected.path,
        selected.manifest_sha256,
    )
    assert second == [0, 0], [
        (run_dir / f"attempt-002-node-{rank}.log").read_text() for rank in range(2)
    ]

    expected = json.loads((reference / "summary.json").read_text())
    actual = json.loads((run_dir / "attempts/attempt-002/summary.json").read_text())
    for field in (
        "model_sha256", "optimizer_sha256", "scheduler_sha256", "scaler_sha256",
        "global_step", "consumed_batches",
    ):
        assert actual[field] == expected[field], field
    for rank in range(2):
        original = _step_evidence(reference / f"attempts/attempt-001/rank-{rank}.jsonl")
        before = _step_evidence(run_dir / f"attempts/attempt-001/rank-{rank}.jsonl")
        after = _step_evidence(run_dir / f"attempts/attempt-002/rank-{rank}.jsonl")
        effective = [row for row in before if row[0] <= selected.global_step] + after
        assert effective == original
        loads = [
            json.loads(line)
            for line in (run_dir / f"attempts/attempt-002/rank-{rank}.jsonl").read_text().splitlines()
            if json.loads(line).get("event_type") == "state_loaded"
        ]
        assert len(loads) == 1 and loads[0]["global_step"] == selected.global_step
