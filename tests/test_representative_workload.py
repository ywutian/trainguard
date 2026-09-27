"""Representative byte-level GPT workload: exact recovery and omitted-state controls."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from trainguard.config import load_config
from trainguard.controller import resume, run
from trainguard.validation import validate_runs

ROOT = Path(__file__).parents[1]


def _copy_workload(tmp_path: Path) -> Path:
    """Run from a private copy so tamper checks never touch the repository files."""
    shutil.copytree(ROOT / "examples/charlm", tmp_path / "examples/charlm")
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs/charlm_cpu.yaml"
    shutil.copyfile(ROOT / "configs/charlm_cpu.yaml", config)
    return config


def _losses(run_dir: Path) -> list[float]:
    events = [
        json.loads(line)
        for line in (run_dir / "attempts/attempt-001/rank-0.jsonl").read_text().splitlines()
    ]
    return [event["loss"] for event in events if event["event_type"] == "step_completed"]


def test_representative_workload_recovers_exactly_and_rejects_missing_state(
    tmp_path: Path,
) -> None:
    config = _copy_workload(tmp_path)
    reference, passed = run(config, tmp_path / "runs")
    assert passed, (reference / "launcher.log").read_text()
    losses = _losses(reference)
    assert len(losses) == 6 and losses[-1] < losses[0]

    raw = load_config(config).model_dump()
    raw["checkpoint"].update(mode="sync", interval_steps=1)
    raw["fault"].update(kind="worker_exit", step=3, rank=0)
    faulted = tmp_path / "faulted.json"
    faulted.write_text(json.dumps(raw))
    recovered, passed = run(faulted, tmp_path / "runs", allow_experiment=True)
    assert passed, (recovered / "launcher.log").read_text()
    exact = validate_runs(reference, recovered)
    assert exact["passed"], exact

    outcomes = {}
    for omitted in ("rng", "optimizer", "stream", "extra"):
        raw["recovery"] = {**raw["recovery"], "omit_state": omitted}
        faulted.write_text(json.dumps(raw))
        negative, passed = run(faulted, tmp_path / "runs", allow_experiment=True)
        # An omission must never produce an exact match: it either fails closed
        # during the run or diverges from the reference.
        outcomes[omitted] = (
            "diverged" if passed and not validate_runs(reference, negative)["passed"]
            else "failed_closed" if not passed else "MATCHED"
        )
    assert "MATCHED" not in outcomes.values(), outcomes
    assert outcomes["optimizer"] == "failed_closed", outcomes

    corpus = tmp_path / "examples/charlm/pride_and_prejudice.txt"
    corpus.write_text(corpus.read_text() + "\nAn added line.\n")
    with pytest.raises(ValueError, match="data corpus SHA-256"):
        resume(recovered)
