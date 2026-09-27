import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from trainguard.config import ProjectConfig, load_config
from trainguard.controller import resume, run
from trainguard.external_workload import frozen_workload_path
from trainguard.validation import validate_runs

EXAMPLE = Path(__file__).parents[1] / "examples" / "external_cpu_workload.py"
DEMO = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"


def _external_config(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "customer_workload.py"
    shutil.copyfile(EXAMPLE, source)
    raw = load_config(DEMO).model_dump()
    raw["model"]["dropout"] = 0.4
    raw["external_workload"] = {
        "version": 1,
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    return config, source


def test_external_workload_is_cpu_ddp_only_and_bound_to_its_bytes(tmp_path: Path) -> None:
    config_path, source = _external_config(tmp_path)
    raw = json.loads(config_path.read_text())
    for field, value in (("device", "cuda"), ("strategy", "fsdp2")):
        altered = json.loads(config_path.read_text())
        altered["run"][field] = value
        with pytest.raises(ValidationError, match="external workload v1 requires CPU DDP"):
            ProjectConfig.model_validate(altered)
    source.write_bytes(source.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="SHA-256"):
        run(config_path, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
    assert raw["external_workload"]["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()


def test_external_workload_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "workload.fifo"
    os.mkfifo(fifo)
    command = (
        "from pathlib import Path; from types import SimpleNamespace; "
        "from trainguard.external_workload import read_verified_source; "
        "settings = SimpleNamespace(path=__import__('sys').argv[1], sha256='0' * 64); "
        "config = SimpleNamespace(external_workload=settings); "
        "read_verified_source(config)"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(fifo)],
        capture_output=True, text=True, timeout=5, check=False,
    )
    assert result.returncode != 0
    assert "external workload must be a regular file" in result.stderr


def test_preflight_freezes_the_verified_bytes_before_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trainguard import controller

    config_path, source = _external_config(tmp_path)
    original = source.read_bytes()
    actual_preflight = controller.preflight

    def change_source_after_preflight(config):
        approved = actual_preflight(config)
        source.write_bytes(b"unapproved replacement")
        return approved

    monkeypatch.setattr(controller, "preflight", change_source_after_preflight)
    run_dir, succeeded = controller.run(config_path, tmp_path / "runs")
    assert succeeded, (run_dir / "launcher.log").read_text()
    assert frozen_workload_path(run_dir).read_bytes() == original
    monkeypatch.setattr(controller, "preflight", actual_preflight)
    assert controller.resume(run_dir)


def test_external_two_rank_recovery_and_omitted_state_controls(tmp_path: Path) -> None:
    config_path, source = _external_config(tmp_path)
    reference, succeeded = run(config_path, tmp_path / "runs")
    assert succeeded, (reference / "launcher.log").read_text()
    assert frozen_workload_path(reference).read_bytes() == source.read_bytes()
    events = [
        json.loads(line)
        for line in (reference / "attempts/attempt-001/rank-0.jsonl").read_text().splitlines()
    ]
    sample_lengths = [
        len(event["sample_ids"]) for event in events
        if event["event_type"] == "step_completed"
    ]
    assert sample_lengths == [2, 1, 2, 1]

    raw = json.loads(config_path.read_text())
    raw["checkpoint"] = {"mode": "sync", "interval_steps": 1}
    raw["fault"] = {"kind": "worker_exit", "step": 3, "rank": 0}
    config_path.write_text(json.dumps(raw))
    recovered, succeeded = run(config_path, tmp_path / "runs", allow_experiment=True)
    assert succeeded, (recovered / "launcher.log").read_text()
    exact = validate_runs(reference, recovered)
    assert exact["passed"], exact

    for omitted in ("rng", "optimizer", "cursor"):
        raw["recovery"] = {"omit_state": omitted}
        config_path.write_text(json.dumps(raw))
        negative, succeeded = run(config_path, tmp_path / "runs", allow_experiment=True)
        assert succeeded, (negative / "launcher.log").read_text()
        comparison = validate_runs(reference, negative)
        assert not comparison["passed"], (omitted, comparison)
        assert "final model_sha256 differs" in comparison["differences"]
        if omitted == "cursor":
            assert any("sample sequence" in item for item in comparison["differences"])

    source.unlink()
    assert resume(recovered)
    frozen_workload_path(recovered).write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA-256"):
        resume(recovered)
    assert not validate_runs(reference, recovered)["passed"]
