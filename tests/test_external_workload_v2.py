import hashlib
import json
import shutil
from pathlib import Path

import pytest

from trainguard.checkpoint import CheckpointInvalid, _check_rank_states
from trainguard.config import load_config
from trainguard.controller import resume, run
from trainguard.external_workload import frozen_data_path, frozen_dependency_path
from trainguard.validation import validate_runs

ROOT = Path(__file__).parents[1]


def _configured_workload(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "customer_workload.py"
    helper = tmp_path / "external_v2_helper.py"
    data = tmp_path / "training.json"
    shutil.copyfile(ROOT / "examples/external_cpu_workload_v2.py", source)
    shutil.copyfile(ROOT / "examples/external_v2_helper.py", helper)
    shutil.copyfile(ROOT / "examples/external_v2_data.json", data)
    raw = load_config(ROOT / "configs/cpu_demo.yaml").model_dump()
    raw["model"]["dropout"] = 0.35
    raw["external_workload"] = {
        "version": 2,
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependencies": [{
            "module": "external_v2_helper",
            "path": str(helper),
            "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        }],
        "data_files": [{
            "name": "training",
            "path": str(data),
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        }],
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    return config, helper, data


def test_v2_rejects_changed_helper_and_data_before_run(tmp_path: Path) -> None:
    config, helper, data = _configured_workload(tmp_path)
    helper.write_text(helper.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="dependency external_v2_helper SHA-256"):
        run(config, tmp_path / "runs")
    shutil.copyfile(ROOT / "examples/external_v2_helper.py", helper)
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="data training SHA-256"):
        run(config, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_v2_rejects_undeclared_direct_import(tmp_path: Path) -> None:
    config, _, _ = _configured_workload(tmp_path)
    raw = json.loads(config.read_text())
    source = Path(raw["external_workload"]["path"])
    source.write_text(source.read_text().replace("external_v2_helper", "unbound_helper"))
    raw["external_workload"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="undeclared import"):
        run(config, tmp_path / "runs")


def test_v2_restores_rng_after_external_constructors_and_load_hooks(tmp_path: Path) -> None:
    config, _, _ = _configured_workload(tmp_path)
    raw = json.loads(config.read_text())
    source = Path(raw["external_workload"]["path"])
    content = source.read_text()
    content = content.replace(
        "WORKLOAD_API_VERSION = 2\n",
        "WORKLOAD_API_VERSION = 2\n\n"
        "def consume_process_rng():\n"
        "    import random\n"
        "    import numpy as np\n"
        "    random.random()\n"
        "    np.random.random()\n"
        "    torch.rand(())\n",
    )
    for before, after in (
        ("        self.config = config\n", "        consume_process_rng()\n        self.config = config\n"),
        (
            '        self.rng_state = value["rng_state"]\n',
            '        consume_process_rng()\n        self.rng_state = value["rng_state"]\n',
        ),
        ("        self.calls = 0\n", "        consume_process_rng()\n        self.calls = 0\n"),
        (
            '        self.calls = value["calls"]\n',
            '        consume_process_rng()\n        self.calls = value["calls"]\n',
        ),
    ):
        assert before in content
        content = content.replace(before, after, 1)
    source.write_text(content)
    raw["external_workload"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    config.write_text(json.dumps(raw))

    reference, passed = run(config, tmp_path / "runs")
    assert passed, (reference / "launcher.log").read_text()
    raw["checkpoint"] = {"mode": "sync", "interval_steps": 1}
    raw["fault"] = {"kind": "worker_exit", "step": 3, "rank": 0}
    config.write_text(json.dumps(raw))
    recovered, passed = run(config, tmp_path / "runs", allow_experiment=True)
    assert passed, (recovered / "launcher.log").read_text()
    assert validate_runs(reference, recovered)["passed"]
    events = [
        json.loads(line)
        for line in (recovered / "attempts/attempt-002/rank-0.jsonl").read_text().splitlines()
    ]
    types = [event["event_type"] for event in events]
    assert types.index("state_loaded") < types.index("batch_consumed")

    raw["recovery"] = {"omit_state": "rng"}
    config.write_text(json.dumps(raw))
    negative, passed = run(config, tmp_path / "runs", allow_experiment=True)
    assert passed, (negative / "launcher.log").read_text()
    comparison = validate_runs(reference, negative)
    assert not comparison["passed"]
    assert "final model_sha256 differs" in comparison["differences"]


def test_v2_two_rank_exact_recovery_and_missing_state_controls(tmp_path: Path) -> None:
    config, helper, data = _configured_workload(tmp_path)
    reference, passed = run(config, tmp_path / "runs")
    assert passed, (reference / "launcher.log").read_text()
    assert frozen_dependency_path(reference, "external_v2_helper").read_bytes() == helper.read_bytes()
    assert frozen_data_path(reference, "training").read_bytes() == data.read_bytes()

    raw = json.loads(config.read_text())
    raw["checkpoint"] = {"mode": "sync", "interval_steps": 1}
    raw["fault"] = {"kind": "worker_exit", "step": 3, "rank": 0}
    config.write_text(json.dumps(raw))
    recovered, passed = run(config, tmp_path / "runs", allow_experiment=True)
    assert passed, (recovered / "launcher.log").read_text()
    exact = validate_runs(reference, recovered)
    assert exact["passed"], exact
    restored = [
        json.loads(line) for line in (recovered / "attempts/attempt-002/rank-0.jsonl").read_text().splitlines()
    ]
    assert any(event["event_type"] == "state_loaded" for event in restored)

    selected = recovered / "checkpoints/step-000002-attempt-001"
    rank_file = selected / "rank-0.json"
    original_rank_state = rank_file.read_bytes()
    missing = json.loads(original_rank_state)
    missing["external_state"].pop("extra")
    rank_file.write_text(json.dumps(missing))
    try:
        with pytest.raises(CheckpointInvalid, match="external stream or extra state"):
            _check_rank_states(
                selected, load_config(recovered / "config.json"),
                json.loads((recovered / "run.json").read_text())["run_id"],
                "attempt-001", 2, require_trainable_state=True,
            )
    finally:
        rank_file.write_bytes(original_rank_state)

    for omitted, field in (
        ("stream", "stream_sha256"),
        ("extra", "extra_sha256"),
    ):
        raw["recovery"] = {"omit_state": omitted}
        config.write_text(json.dumps(raw))
        negative, passed = run(config, tmp_path / "runs", allow_experiment=True)
        assert passed, (negative / "launcher.log").read_text()
        comparison = validate_runs(reference, negative)
        assert not comparison["passed"], (omitted, comparison)
        assert f"final {field} differs" in comparison["differences"]

    helper.write_text(helper.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="dependency external_v2_helper SHA-256"):
        resume(recovered)
    assert not validate_runs(reference, recovered)["passed"]
    shutil.copyfile(ROOT / "examples/external_v2_helper.py", helper)
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="data training SHA-256"):
        resume(recovered)
    assert not validate_runs(reference, recovered)["passed"]
