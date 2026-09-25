import json
from pathlib import Path

from trainguard.checkpoint import latest_valid_checkpoint
from trainguard.config import ProjectConfig, load_config
from trainguard.controller import run


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_two_rank_reference_run_is_repeatable(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    first_dir, first_ok = run(config, tmp_path)
    second_dir, second_ok = run(config, tmp_path)
    assert first_ok and second_ok

    first = json.loads((first_dir / "summary.json").read_text())
    second = json.loads((second_dir / "summary.json").read_text())
    assert first["global_step"] == second["global_step"] == 4
    assert first["world_size"] == second["world_size"] == 2
    assert first["model_sha256"] == second["model_sha256"]
    assert first["config_fingerprint"] == second["config_fingerprint"]
    assert first["training_elapsed_seconds"] > 0
    assert second["training_elapsed_seconds"] > 0

    for run_dir in (first_dir, second_dir):
        rank0 = _events(run_dir / "attempts" / "attempt-001" / "rank-0.jsonl")
        rank1 = _events(run_dir / "attempts" / "attempt-001" / "rank-1.jsonl")
        steps0 = [event for event in rank0 if event["event_type"] == "step_completed"]
        steps1 = [event for event in rank1 if event["event_type"] == "step_completed"]
        assert [event["global_step"] for event in steps0] == [1, 2, 3, 4]
        assert [event["global_step"] for event in steps1] == [1, 2, 3, 4]
        for left, right in zip(steps0, steps1, strict=True):
            assert set(left["sample_ids"]).isdisjoint(right["sample_ids"])


def test_sync_dcp_checkpoint_is_committed_and_loadable(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    raw = load_config(source).model_dump()
    raw["checkpoint"] = {"mode": "sync", "interval_steps": 2}
    config = ProjectConfig.model_validate(raw)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    run_dir, succeeded = run(config_path, tmp_path / "runs")
    assert succeeded, (run_dir / "launcher.log").read_text()
    selected = latest_valid_checkpoint(run_dir, config, json.loads((run_dir / "run.json").read_text())["run_id"])
    assert selected is not None
    assert selected.global_step == 4
    assert (selected.path / "dcp" / ".metadata").is_file()
    assert len(list((run_dir / "checkpoints").iterdir())) == 2
