import json
from pathlib import Path

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

    for run_dir in (first_dir, second_dir):
        rank0 = _events(run_dir / "attempts" / "attempt-001" / "rank-0.jsonl")
        rank1 = _events(run_dir / "attempts" / "attempt-001" / "rank-1.jsonl")
        steps0 = [event for event in rank0 if event["event_type"] == "step_completed"]
        steps1 = [event for event in rank1 if event["event_type"] == "step_completed"]
        assert [event["global_step"] for event in steps0] == [1, 2, 3, 4]
        assert [event["global_step"] for event in steps1] == [1, 2, 3, 4]
        for left, right in zip(steps0, steps1, strict=True):
            assert set(left["sample_ids"]).isdisjoint(right["sample_ids"])
