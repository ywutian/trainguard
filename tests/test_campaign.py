import json
from pathlib import Path

import pytest


def test_cpu_campaign_closes_recovery_matrix(tmp_path):
    from trainguard import campaign

    source = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    directory = campaign.run_campaign(source, tmp_path)
    result = json.loads((directory / "acceptance.json").read_text())
    assert result["status"] == "SUCCEEDED"
    assert len(result["cases"]) == 10
    assert all(case["status"] == "PASSED" for case in result["cases"])
    assert all(case["recovery_count"] >= 1 for case in result["cases"])
    campaign.resume_campaign(directory)
    again = json.loads((directory / "acceptance.json").read_text())
    assert [(row["name"], row["run_dir"]) for row in result["cases"]] == [
        (row["name"], row["run_dir"]) for row in again["cases"]
    ]
    negative = next(case for case in result["cases"] if case["name"] == "omit-rng")
    (Path(negative["run_dir"]) / "summary.json").unlink()
    with pytest.raises(ValueError, match="evidence"):
        campaign.resume_campaign(directory)
    assert json.loads((directory / "acceptance.json").read_text())["status"] == "FAILED"


def test_failed_reference_can_be_retried_without_discarding_history(tmp_path, monkeypatch):
    from trainguard import campaign

    source = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    real_run = campaign.run
    launches = []

    def failed_once(config, root, *, allow_experiment=False):
        launches.append(root)
        if len(launches) == 1:
            directory = root / "failed-reference"
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(config.read_text())
            (directory / "run.json").write_text(json.dumps({"status": "FAILED"}))
            return directory, False
        return real_run(config, root, allow_experiment=allow_experiment)

    monkeypatch.setattr(campaign, "run", failed_once)
    monkeypatch.setattr(campaign, "resume", lambda path: False)
    directory = campaign.run_campaign(source, tmp_path)
    result = json.loads((directory / "acceptance.json").read_text())
    assert result["status"] == "FAILED"
    # Isolate the reference lifecycle; the full matrix is exercised above.
    result["cases"] = []
    (directory / "acceptance.json").write_text(json.dumps(result))
    campaign.resume_campaign(directory)
    final = json.loads((directory / "acceptance.json").read_text())
    assert final["status"] == "SUCCEEDED"
    assert len(launches) == 2
    assert final["reference_history"][0]["run_dir"] == result["reference_dir"]
    assert Path(result["reference_dir"]).is_dir()


def test_invalid_dataset_is_rejected_before_launch(tmp_path):
    import hashlib

    from trainguard.config import load_config
    from trainguard.controller import run

    source = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    raw = load_config(source).model_dump()
    dataset = tmp_path / "tokens.jsonl"
    dataset.write_text("{}\n")
    raw["data"] = {
        "kind": "jsonl",
        "path": str(dataset),
        "sha256": hashlib.sha256(b"other").hexdigest(),
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="SHA-256"):
        run(path, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_case_retry_waits_for_old_worker_ownership_check(tmp_path, monkeypatch):
    from trainguard import campaign
    from trainguard.controller import RunActiveError

    old_run = tmp_path / "old"
    old_run.mkdir()
    config = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    from trainguard.config import load_config
    contents = json.dumps(load_config(config).model_dump())
    config_copy = tmp_path / "config.json"
    config_copy.write_text(contents)
    (old_run / "config.json").write_text(contents)
    (old_run / "run.json").write_text(json.dumps({"status": "FAILED"}))
    owner = {"run_dir": str(old_run)}
    monkeypatch.setattr(campaign, "_persist", lambda *args: None)
    launches = []
    monkeypatch.setattr(campaign, "run", lambda *args, **kwargs: launches.append((args, kwargs)))

    def active(path):
        raise RunActiveError("worker group remains active")

    monkeypatch.setattr(campaign, "resume", active)
    for _ in range(2):
        with pytest.raises(RunActiveError):
            campaign._reconcile_or_retry(tmp_path, {}, owner, config_copy, tmp_path / "runs")
    assert launches == []
    monkeypatch.setattr(campaign, "resume", lambda path: False)
    campaign._reconcile_or_retry(tmp_path, {}, owner, config_copy, tmp_path / "runs")
    assert len(launches) == 1
    assert owner["history"][0]["run_dir"] == str(old_run)
    assert "run_dir" not in owner
