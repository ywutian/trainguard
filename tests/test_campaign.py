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
