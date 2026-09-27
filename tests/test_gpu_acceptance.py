import json
from pathlib import Path

import pytest
import torch

from trainguard.campaign import run_campaign
from trainguard.config import load_config


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two actual CUDA devices")
@pytest.mark.parametrize(
    "strategy,precision", [("ddp", "fp32"), ("ddp", "bf16"), ("ddp", "fp16"), ("fsdp2", "fp32")]
)
def test_actual_cuda_recovery_campaign(tmp_path, strategy, precision):
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["run"].update(device="cuda", backend="nccl", strategy=strategy)
    raw["training"]["precision"] = precision
    raw["recovery"]["progress_timeout_seconds"] = 30
    path = tmp_path / "cuda.json"
    path.write_text(json.dumps(raw))
    directory = run_campaign(path, tmp_path / "runs")
    result = json.loads((directory / "acceptance.json").read_text())
    assert result["status"] == "SUCCEEDED", result
