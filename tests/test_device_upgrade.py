from pathlib import Path

import pytest
import torch

from trainguard.config import ProjectConfig, load_config
from trainguard.controller import run


def test_cuda_configuration_is_accepted_but_missing_devices_fail_before_launch(tmp_path):
    raw = load_config(Path(__file__).parents[1] / 'configs/cpu_demo.yaml').model_dump()
    raw['run'].update(device='cuda', backend='nccl')
    config = ProjectConfig.model_validate(raw)
    if torch.cuda.device_count() >= config.run.world_size:
        pytest.skip('this test verifies the unavailable-device preflight')
    path = tmp_path / 'cuda.json'
    path.write_text(config.model_dump_json())
    with pytest.raises(RuntimeError, match='CUDA'):
        run(path, tmp_path / 'runs')
    assert not (tmp_path / 'runs').exists()


def test_fsdp2_requires_cuda():
    raw = load_config(Path(__file__).parents[1] / 'configs/cpu_demo.yaml').model_dump()
    raw['run']['strategy'] = 'fsdp2'
    with pytest.raises(ValueError, match='CUDA'):
        ProjectConfig.model_validate(raw)
