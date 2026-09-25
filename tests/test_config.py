from pathlib import Path

import pytest
from pydantic import ValidationError

from trainguard.config import ProjectConfig, load_config


def test_demo_config_loads_with_stable_fingerprint() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    assert config.run.world_size == 2
    assert config.fingerprint() == config.fingerprint()
    assert len(config.fingerprint()) == 64


def test_unsupported_gpu_and_unknown_fields_are_rejected() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["run"]["device"] = "cuda"
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)
    raw["run"]["device"] = "cpu"
    raw["checkpoint"] = {"mode": "sync"}
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_incompatible_attention_dimensions_are_rejected() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["model"]["hidden_size"] = 63
    with pytest.raises(ValidationError, match="divisible"):
        ProjectConfig.model_validate(raw)
