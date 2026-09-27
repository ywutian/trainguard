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
    raw["unexpected"] = {"mode": "sync"}
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_incompatible_attention_dimensions_are_rejected() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["model"]["hidden_size"] = 63
    with pytest.raises(ValidationError, match="divisible"):
        ProjectConfig.model_validate(raw)


def test_recovery_configuration_and_workload_fingerprint() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["checkpoint"] = {"mode": "async", "interval_steps": 2}
    raw["recovery"] = {"max_restarts": 2, "progress_timeout_seconds": 15}
    configured = ProjectConfig.model_validate(raw)
    assert configured.checkpoint.mode == "async"
    assert configured.workload_fingerprint() == valid.workload_fingerprint()
    assert configured.fingerprint() != valid.fingerprint()


def test_fault_step_must_have_a_fault_kind() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["fault"] = {"kind": "none", "step": 2}
    with pytest.raises(ValidationError, match="fault step"):
        ProjectConfig.model_validate(raw)


def test_save_fault_must_target_a_checkpoint_boundary() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["fault"] = {"kind": "save_interrupt", "step": 1, "rank": 0}
    with pytest.raises(ValidationError, match="checkpoint boundary"):
        ProjectConfig.model_validate(raw)
    raw["checkpoint"] = {"mode": "sync", "interval_steps": 2}
    with pytest.raises(ValidationError, match="checkpoint boundary"):
        ProjectConfig.model_validate(raw)


def test_retention_budget_requires_enabled_retention() -> None:
    valid = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = valid.model_dump()
    raw["checkpoint"]["max_retained_bytes"] = 1024
    with pytest.raises(ValidationError, match="keep_last_k"):
        ProjectConfig.model_validate(raw)


@pytest.mark.parametrize("unsafe", ["fault", "omit", "no_checkpoint"])
def test_guarded_profile_rejects_experiment_controls(unsafe: str) -> None:
    raw = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml").model_dump()
    raw["run"]["profile"] = "guarded"
    raw["checkpoint"]["mode"] = "sync"
    if unsafe == "fault":
        raw["fault"] = {"kind": "worker_exit", "step": 2}
    elif unsafe == "omit":
        raw["recovery"]["omit_state"] = "rng"
    else:
        raw["checkpoint"]["mode"] = "none"
    with pytest.raises(ValidationError, match="guarded runs"):
        ProjectConfig.model_validate(raw)


def test_guarded_profile_accepts_complete_non_fault_run() -> None:
    raw = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml").model_dump()
    raw["run"]["profile"] = "guarded"
    raw["checkpoint"]["mode"] = "sync"
    assert ProjectConfig.model_validate(raw).run.profile == "guarded"


@pytest.mark.parametrize("duplicate", ["top_level", "nested"])
def test_duplicate_yaml_configuration_keys_are_rejected(tmp_path: Path, duplicate: str) -> None:
    source = (Path(__file__).parents[1] / "configs/cpu_demo.yaml").read_text()
    if duplicate == "top_level":
        source += "\nrun:\n  profile: experiment\n"
    else:
        source = source.replace("  seed: 42", "  seed: 42\n  seed: 7")
    path = tmp_path / "duplicate.yaml"
    path.write_text(source)
    with pytest.raises(ValueError, match="duplicate YAML configuration key"):
        load_config(path)


def test_worker_rejects_config_that_differs_from_controller_fingerprint(tmp_path: Path) -> None:
    from trainguard.trainer import train

    with pytest.raises(ValueError, match="controller-approved"):
        train(
            Path(__file__).parents[1] / "configs/cpu_demo.yaml",
            tmp_path, "run", "attempt-001", expected_config_fingerprint="0" * 64,
        )
