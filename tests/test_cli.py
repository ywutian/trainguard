from pathlib import Path

import pytest
from typer.testing import CliRunner

from trainguard.cli import app
from trainguard.controller import ExperimentNotAuthorizedError
from trainguard.controller import run as launch_run


def test_recovery_and_experiment_commands_are_available() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "resume" in result.stdout
    assert "validate" in result.stdout
    assert "benchmark" in result.stdout


def test_benchmark_exposes_optional_warmup_rounds() -> None:
    result = CliRunner().invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--warmups" in result.stdout


def test_installed_template_can_be_written_once(tmp_path: Path) -> None:
    output = tmp_path / "demo.yaml"
    runner = CliRunner()
    first = runner.invoke(app, ["init-config", "--output", str(output)])
    assert first.exit_code == 0
    assert "total_steps: 4" in output.read_text()
    second = runner.invoke(app, ["init-config", "--output", str(output)])
    assert second.exit_code == 1


def test_cli_requires_explicit_fault_experiment_opt_in(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "recovery_demo.yaml"
    result = CliRunner().invoke(
        app, ["run", "--config", str(config), "--output-root", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "--allow-experiment" in result.output
    assert not list(tmp_path.iterdir())


def test_controller_entry_rejects_unauthorized_fault_before_creating_run(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs" / "recovery_demo.yaml"
    with pytest.raises(ExperimentNotAuthorizedError):
        launch_run(config, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
