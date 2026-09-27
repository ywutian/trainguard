import json
import shutil
from pathlib import Path

import pytest
from typer.main import get_command
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
    command = get_command(app).commands["benchmark"]
    assert any("--warmups" in parameter.opts for parameter in command.params)


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


def test_validate_cli_requires_an_independent_reference_run(tmp_path: Path) -> None:
    reference_config = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    recovery_config = Path(__file__).parents[1] / "configs/recovery_demo.yaml"
    reference, reference_ok = launch_run(reference_config, tmp_path / "reference")
    recovered, recovered_ok = launch_run(
        recovery_config, tmp_path / "recovered", allow_experiment=True
    )
    assert reference_ok and recovered_ok
    runner = CliRunner()

    distinct_report = tmp_path / "distinct.json"
    distinct = runner.invoke(app, [
        "validate", "--reference", str(reference), "--recovered", str(recovered),
        "--report", str(distinct_report),
    ])
    assert distinct.exit_code == 0, distinct.output
    assert "Recovery matches reference" in distinct.output
    independent = json.loads(distinct_report.read_text())
    assert independent["passed"] is True
    assert independent["comparison_kind"] == "INDEPENDENT_REFERENCE"
    assert independent["independent_reference"] is True

    for name, duplicate in (
        ("same-path", recovered),
        ("copied-run", tmp_path / "copied-run"),
    ):
        if name == "copied-run":
            shutil.copytree(recovered, duplicate)
        report = tmp_path / f"{name}.json"
        result = runner.invoke(app, [
            "validate", "--reference", str(recovered), "--recovered", str(duplicate),
            "--report", str(report),
        ])
        assert result.exit_code == 2, result.output
        assert "Recovery matches reference" not in result.output
        assert "independent reference run is required" in result.output.lower()
        self_check = json.loads(report.read_text())
        assert self_check["passed"] is True
        assert self_check["comparison_kind"] == "SELF_CHECK"
        assert self_check["independent_reference"] is False
