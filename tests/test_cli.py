from typer.testing import CliRunner

from trainguard.cli import app


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
