from typer.testing import CliRunner

from trainguard.cli import app


def test_recovery_and_experiment_commands_are_available() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "resume" in result.stdout
    assert "validate" in result.stdout
    assert "benchmark" in result.stdout
