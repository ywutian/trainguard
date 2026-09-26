"""Command-line entry points for training, recovery, and experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trainguard import __version__
from trainguard.benchmark import run_benchmark
from trainguard.config import load_config
from trainguard.controller import RunActiveError
from trainguard.controller import resume as resume_run
from trainguard.controller import run as launch_run
from trainguard.events import write_json_atomic
from trainguard.validation import validate_runs

app = typer.Typer(help="Distributed training correctness experiments.")
DEFAULT_OUTPUT_ROOT = Path("runs")


@app.command("version")
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("validate-config")
def validate_config(
    config: Annotated[
        Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)
    ],
) -> None:
    """Check a CPU training configuration and print its fingerprint."""
    settings = load_config(config)
    typer.echo(settings.fingerprint())


@app.command("run")
def run(
    config: Annotated[
        Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)
    ],
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Run a fixed-size CPU DDP workload with bounded recovery."""
    run_dir, succeeded = launch_run(config, output_root)
    typer.echo(f"Run directory: {run_dir}")
    if not succeeded:
        typer.echo(f"Training failed; inspect {run_dir / 'launcher.log'}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Training completed; summary: {run_dir / 'summary.json'}")


@app.command("resume")
def resume(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
) -> None:
    """Resume a stopped run from its newest valid committed checkpoint."""
    try:
        succeeded = resume_run(run_dir)
    except RunActiveError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if not succeeded:
        typer.echo(f"Recovery failed; inspect {run_dir / 'run.json'}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Training completed; summary: {run_dir / 'summary.json'}")


@app.command("validate")
def validate(
    reference: Annotated[Path, typer.Option("--reference", exists=True, file_okay=False)],
    recovered: Annotated[Path, typer.Option("--recovered", exists=True, file_okay=False)],
    report: Annotated[Path | None, typer.Option("--report")] = None,
) -> None:
    """Compare final state and effective samples with an uninterrupted run."""
    result = validate_runs(reference, recovered)
    destination = report or recovered / "validation.json"
    write_json_atomic(destination, result)
    typer.echo(f"Validation report: {destination}")
    if not result["passed"]:
        typer.echo("Recovery differs from reference", err=True)
        raise typer.Exit(1)
    typer.echo("Recovery matches reference")


@app.command("benchmark")
def benchmark(
    config: Annotated[
        Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)
    ],
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    repetitions: Annotated[int, typer.Option("--repetitions", min=3)] = 3,
    warmups: Annotated[int, typer.Option("--warmups", min=0)] = 1,
) -> None:
    """Measure no checkpoint, synchronous DCP, and native asynchronous DCP."""
    directory = run_benchmark(config, output_root, repetitions, warmups=warmups)
    typer.echo(f"Benchmark report: {directory / 'report.md'}")
