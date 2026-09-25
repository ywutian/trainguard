"""Command-line entry points for the runnable baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from trainguard import __version__
from trainguard.config import load_config
from trainguard.controller import run as launch_run

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
    """Check a CPU baseline configuration and print its fingerprint."""
    settings = load_config(config)
    typer.echo(settings.fingerprint())


@app.command("run")
def run(
    config: Annotated[
        Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)
    ],
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Run one fixed-size CPU DDP attempt."""
    run_dir, succeeded = launch_run(config, output_root)
    typer.echo(f"Run directory: {run_dir}")
    if not succeeded:
        typer.echo(f"Training failed; inspect {run_dir / 'launcher.log'}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Training completed; summary: {run_dir / 'summary.json'}")
