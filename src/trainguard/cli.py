"""Command-line entry points for training, recovery, and experiments."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer

from trainguard import __version__
from trainguard.benchmark import resume_benchmark, run_benchmark
from trainguard.campaign import resume_campaign, run_campaign
from trainguard.config import load_config
from trainguard.controller import ExperimentNotAuthorizedError, RunActiveError
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


@app.command("init-config")
def init_config(
    output: Annotated[Path, typer.Option("--output")] = Path("cpu_demo.yaml"),
) -> None:
    """Write the packaged CPU example without replacing an existing file."""
    try:
        template = files("trainguard").joinpath("templates/cpu_demo.yaml").read_text(
            encoding="utf-8"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(template)
    except FileExistsError as exc:
        typer.echo(f"Configuration already exists: {output}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Configuration written: {output}")


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)],
) -> None:
    """Check a fixed-topology training configuration and print its fingerprint."""
    settings = load_config(config)
    typer.echo(settings.fingerprint())


@app.command("run")
def run(
    config: Annotated[Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    allow_experiment: Annotated[bool, typer.Option("--allow-experiment")] = False,
    reference_store: Annotated[Path | None, typer.Option("--reference-store")] = None,
) -> None:
    """Run a fixed-size training workload with bounded recovery."""
    try:
        run_dir, succeeded = launch_run(
            config, output_root, allow_experiment=allow_experiment,
            reference_store_path=reference_store,
        )
    except ExperimentNotAuthorizedError as exc:
        typer.echo(f"{exc}; pass --allow-experiment in an isolated test", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Run directory: {run_dir}")
    if not succeeded:
        typer.echo(f"Training failed; inspect {run_dir / 'launcher.log'}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Training completed; summary: {run_dir / 'summary.json'}")


@app.command("support-bundle")
def support_bundle(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Export a limited, read-only diagnostic summary for customer review."""
    from trainguard.support import SupportBundleError, export_support_bundle

    try:
        export_support_bundle(run_dir, output)
    except SupportBundleError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Diagnostic bundle: {output}")


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
    config: Annotated[Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    repetitions: Annotated[int, typer.Option("--repetitions", min=3)] = 3,
    warmups: Annotated[int, typer.Option("--warmups", min=0)] = 1,
) -> None:
    """Measure no checkpoint, synchronous DCP, and native asynchronous DCP."""
    directory = run_benchmark(config, output_root, repetitions, warmups=warmups)
    typer.echo(f"Benchmark report: {directory / 'report.md'}")


@app.command("benchmark-resume")
def benchmark_resume(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Continue missing experiment slots with the original source and workload."""
    result = resume_benchmark(directory)
    typer.echo(f"Benchmark report: {result / 'report.md'}")


@app.command("acceptance")
def acceptance(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
) -> None:
    """Run reference, recovery faults and omitted-state negative controls."""
    import json

    directory = run_campaign(config, output_root)
    typer.echo(f"Acceptance report: {directory / 'report.md'}")
    result = json.loads((directory / "acceptance.json").read_text())
    if result["status"] != "SUCCEEDED":
        raise typer.Exit(2 if result["status"] == "BLOCKED" else 1)


@app.command("acceptance-resume")
def acceptance_resume(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Continue a stopped acceptance campaign and recheck completed evidence."""
    import json

    resume_campaign(directory)
    typer.echo(f"Acceptance report: {directory / 'report.md'}")
    if json.loads((directory / "acceptance.json").read_text())["status"] != "SUCCEEDED":
        raise typer.Exit(1)


@app.command("checkpoint-budget")
def checkpoint_budget(
    overhead_seconds: Annotated[float, typer.Option(min=0)],
    mtbf_seconds: Annotated[float, typer.Option(min=0)],
    commit_lag_seconds: Annotated[float, typer.Option(min=0)],
    rto_seconds: Annotated[float, typer.Option(min=0)],
    rollback_budget_seconds: Annotated[float, typer.Option(min=0)],
    upload_seconds: Annotated[float, typer.Option(min=0)],
) -> None:
    """Estimate an interval using measured costs and an explicit job MTBF assumption."""
    import json

    from trainguard.policy import suggest_interval

    result = suggest_interval(
        overhead_seconds=overhead_seconds,
        mtbf_seconds=mtbf_seconds,
        commit_lag_seconds=commit_lag_seconds,
        rto_seconds=rto_seconds,
        rollback_budget_seconds=rollback_budget_seconds,
        upload_seconds=upload_seconds,
    )
    typer.echo(json.dumps(result, indent=2))
    if not result["feasible"]:
        raise typer.Exit(1)


@app.command("audit-checkpoints")
def audit_checkpoints(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Fully inspect checkpoint history and index all outcomes."""
    import json

    from trainguard.controller import _controller_lock, _scan_checkpoints
    from trainguard.run_store import RunStore

    status = json.loads((run_dir / "run.json").read_text())
    config = load_config(run_dir / "config.json")
    with _controller_lock(run_dir):
        store = RunStore(run_dir / "run.sqlite3")
        try:
            selected = _scan_checkpoints(run_dir, config, status["run_id"], store, audit=True)
        finally:
            store.close()
    typer.echo(str(selected.path) if selected else "No valid committed checkpoint")


@app.command("storage-benchmark")
def storage_benchmark(
    output_root: Annotated[Path, typer.Option("--output-root")] = DEFAULT_OUTPUT_ROOT,
    repetitions: Annotated[int, typer.Option("--repetitions", min=1)] = 3,
) -> None:
    """Measure 64/256 MiB local DCP payloads separately from training."""
    from trainguard.storage_benchmark import run_storage_benchmark

    directory = run_storage_benchmark(output_root, repetitions=repetitions)
    typer.echo(f"Storage report: {directory / 'report.md'}")
