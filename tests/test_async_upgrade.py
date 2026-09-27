import json
import time
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from trainguard import checkpoint_io
from trainguard.config import load_config
from trainguard.controller import run


@pytest.fixture
def single_group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


def test_future_ready_and_failure_are_collective(single_group, tmp_path):
    ready = getattr(checkpoint_io, "save_ready", None)
    assert callable(ready), "save_ready must coordinate pending futures"
    future = Future()
    pending = checkpoint_io.PendingSave(tmp_path, 1, 0, 0, future)
    assert not ready(pending, single_group)
    future.set_result(None)
    assert ready(pending, single_group)
    failed = Future()
    failed.set_exception(RuntimeError("upload failed"))
    pending.future = failed
    with pytest.raises(RuntimeError, match="upload"):
        ready(pending, single_group)
    assert not (tmp_path / "COMMITTED").exists()


def test_one_slow_rank_blocks_commit(single_group, tmp_path, monkeypatch):
    ready = getattr(checkpoint_io, "save_ready", None)
    assert callable(ready), "save_ready must coordinate pending futures"
    future = Future()
    future.set_result(None)
    pending = checkpoint_io.PendingSave(tmp_path, 1, 0, 0, future)
    original = dist.all_reduce

    def remote_pending(tensor, *args, **kwargs):
        original(tensor, *args, **kwargs)
        tensor[0] += 1

    monkeypatch.setattr(dist, "all_reduce", remote_pending)
    assert not ready(pending, single_group)
    assert not (tmp_path / "COMMITTED").exists()


def test_async_commit_is_observed_before_next_save_interval(tmp_path):
    source = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    source["training"]["total_steps"] = 300
    source["checkpoint"] = {"mode": "async", "interval_steps": 100}
    config = tmp_path / "async.json"
    config.write_text(json.dumps(source))
    directory, succeeded = run(config, tmp_path / "runs")
    assert succeeded, (directory / "launcher.log").read_text()
    events = [
        json.loads(line)
        for line in (directory / "attempts/attempt-001/rank-0.jsonl").read_text().splitlines()
    ]
    saved = next(event for event in events if event["event_type"] == "checkpoint_committed")
    assert saved.get("committed_at_step", 200) < 200
    for field in (
        "preparation_seconds",
        "upload_seconds",
        "main_thread_wait_seconds",
        "eligibility_lag_seconds",
        "recoverable_step_lag",
    ):
        assert field in saved
        assert saved[field] >= 0


def test_pending_upload_expires_at_coordinated_boundary(single_group, tmp_path):
    import time
    pending = checkpoint_io.PendingSave(tmp_path, 1, 0, 0, Future())
    pending.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match='deadline'):
        checkpoint_io.save_ready(pending, single_group)


def test_completed_upload_without_callback_timestamp_cannot_pass_expired_deadline(
    single_group, tmp_path
):
    future = Future()
    future.set_result(None)
    pending = checkpoint_io.PendingSave(tmp_path, 1, 0, 0, future)
    pending.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="deadline"):
        checkpoint_io.save_ready(pending, single_group)


def test_final_flush_coordinates_upload_failure(single_group, tmp_path):
    settings = load_config(Path(__file__).parents[1] / 'configs/cpu_demo.yaml')
    future = Future()
    future.set_exception(ValueError('bad storage'))
    pending = checkpoint_io.PendingSave(tmp_path, 1, 0, 0, future)
    with pytest.raises(RuntimeError, match='upload'):
        checkpoint_io.finish_save(pending, settings, 'run', 'attempt-001', 0,
                                  tmp_path / 'events.jsonl', control_group=single_group)
    assert not (tmp_path / 'COMMITTED').exists()


def _commit_failure_worker(rank: int, directory: str) -> None:
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'group'}", rank=rank, world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        config = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml")
        pending = checkpoint_io.PendingSave(
            root / "candidate", 1, time.monotonic(), 0.0, None,
            upload_started=time.monotonic(), upload_finished=time.monotonic(),
        )
        if rank == 0:
            def fail_commit(*args, **kwargs):
                raise OSError("injected commit I/O failure")

            checkpoint_io.commit_checkpoint = fail_commit
        (root / f"rank-{rank}.jsonl").write_text("{}\n")
        started = time.monotonic()
        try:
            checkpoint_io.finish_save(
                pending, config, "run", "attempt-001", rank,
                root / f"rank-{rank}.jsonl", control_group=dist.group.WORLD,
            )
        except (OSError, RuntimeError) as exc:
            result = {"error": str(exc), "elapsed_seconds": time.monotonic() - started}
        else:
            result = {"error": None, "elapsed_seconds": time.monotonic() - started}
        (root / f"result-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_commit_failure_reaches_every_rank_before_group_timeout(tmp_path):
    mp.spawn(_commit_failure_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"result-{rank}.json").read_text())
        assert "injected commit I/O failure" in result["error"]
        assert "Timed out" not in result["error"]
    assert not (tmp_path / "candidate" / "COMMITTED").exists()


def _cancelled_upload_worker(rank: int, directory: str) -> None:
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=f"file://{root / 'cancel-group'}", rank=rank, world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        future = Future()
        if rank == 0:
            future.cancel()
        else:
            future.set_result(None)
        pending = checkpoint_io.PendingSave(root / "candidate", 1, 0, 0, future)
        started = time.monotonic()
        try:
            checkpoint_io.save_ready(pending, dist.group.WORLD)
        except RuntimeError as exc:
            result = {"error": str(exc), "elapsed_seconds": time.monotonic() - started}
        else:
            result = {"error": None, "elapsed_seconds": time.monotonic() - started}
        (root / f"cancel-result-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_cancelled_upload_notifies_every_rank_before_group_timeout(tmp_path):
    mp.spawn(_cancelled_upload_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"cancel-result-{rank}.json").read_text())
        assert "checkpoint upload failed" in result["error"]
        assert result["elapsed_seconds"] < 8
    assert not (tmp_path / "candidate" / "COMMITTED").exists()
