"""Local DCP payload microbenchmarks, separate from end-to-end training."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from trainguard.checkpoint import candidate_path, capture_rank_state, commit_checkpoint
from trainguard.config import ProjectConfig
from trainguard.environment import environment_snapshot
from trainguard.events import write_json_atomic


def run_storage_benchmark(
    output_root: Path, sizes_mib: tuple[int, ...] = (64, 256), repetitions: int = 3
) -> Path:
    if (
        repetitions < 1
        or not sizes_mib
        or any(type(size) is not int or not 1 <= size <= 1024 for size in sizes_mib)
    ):
        raise ValueError("storage benchmark requires positive sizes up to 1024 MiB and repetitions")
    if dist.is_initialized():
        raise RuntimeError("storage benchmark requires its own process group")
    directory = (output_root / f"storage-{uuid.uuid4().hex[:12]}").resolve()
    directory.mkdir(parents=True)
    result = {
        "status": "RUNNING",
        "rows": [],
        "environment": environment_snapshot(1, "cpu", directory),
        "measurement": "single-rank synthetic payload; excludes model training; host and cache not isolated",
    }
    write_json_atomic(directory / "storage.json", result)
    config = ProjectConfig.model_validate(
        {
            "run": {"world_size": 1},
            "training": {"total_steps": 1, "sequence_length": 16, "batch_size_per_rank": 1},
        }
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/coordination", rank=0, world_size=1
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for size in sizes_mib:
            for repeat in range(1, repetitions + 1):
                for mode in ("sync", "async"):
                    row = {"size_mib": size, "repeat": repeat, "mode": mode, "status": "RUNNING"}
                    result["rows"].append(row)
                    write_json_atomic(directory / "storage.json", result)
                    case = directory / f"{size}-{repeat}-{mode}"
                    path = candidate_path(case, "attempt-001", 1)
                    path.mkdir(parents=True)
                    tensor = torch.full((size * 2**20 // 4,), 0.125)
                    state = {"payload": tensor}
                    started = time.monotonic()
                    if mode == "async":
                        future = dcp.async_save(state, checkpoint_id=path / "dcp")
                        staging = time.monotonic() - started
                        submitted = time.monotonic()
                        completed = []
                        ready = threading.Event()

                        def mark(future, completed=completed, ready=ready):
                            completed.append(time.monotonic())
                            ready.set()

                        future.add_done_callback(mark)
                        waiting = time.monotonic()
                        future.result(timeout=120)
                        ready.wait(timeout=1)
                        blocked = time.monotonic() - waiting
                        upload = max(
                            0.0, (completed[0] if completed else time.monotonic()) - submitted
                        )
                    else:
                        dcp.save(state, checkpoint_id=path / "dcp")
                        staging, upload = 0.0, time.monotonic() - started
                        blocked = upload
                    write_json_atomic(
                        path / "rank-0.json",
                        capture_rank_state(config, "micro", "attempt-001", 0, 1, {}),
                    )
                    commit_started = time.monotonic()
                    record = commit_checkpoint(path, config, "micro", "attempt-001", 1)
                    hashing = time.monotonic() - commit_started
                    target = {"payload": torch.zeros_like(tensor)}
                    loading = time.monotonic()
                    dcp.load(target, checkpoint_id=path / "dcp")
                    load_seconds = time.monotonic() - loading
                    validated = torch.equal(tensor, target["payload"])
                    row.update(
                        status="PASSED" if validated else "FAILED",
                        validated=validated,
                        tensor_bytes=tensor.numel() * tensor.element_size(),
                        payload_bytes=sum(item["size"] for item in record.manifest["files"]),
                        staging_seconds=staging,
                        upload_seconds=upload,
                        main_thread_wait_seconds=blocked,
                        hash_commit_seconds=hashing,
                        load_seconds=load_seconds,
                    )
                    write_json_atomic(directory / "storage.json", result)
                    if not validated:
                        raise RuntimeError("storage tensor differs after DCP load")
        result["status"] = "SUCCEEDED"
    except BaseException as exc:
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json_atomic(directory / "storage.json", result)
        dist.destroy_process_group()
        torch.set_num_threads(old_threads)
    lines = [
        "# Local DCP storage microbenchmark",
        "",
        result["measurement"],
        "",
        "| MiB | Mode | Repeat | Staging s | Upload s | Hash + commit s | Load s | Exact |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['size_mib']} | {row['mode']} | {row['repeat']} | {row['staging_seconds']:.4f} | {row['upload_seconds']:.4f} | {row['hash_commit_seconds']:.4f} | {row['load_seconds']:.4f} | {row['validated']} |"
        )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return directory
