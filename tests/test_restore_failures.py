"""Conservative classification of interrupted checkpoint restores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.config import load_config
from trainguard.events import write_json_atomic
from trainguard.restore_failures import (
    failed_restore_candidates,
    record_group_ended,
    record_restore_incomplete,
    record_restore_progress,
)
from trainguard.run_store import RunStore

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
ATTEMPT = "attempt-002"


def _attempt(checkpoint: Path) -> list[dict]:
    return [{"attempt_id": ATTEMPT, "resume_checkpoint": str(checkpoint)}]


@pytest.mark.parametrize(
    ("phases", "reason", "expected"),
    [
        (("restore_started", "restore_started"), "worker startup or first update stalled", True),
        (("state_loaded", "restore_started"), "launcher exit code 74", True),
        (("payload_loaded", "restore_started"), "launcher exit code 74", True),
        (("payload_loaded", "payload_loaded"), "worker startup or first update stalled", False),
        (("state_loaded", "state_loaded"), "worker startup or first update stalled", False),
        (("restore_started", None), "launcher exit code 74", False),
        ((None, "restore_started"), "launcher exit code 74", False),
        (("restore_started", "restore_started"), "interrupted by user", False),
        (("restore_started", "restore_started"), "controller OSError: unavailable", False),
    ],
)
def test_incomplete_verdict_requires_all_rank_start_and_post_cleanup(
    tmp_path: Path, phases: tuple[str | None, str | None], reason: str, expected: bool
) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-000002-attempt-001"
    for rank, phase in enumerate(phases):
        if phase is not None:
            record_restore_progress(
                tmp_path, "run", ATTEMPT, rank, checkpoint, DIGEST, phase
            )
    record_group_ended(tmp_path, "run", ATTEMPT, reason, "controller_cleanup")
    assert record_restore_incomplete(
        tmp_path, "run", ATTEMPT, checkpoint, 2, DIGEST
    ) is expected
    failures = failed_restore_candidates(tmp_path, "run", 2, _attempt(checkpoint))
    assert (DIGEST in failures.incomplete.get(checkpoint, set())) is expected
    assert failures.explicit == {}
    if expected:
        assert record_restore_incomplete(
            tmp_path, "run", ATTEMPT, checkpoint, 2, DIGEST
        )


def test_no_verdict_before_worker_group_cleanup(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-000002-attempt-001"
    for rank in range(2):
        record_restore_progress(
            tmp_path, "run", ATTEMPT, rank, checkpoint, DIGEST, "restore_started"
        )
    assert not record_restore_incomplete(
        tmp_path, "run", ATTEMPT, checkpoint, 2, DIGEST
    )


def test_mismatched_progress_or_verdict_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-000002-attempt-001"
    record_group_ended(tmp_path, "run", ATTEMPT, "launcher exit code 74", "controller_cleanup")
    record_restore_progress(
        tmp_path, "run", ATTEMPT, 0, checkpoint, DIGEST, "restore_started"
    )
    record_restore_progress(
        tmp_path, "run", ATTEMPT, 1, checkpoint, OTHER_DIGEST, "restore_started"
    )
    with pytest.raises(ValueError, match="progress evidence identity"):
        record_restore_incomplete(tmp_path, "run", ATTEMPT, checkpoint, 2)
    record_restore_progress(
        tmp_path, "run", ATTEMPT, 1, checkpoint, DIGEST, "restore_started"
    )
    assert record_restore_incomplete(tmp_path, "run", ATTEMPT, checkpoint, 2, DIGEST)
    verdict = tmp_path / "attempts" / ATTEMPT / "restore-incomplete.json"
    data = json.loads(verdict.read_text())
    data["manifest_sha256"] = OTHER_DIGEST
    write_json_atomic(verdict, data)
    with pytest.raises(ValueError, match="progress evidence identity"):
        failed_restore_candidates(tmp_path, "run", 2, _attempt(checkpoint))


def test_interrupted_attempt_cannot_publish_incomplete_verdict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-000002-attempt-001"
    for rank in range(2):
        record_restore_progress(
            tmp_path, "run", ATTEMPT, rank, checkpoint, DIGEST, "restore_started"
        )
    record_group_ended(tmp_path, "run", ATTEMPT, "interrupted by user", "controller_cleanup")
    write_json_atomic(
        tmp_path / "attempts" / ATTEMPT / "restore-incomplete.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "attempt_id": ATTEMPT,
            "checkpoint_path": str(checkpoint),
            "manifest_sha256": DIGEST,
            "incomplete_ranks": [0, 1],
        },
    )
    with pytest.raises(ValueError, match="lacks matching evidence"):
        failed_restore_candidates(tmp_path, "run", 2, _attempt(checkpoint))


def test_explicit_failure_remains_distinct_from_incomplete_verdict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-000002-attempt-001"
    record_group_ended(tmp_path, "run", ATTEMPT, "launcher exit code 1", "controller_cleanup")
    for rank in range(2):
        record_restore_progress(
            tmp_path, "run", ATTEMPT, rank, checkpoint, DIGEST, "restore_started"
        )
    write_json_atomic(
        tmp_path / "attempts" / ATTEMPT / "rank-0-restore-failure.json",
        {
            "schema_version": 1,
            "run_id": "run",
            "attempt_id": ATTEMPT,
            "rank": 0,
            "checkpoint_path": str(checkpoint),
            "manifest_sha256": DIGEST,
            "error_type": "CheckpointException",
        },
    )
    assert record_restore_incomplete(tmp_path, "run", ATTEMPT, checkpoint, 2, DIGEST)
    failures = failed_restore_candidates(tmp_path, "run", 2, _attempt(checkpoint))
    assert failures.explicit == {checkpoint: {DIGEST}}
    assert failures.incomplete == {checkpoint: {DIGEST}}


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (("restore_started", "restore_started"), True),
        (("restore_started", None), False),
        (("state_loaded", "state_loaded"), False),
    ],
)
def test_controller_restart_classifies_after_worker_group_disappeared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    phases: tuple[str | None, str | None], expected: bool,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml")
    checkpoint = tmp_path / "checkpoints" / "step-000002-attempt-001"
    for rank, phase in enumerate(phases):
        if phase is not None:
            record_restore_progress(
                tmp_path, "run", ATTEMPT, rank, checkpoint, DIGEST, phase
            )
    store = RunStore(tmp_path / "run.sqlite3")
    store.create_run("run", config.fingerprint(), "now")
    store.start_attempt("run", "attempt-001", 1, None, 0)
    store.finish_attempt("attempt-001", "FAILED", 71, "injected worker exit")
    store.start_attempt("run", ATTEMPT, 2, str(checkpoint), 2)
    monkeypatch.setattr(controller, "_owned_process_alive", lambda *_args: False)

    class ScanReached(BaseException):
        pass

    def stop_before_selection(*_args):
        raise ScanReached

    monkeypatch.setattr(controller, "_scan_checkpoints", stop_before_selection)
    try:
        with pytest.raises(ScanReached):
            controller._drive(tmp_path, {"run_id": "run"}, store, config)
        assert store.attempts("run")[-1]["status"] == "INTERRUPTED"
    finally:
        store.close()
    group = json.loads((tmp_path / "attempts" / ATTEMPT / "worker-group-ended.json").read_text())
    assert group["observation"] == "no_owned_workers"
    assert (tmp_path / "attempts" / ATTEMPT / "restore-incomplete.json").is_file() is expected
