"""Successful run reports must still have usable checkpoint evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainguard import controller, run_evidence
from trainguard.checkpoint import candidate_path, ordered_candidates
from trainguard.config import load_config
from trainguard.support import SupportBundleError, build_support_bundle
from trainguard.validation import validate_runs


def _source(tmp_path: Path, *, guarded: bool, checkpoint_mode: str) -> Path:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["training"]["total_steps"] = 3
    raw["checkpoint"].update(mode=checkpoint_mode, interval_steps=1)
    if guarded:
        raw["run"]["profile"] = "guarded"
        raw["checkpoint"].update(
            keep_last_k=2,
            max_checkpoint_bytes=10_000_000,
            max_retained_bytes=20_000_000,
            min_free_bytes=1_000_000,
            max_event_log_bytes=1_000_000,
        )
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _assert_rejected(run_dir: Path) -> None:
    result = validate_runs(run_dir, run_dir)
    assert result["passed"] is False
    assert any("checkpoint" in item for item in result["differences"])
    with pytest.raises(SupportBundleError, match="checkpoint"):
        build_support_bundle(run_dir)
    assert run_evidence.trusted_measurement(run_dir) is None


def test_guarded_completed_evidence_rechecks_final_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = tmp_path / "customer.key"
    key.write_bytes(b"customer-owned-sample-key-with-enough-bytes")
    key.chmod(0o600)
    monkeypatch.setenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE", str(key))
    run_dir, succeeded = controller.run(
        _source(tmp_path, guarded=True, checkpoint_mode="sync"), tmp_path / "runs"
    )
    assert succeeded
    assert validate_runs(run_dir, run_dir)["passed"] is True
    assert build_support_bundle(run_dir)["status"] == "SUCCEEDED"
    assert run_evidence.trusted_measurement(run_dir) is not None

    candidates = ordered_candidates(run_dir)
    assert len(candidates) == 2
    for candidate in candidates:
        marker = candidate / "COMMITTED"
        original = marker.read_bytes()
        marker.write_bytes(b"invalid marker\n")
        _assert_rejected(run_dir)
        marker.write_bytes(original)
        assert validate_runs(run_dir, run_dir)["passed"] is True

    excess = run_dir / "checkpoints" / "unexpected.bin"
    with excess.open("wb") as stream:
        stream.truncate(20_000_001)
    with monkeypatch.context() as patch:
        patch.setattr(
            run_evidence, "validate_checkpoint",
            lambda *args, **kwargs: pytest.fail("payload validation preceded capacity preflight"),
        )
        _assert_rejected(run_dir)
    excess.unlink()
    assert validate_runs(run_dir, run_dir)["passed"] is True


def test_experiment_completed_evidence_rechecks_final_checkpoint(tmp_path: Path) -> None:
    run_dir, succeeded = controller.run(
        _source(tmp_path, guarded=False, checkpoint_mode="sync"), tmp_path / "runs"
    )
    assert succeeded
    assert validate_runs(run_dir, run_dir)["passed"] is True
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    final = status["post_run_audit"]["final_checkpoint"]
    candidate = candidate_path(run_dir, final["attempt_id"], final["global_step"])
    marker = candidate / "COMMITTED"
    original = marker.read_bytes()
    marker.write_bytes(b"invalid marker\n")
    _assert_rejected(run_dir)
    marker.write_bytes(original)
    assert validate_runs(run_dir, run_dir)["passed"] is True

    excess = candidate / "unexpected.bin"
    with excess.open("wb") as stream:
        stream.truncate(run_evidence.DEFAULT_EXPERIMENT_CHECKPOINT_AUDIT_BYTES + 1)
    _assert_rejected(run_dir)


def test_checkpoint_free_completed_audit_must_remain_not_applicable(tmp_path: Path) -> None:
    run_dir, succeeded = controller.run(
        _source(tmp_path, guarded=False, checkpoint_mode="none"), tmp_path / "runs"
    )
    assert succeeded
    assert validate_runs(run_dir, run_dir)["passed"] is True
    status_path = run_dir / "run.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["post_run_audit"]["status"] = "PASSED"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    result = validate_runs(run_dir, run_dir)
    assert result["passed"] is False
    assert any("checkpoint-free post-run audit" in item for item in result["differences"])
    with pytest.raises(SupportBundleError, match="checkpoint-free post-run audit"):
        build_support_bundle(run_dir)
    assert run_evidence.trusted_measurement(run_dir) is None
