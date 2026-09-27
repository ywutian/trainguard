"""Real CPU evidence checks for customer-keyed guarded sample records."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.config import load_config
from trainguard.privacy import protect_sample_event
from trainguard.validation import validate_runs


def _guarded_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    key_path = tmp_path / "customer-sample.key"
    key_path.write_bytes(b"a-private-customer-key-with-at-least-32-bytes")
    key_path.chmod(0o600)
    monkeypatch.setenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE", str(key_path))
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["run"]["profile"] = "guarded"
    raw["recovery"]["max_restarts"] = 1
    raw["checkpoint"].update(
        mode="sync", interval_steps=1, keep_last_k=2,
        max_checkpoint_bytes=10_000_000, max_retained_bytes=20_000_000,
        min_free_bytes=1_000_000, max_event_log_bytes=1_000_000,
    )
    path = tmp_path / "guarded.json"
    path.write_text(json.dumps(raw))
    return path


def _sample_events(run_dir: Path) -> list[dict]:
    events = []
    for path in sorted((run_dir / "attempts").glob("*/rank-*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    return [
        event for event in events
        if event["event_type"] in {"batch_consumed", "step_completed", "update_skipped"}
    ]


def test_guarded_evidence_hides_ids_and_compares_exact_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    reference, reference_ok = controller.run(source, tmp_path / "reference")
    recovered, recovered_ok = controller.run(source, tmp_path / "recovered")
    assert reference_ok and recovered_ok
    for run_dir in (reference, recovered):
        assert "TRAINGUARD_SAMPLE_HMAC_KEY_FILE" not in (run_dir / "run.json").read_text()
        for event in _sample_events(run_dir):
            assert "sample_ids" not in event
            assert event["sample_count"] >= 1
            assert len(event["sample_commitment"]) == 64
            assert len(event["sample_evidence_mac"]) == 64
    result = validate_runs(reference, recovered)
    assert result["passed"], result["differences"]


def test_guarded_missing_wrong_key_and_sample_tampering_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert succeeded
    key_path = Path(os.environ["TRAINGUARD_SAMPLE_HMAC_KEY_FILE"])
    monkeypatch.delenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE")
    with pytest.raises(ValueError, match="key file is required"):
        controller.resume(run_dir)
    assert not validate_runs(run_dir, run_dir)["passed"]
    wrong_key = tmp_path / "wrong.key"
    wrong_key.write_bytes(b"another-private-customer-key-with-32-bytes")
    wrong_key.chmod(0o600)
    monkeypatch.setenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE", str(wrong_key))
    with pytest.raises(ValueError, match="differs from the run identity"):
        controller.resume(run_dir)
    assert not validate_runs(run_dir, run_dir)["passed"]
    monkeypatch.setenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE", str(key_path))
    path = run_dir / "attempts" / "attempt-001" / "rank-0.jsonl"
    original = path.read_text()
    lines = [json.loads(line) for line in original.splitlines()]
    sample = next(event for event in lines if event["event_type"] == "step_completed")
    sample["sample_commitment"] = "0" * 64
    path.write_text("".join(json.dumps(event) + "\n" for event in lines))
    result = validate_runs(run_dir, run_dir)
    assert not result["passed"]
    assert any("sample evidence MAC differs" in item for item in result["differences"])
    assert not controller.resume(run_dir)
    failed_status = json.loads((run_dir / "run.json").read_text())
    assert failed_status["status"] == "FAILED"
    assert failed_status["post_run_audit"]["status"] == "INVALIDATED"
    path.write_text(original)
    assert controller.resume(run_dir)
    assert validate_runs(run_dir, run_dir)["passed"]
    status_path = run_dir / "run.json"
    original_status = status_path.read_text()
    status = json.loads(original_status)
    status["sample_key_id"] = "é" * 64
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="identity is missing"):
        controller.resume(run_dir)
    assert not validate_runs(run_dir, run_dir)["passed"]
    status_path.write_text(original_status)


def test_guarded_signed_duplicate_synthetic_samples_fail_completion_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert succeeded
    path = run_dir / "attempts" / "attempt-001" / "rank-0.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(event for event in lines if event["event_type"] == "step_completed")
    fields = {
        name: value for name, value in target.items()
        if name not in {"time", "sample_count", "sample_commitment", "sample_evidence_mac"}
    }
    key = Path(os.environ["TRAINGUARD_SAMPLE_HMAC_KEY_FILE"]).read_bytes()
    signed = protect_sample_event(fields, [7, 7], key)
    target.update({
        name: value for name, value in signed.items()
        if name in {"sample_count", "sample_commitment", "sample_evidence_mac"}
    })
    path.write_text("".join(json.dumps(event) + "\n" for event in lines))
    result = validate_runs(run_dir, run_dir)
    assert not result["passed"]
    assert any("synthetic sample commitment differs" in item for item in result["differences"])
    assert not controller.resume(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["status"] == "FAILED"


def test_guarded_unicode_mac_invalidates_previous_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert succeeded
    path = run_dir / "attempts" / "attempt-001" / "rank-0.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(event for event in lines if event["event_type"] == "step_completed")
    target["sample_evidence_mac"] = "é" * 64
    path.write_text("".join(json.dumps(event) + "\n" for event in lines))
    result = validate_runs(run_dir, run_dir)
    assert not result["passed"]
    assert any("commitment is incomplete" in item for item in result["differences"])
    assert not controller.resume(run_dir)
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FAILED"
    assert status["post_run_audit"]["status"] == "INVALIDATED"


def test_guarded_key_is_checked_before_external_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    monkeypatch.delenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE")
    invoked = []
    monkeypatch.setattr(controller, "preflight", lambda config: invoked.append(config))
    with pytest.raises(ValueError, match="key file is required"):
        controller.run(source, tmp_path / "runs")
    assert not invoked
    assert not (tmp_path / "runs").exists()


def test_guarded_recovery_matches_uninterrupted_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    reference, succeeded = controller.run(source, tmp_path / "reference")
    assert succeeded
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('RANK') == '0':\n"
        "    import trainguard.checkpoint_io as io\n"
        "    original = io.finish_save\n"
        "    def stop_after_commit(*args, **kwargs):\n"
        "        result = original(*args, **kwargs)\n"
        "        if args[0].step == 1 and args[3] == 'attempt-001':\n"
        "            os._exit(77)\n"
        "        return result\n"
        "    io.finish_save = stop_after_commit\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(hook) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    recovered, succeeded = controller.run(source, tmp_path / "recovered")
    assert succeeded, (recovered / "launcher.log").read_text()
    assert (recovered / "attempts" / "attempt-002").exists()
    result = validate_runs(reference, recovered)
    assert result["passed"], result["differences"]


def test_guarded_accumulated_synthetic_samples_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    raw = json.loads(source.read_text())
    raw["training"]["gradient_accumulation_steps"] = 2
    source.write_text(json.dumps(raw))
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert succeeded, (run_dir / "launcher.log").read_text()
    result = validate_runs(run_dir, run_dir)
    assert result["passed"], result["differences"]
