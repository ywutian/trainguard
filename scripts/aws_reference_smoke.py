"""Run the checkpoint protocol against a real S3 bucket and DynamoDB table.

Usage (credentials come from the ambient AWS chain; this script never reads keys):

    uv run --with boto3 python scripts/aws_reference_smoke.py \
        --location 'aws://<bucket>/<prefix>?region=<region>&table=<table>' \
        --receipt runs/aws-smoke.json [--training]

Each invocation uses a fresh run ID and removes only that run's objects and
records afterwards. A passing receipt is evidence for the named bucket, table,
region and credentials at that time; it says nothing about other services,
cross-host isolation, or power loss.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from trainguard import aws_store
from trainguard.remote_protocol import (
    EpochAuthority,
    FencedOut,
    InMemoryIsolationOracle,
    PreconditionFailed,
    RemoteCheckpointProtocol,
    ResponseLost,
)

ROOT = Path(__file__).resolve().parents[1]


def _publish(protocol, controller, worker, generation: str, step: int):
    expected = {
        path: protocol.write_payload(worker, generation, path, f"{generation}:{path}".encode())
        for path in ("dcp/.metadata", "dcp/__0_0.distcp", "rank-0.json")
    }
    protocol.seal(controller, generation, step, expected)
    return protocol.publish(controller, generation)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def mutable_revisions(store, run_id: str) -> None:
    key = f"runs/{run_id}/HEAD"
    first = store.put(key, b"A", if_none_match=True)
    second = store.put(key, b"B", if_match=first.etag)
    third = store.put(key, b"A", if_match=second.etag)
    _check(len({first.etag, second.etag, third.etag}) == 3, "revision token repeated")
    try:
        store.put(key, b"stale", if_match=first.etag)
    except PreconditionFailed:
        pass
    else:
        raise AssertionError("stale revision overwrote HEAD")
    _check(store.get(key) == third, "HEAD readback differs")


def create_only_generation(store, run_id: str) -> None:
    key = f"runs/{run_id}/generations/g0/payload/object"
    store.put(key, b"first", if_none_match=True)
    try:
        store.put(key, b"second", if_none_match=True)
    except PreconditionFailed:
        pass
    else:
        raise AssertionError("S3 accepted a second create for one key")
    _check(store.get(key).data == b"first", "generation object changed")


def concurrent_head_writers(store, run_id: str, writers: int = 8) -> None:
    key = f"runs/{run_id}/AUTHORITY"
    first = store.put(key, b"initial", if_none_match=True)
    barrier = threading.Barrier(writers)
    outcomes: list[str] = []

    def race(index: int) -> None:
        barrier.wait(timeout=30)
        try:
            store.put(key, f"writer-{index}".encode(), if_match=first.etag)
            outcomes.append("committed")
        except PreconditionFailed:
            outcomes.append("conflict")

    threads = [threading.Thread(target=race, args=(index,)) for index in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    _check(sorted(outcomes) == ["committed"] + ["conflict"] * (writers - 1),
           f"CAS race outcomes were {sorted(outcomes)}")


def publish_fallback_and_takeover(store, raw_s3, run_id: str) -> None:
    oracle = InMemoryIsolationOracle()
    authority = EpochAuthority(oracle)
    controller = authority.start(run_id, "controller-a")
    worker = authority.worker(controller, "worker-a")
    protocol = RemoteCheckpointProtocol(store, authority, run_id, "smoke-identity")
    older = _publish(protocol, controller, worker, "generation-a", 10)
    newest = _publish(protocol, controller, worker, "generation-b", 20)
    _check(protocol.select_latest().chosen == newest, "newest publication not selected")
    damaged = f"{store.aws.prefix}/" + protocol.payload_key("generation-b", "rank-0.json")
    raw_s3.put_object(Bucket=store.aws.bucket, Key=damaged, Body=b"bitrot")
    selection = protocol.select_latest()
    _check(selection.chosen == older and len(selection.rejected) == 1, "no fallback on damage")
    oracle.confirm(run_id, controller.epoch)
    successor = protocol.takeover("controller-b")
    for stale in (
        lambda: protocol.write_payload(worker, "generation-stale", "weights", b"old"),
        lambda: protocol.publish(controller, "generation-a"),
    ):
        try:
            stale()
        except FencedOut:
            continue
        raise AssertionError("old epoch actor was not fenced")
    new_worker = authority.worker(successor, "worker-b")
    _check(_publish(protocol, successor, new_worker, "generation-c", 30).epoch == 2,
           "successor publication failed")


def lost_ack_readback(store, run_id: str) -> None:
    """Drop the response after the real service applied the write, then reconcile."""
    oracle = InMemoryIsolationOracle()
    authority = EpochAuthority(oracle)
    controller = authority.start(run_id, "controller-lost")
    worker = authority.worker(controller, "worker-lost")
    protocol = RemoteCheckpointProtocol(store, authority, run_id, "smoke-identity")
    real_put = store.put
    lose = {protocol.manifest_key("generation-lost"), protocol.head_key}

    def put(key, data, **conditions):
        stored = real_put(key, data, **conditions)
        if key in lose:
            lose.discard(key)
            raise ResponseLost("injected loss after the service applied the write")
        return stored

    store.put = put
    try:
        published = _publish(protocol, controller, worker, "generation-lost", 5)
    finally:
        del store.put
    _check(protocol.select_latest().chosen == published, "readback did not confirm publication")


def training_recovery(location: str, work: Path) -> None:
    """Real two-rank CPU training: fault, delete local checkpoints, restore from HEAD."""
    from trainguard import controller
    from trainguard.config import load_config
    from trainguard.validation import validate_runs

    raw = load_config(ROOT / "configs/cpu_demo.yaml").model_dump()
    raw["checkpoint"].update(mode="sync", interval_steps=1)
    raw["recovery"].update(max_restarts=3, progress_timeout_seconds=60)
    reference_config = work / "reference.json"
    reference_config.write_text(json.dumps({**raw, "checkpoint": {**raw["checkpoint"], "mode": "none"}}))
    reference, ok = controller.run(reference_config, work / "reference")
    _check(ok, "uninterrupted reference run failed")
    raw["fault"].update(kind="worker_exit", step=3, rank=0)
    faulted = work / "faulted.json"
    faulted.write_text(json.dumps(raw))
    recovered, ok = controller.run(
        faulted, work / "recovered", allow_experiment=True, reference_store_path=location,
    )
    _check(ok, "faulted run did not recover through the AWS reference store")
    shutil.rmtree(recovered / "checkpoints")
    _check(controller.resume(recovered), "completed run failed review after local loss")
    comparison = validate_runs(reference, recovered)
    _check(comparison["passed"], f"exact comparison failed: {comparison['differences']}")
    status = json.loads((recovered / "run.json").read_text())
    _check(status["post_run_audit"]["checkpoint_backend"] == "aws_reference_experiment",
           "audit does not name the AWS backend")


def cleanup(store, raw_s3, raw_dynamodb, run_ids: list[str]) -> None:
    for run_id in run_ids:
        prefix = f"{store.aws.prefix}/runs/{run_id}/"
        request = {"Bucket": store.aws.bucket, "Prefix": prefix}
        keys = []
        while True:
            page = raw_s3.list_objects_v2(**request)
            keys.extend(item["Key"] for item in page.get("Contents", ()))
            if not page.get("IsTruncated"):
                break
            request["ContinuationToken"] = page["NextContinuationToken"]
        for key in keys:
            raw_s3.delete_object(Bucket=store.aws.bucket, Key=key)
        # ponytail: deleting a smoke run's records is safe only because its ID is never reused.
        for name in ("HEAD", "AUTHORITY"):
            raw_dynamodb.delete_item(
                TableName=store.aws.table, Key={"pk": {"S": f"{prefix}{name}"}},
            )


def run_smoke(store, raw_s3, raw_dynamodb, *, training_location: str | None = None) -> dict:
    base = f"smoke-{uuid.uuid4().hex[:12]}"
    scenarios = {
        "mutable_revisions": lambda: mutable_revisions(store, f"{base}-rev"),
        "create_only_generation": lambda: create_only_generation(store, f"{base}-gen"),
        "concurrent_head_writers": lambda: concurrent_head_writers(store, f"{base}-cas"),
        "publish_fallback_and_takeover": lambda: publish_fallback_and_takeover(
            store, raw_s3, f"{base}-pub"
        ),
        "lost_ack_readback": lambda: lost_ack_readback(store, f"{base}-lost"),
    }
    results = {}
    run_ids = [f"{base}-{suffix}" for suffix in ("rev", "gen", "cas", "pub", "lost")]
    try:
        for name, scenario in scenarios.items():
            try:
                scenario()
            except Exception as exc:  # noqa: BLE001 - every scenario outcome is recorded
                results[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc(limit=3)}
            else:
                results[name] = {"status": "PASS"}
        if training_location is not None:
            with tempfile.TemporaryDirectory(prefix="trainguard-aws-smoke-") as temporary:
                try:
                    training_recovery(training_location, Path(temporary))
                except Exception as exc:  # noqa: BLE001
                    results["training_recovery"] = {
                        "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    results["training_recovery"] = {"status": "PASS"}
    finally:
        cleanup(store, raw_s3, raw_dynamodb, run_ids)
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "location": store.location,
        "smoke_run_prefix": base,
        "status": "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL",
        "scenarios": results,
        "scope": "named bucket/table/region only; no cross-host, power-loss or GPU claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--training", action="store_true",
                        help="also run a real two-rank CPU fault and HEAD restore")
    args = parser.parse_args()
    store = aws_store.connect(args.location)
    report = run_smoke(
        store, store.s3, store.dynamodb,
        training_location=args.location if args.training else None,
    )
    import boto3
    import botocore

    report["client_versions"] = {"boto3": boto3.__version__, "botocore": botocore.__version__}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "receipt": str(args.receipt)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
