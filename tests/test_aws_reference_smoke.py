"""The live AWS smoke script's scenarios, exercised against service doubles."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from aws_fakes import FakeDynamoDB, FakeS3, FakeTransportError

from trainguard import aws_store

LOCATION = "aws://trainguard-test/smoke?region=us-east-1&table=trainguard-heads"


def _module():
    source = Path(__file__).parents[1] / "scripts" / "aws_reference_smoke.py"
    spec = importlib.util.spec_from_file_location("aws_reference_smoke", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_scenarios_pass_and_clean_up_only_their_run(monkeypatch: pytest.MonkeyPatch) -> None:
    s3, dynamodb = FakeS3(), FakeDynamoDB()

    def connect(value: str, *, read_only: bool = False) -> aws_store.AwsObjectStore:
        return aws_store.AwsObjectStore(
            s3, dynamodb, aws_store.parse_location(value), read_only=read_only,
            transport_errors=(FakeTransportError,),
        )

    monkeypatch.setattr(aws_store, "connect", connect)
    s3.objects["smoke/unrelated"] = b"keep"
    store = connect(LOCATION)
    report = _module().run_smoke(store, s3, dynamodb, training_location=LOCATION)
    assert report["status"] == "PASS", report["scenarios"]
    assert set(report["scenarios"]) == {
        "mutable_revisions", "create_only_generation", "concurrent_head_writers",
        "publish_fallback_and_takeover", "lost_ack_readback", "training_recovery",
    }
    probe_prefix = f"smoke/runs/{report['smoke_run_prefix']}-"
    assert not any(key.startswith(probe_prefix) for key in s3.objects)
    assert not any(key.startswith(probe_prefix) for key in dynamodb.items)
    assert s3.objects["smoke/unrelated"] == b"keep"


def test_smoke_reports_a_broken_service_as_failure() -> None:
    s3, dynamodb = FakeS3(), FakeDynamoDB()
    store = aws_store.AwsObjectStore(
        s3, dynamodb, aws_store.parse_location(LOCATION), transport_errors=(FakeTransportError,),
    )
    original = dynamodb.put_item

    def ignores_conditions(**kwargs):
        kwargs["ConditionExpression"] = "attribute_not_exists(pk)"
        dynamodb.items.pop(kwargs["Item"]["pk"]["S"], None)
        return original(**kwargs)

    dynamodb.put_item = ignores_conditions
    report = _module().run_smoke(store, s3, dynamodb)
    assert report["status"] == "FAIL"
    assert report["scenarios"]["mutable_revisions"]["status"] == "FAIL"
    assert report["scenarios"]["concurrent_head_writers"]["status"] == "FAIL"
