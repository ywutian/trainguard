"""Shared strict parsing of training evidence."""

from __future__ import annotations

import json
from typing import Any

from trainguard.config import ProjectConfig


class EvidenceInvalid(ValueError):
    """Training evidence cannot support a completion claim."""


HASH_FIELDS = ("model_sha256", "optimizer_sha256", "scheduler_sha256")


def parse_event(line: str, run_id: str, attempt_id: str, rank: int) -> dict | None:
    try:
        event = json.loads(line)
    except ValueError as exc:
        raise EvidenceInvalid("invalid JSON") from exc
    if not isinstance(event, dict):
        raise EvidenceInvalid("event is not a mapping")
    if (event.get("run_id"), event.get("attempt_id"), event.get("rank")) != (
        run_id,
        attempt_id,
        rank,
    ):
        return None
    if type(event.get("rank")) is not int:
        raise EvidenceInvalid("invalid rank")
    if not isinstance(event.get("event_type"), str):
        raise EvidenceInvalid("invalid event type")
    if event["event_type"] in {"step_completed", "training_completed", "training_started"} and (
        type(event.get("global_step")) is not int or event["global_step"] < 0
    ):
        raise EvidenceInvalid("invalid step")
    return event


def summary_errors(
    summary: Any,
    config: ProjectConfig,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> list[str]:
    if not isinstance(summary, dict):
        return ["final summary is not a mapping"]
    errors = []
    for field in HASH_FIELDS:
        value = summary.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            errors.append(f"final {field} is missing or invalid")
    expected = {
        "global_step": config.training.total_steps,
        "world_size": config.run.world_size,
        "workload_fingerprint": config.workload_fingerprint(),
    }
    if run_id is not None:
        expected.update(run_id=run_id, config_fingerprint=config.fingerprint())
    if attempt_id is not None:
        expected["attempt_id"] = attempt_id
    for key, value in expected.items():
        if summary.get(key) != value or (type(value) is int and type(summary.get(key)) is not int):
            errors.append(f"final {key} differs from configured total or identity")
    return errors
