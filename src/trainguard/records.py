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
    if type(summary.get("state_schema_version")) is not int or summary["state_schema_version"] != 2:
        errors.append("final summary state schema is unsupported or missing")
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
    if summary.get("state_schema_version") == 2:
        updates, consumed = summary.get("optimizer_updates"), summary.get("consumed_batches")
        if type(updates) is not int or updates != config.training.total_steps:
            errors.append("final optimizer update count differs")
        if (
            type(consumed) is not int
            or consumed < config.training.total_steps * config.training.gradient_accumulation_steps
        ):
            errors.append("final consumed batch count differs")
        rank_states = summary.get("rank_states")
        if not isinstance(rank_states, list) or len(rank_states) != config.run.world_size:
            errors.append("final rank states are incomplete")
        else:
            from trainguard.strategy import state_digest

            external_v2 = config.external_workload is not None and config.external_workload.version == 2
            hash_fields = (*HASH_FIELDS, "scaler_sha256") + (
                ("stream_sha256", "extra_sha256") if external_v2 else ()
            )
            for field in hash_fields:
                values = [
                    item.get(field) if isinstance(item, dict) else None for item in rank_states
                ]
                valid_values = all(
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                    for value in values
                )
                if not valid_values or state_digest(values) != summary.get(field):
                    errors.append(f"final combined rank {field} differs")
                if valid_values and field not in {"stream_sha256", "extra_sha256"} and (
                    config.run.strategy == "ddp" and len(set(values)) > 1
                ):
                    errors.append(f"final DDP rank {field} differs")
            for item in rank_states:
                if (
                    not isinstance(item, dict)
                    or item.get("optimizer_updates") != updates
                    or item.get("consumed_batches") != consumed
                ):
                    errors.append("final rank counters differ")
    return errors
