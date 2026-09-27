"""Explicit checkpoint interval estimates under a sparse-failure approximation."""

from __future__ import annotations

import math


def suggest_interval(
    *,
    overhead_seconds: float,
    mtbf_seconds: float,
    commit_lag_seconds: float,
    rto_seconds: float,
    rollback_budget_seconds: float,
    upload_seconds: float,
) -> dict:
    values = (
        overhead_seconds,
        mtbf_seconds,
        commit_lag_seconds,
        rto_seconds,
        rollback_budget_seconds,
        upload_seconds,
    )
    if (
        any(not math.isfinite(value) or value < 0 for value in values)
        or mtbf_seconds <= 0
        or rollback_budget_seconds <= 0
    ):
        raise ValueError(
            "costs must be finite and non-negative; MTBF and rollback budget must be positive"
        )
    limit = rollback_budget_seconds - commit_lag_seconds
    lower = max(upload_seconds, 1e-9)
    if limit < lower:
        return {
            "feasible": False,
            "interval_seconds": None,
            "reason": "upload duration or commit lag exceeds the rollback budget",
        }
    proposed = math.sqrt(2 * overhead_seconds * mtbf_seconds)
    interval = min(limit, max(lower, proposed))
    waste = (
        overhead_seconds / interval
        + (interval / 2 + commit_lag_seconds + rto_seconds) / mtbf_seconds
    )
    return {
        "feasible": True,
        "interval_seconds": interval,
        "estimated_waste_fraction": waste,
        "unconstrained_interval_seconds": proposed,
        "assumptions": "job-level sparse independent failures; fixed measured costs; one upload in flight",
        "applied_to_training": False,
    }
