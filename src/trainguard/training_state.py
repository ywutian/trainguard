"""Recovery state at a complete gradient accumulation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingState:
    optimizer_updates: int = 0
    consumed_batches: int = 0
    scaler: Any = None

    def snapshot(self) -> dict:
        return {
            "state_schema_version": 2,
            "optimizer_updates": self.optimizer_updates,
            "consumed_batches": self.consumed_batches,
            "accumulation_phase": 0,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }

    def restore(self, local: dict) -> None:
        self.optimizer_updates = local["optimizer_updates"]
        self.consumed_batches = local["consumed_batches"]
        if self.scaler is not None:
            self.scaler.load_state_dict(local["scaler"])


def complete_update(model, optimizer, scheduler, state: TrainingState, control_group) -> bool:
    import torch
    import torch.distributed as dist

    scaler = state.scaler
    if scaler is not None:
        scaler.unscale_(optimizer)
        invalid = any(
            not bool(
                torch.isfinite(
                    parameter.grad.to_local()
                    if hasattr(parameter.grad, "to_local")
                    else parameter.grad
                ).all()
            )
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        flag = torch.tensor([int(invalid)], dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=control_group)
        if flag.item():
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
            saved = scaler.state_dict()
            saved["_growth_tracker"] = 0
            scaler.load_state_dict(saved)
            return False
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    state.optimizer_updates += 1
    return True
