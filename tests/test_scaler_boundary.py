import torch
import torch.distributed as dist

from trainguard.training_state import TrainingState, complete_update


def test_nonfinite_gradient_does_not_advance_optimizer_or_scheduler(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    try:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu")
        state = TrainingState(scaler=scaler)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        scaler.scale(model(torch.ones(1, 1)).sum() * float("inf")).backward()
        assert not complete_update(model, optimizer, scheduler, state, dist.group.WORLD)
        assert state.optimizer_updates == 0
        assert scheduler.last_epoch == 0
        assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        optimizer.zero_grad()
        scaler.scale(model(torch.ones(1, 1)).sum()).backward()
        assert complete_update(model, optimizer, scheduler, state, dist.group.WORLD)
        assert state.optimizer_updates == 1
        assert scheduler.last_epoch == 1
        assert scaler.get_scale() == 32768
    finally:
        dist.destroy_process_group()
