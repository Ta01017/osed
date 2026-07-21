import contextlib

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_osediff_focus_fusion import run_training_loop, simulate_optimizer_update_schedule


class CountingAccelerator:
    def __init__(self, accumulation):
        self.accumulation = accumulation
        self.micro = 0
        self.sync_gradients = False

    @contextlib.contextmanager
    def accumulate(self, model):
        yield
        self.micro += 1
        self.sync_gradients = self.micro % self.accumulation == 0

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        x = batch[0].float()
        return (self.weight * x).mean()


class CountingScheduler:
    def __init__(self):
        self.count = 0

    def step(self):
        self.count += 1


def test_run_training_loop_accumulation_four_max_steps_three():
    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler()
    loader = DataLoader(TensorDataset(torch.ones(20, 1)), batch_size=1)
    checkpoints, validations = [], []
    result = run_training_loop(
        accelerator=CountingAccelerator(4),
        model=model,
        train_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        max_train_steps=3,
        checkpointing_steps=2,
        validation_steps=3,
        compute_loss_fn=lambda m, b: m(b),
        checkpoint_fn=checkpoints.append,
        validation_fn=validations.append,
    )
    assert result["forward_micro_batches"] == 12
    assert result["backward_calls"] == 12
    assert result["optimizer_steps"] == 3
    assert scheduler.count == 3
    assert result["global_step"] == 3
    assert checkpoints == [2]
    assert validations == [3]


def test_run_training_loop_resume_counts_only_remaining_updates():
    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler()
    loader = DataLoader(TensorDataset(torch.ones(20, 1)), batch_size=1)
    result = run_training_loop(
        accelerator=CountingAccelerator(4),
        model=model,
        train_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        max_train_steps=3,
        global_step=1,
        compute_loss_fn=lambda m, b: m(b),
    )
    assert result["optimizer_steps"] == 2
    assert result["scheduler_steps"] == 2
    assert result["global_step"] == 3


def test_simulation_remains_auxiliary_only():
    s = simulate_optimizer_update_schedule(20, 4, 3)
    assert s["optimizer_steps"] == 3
