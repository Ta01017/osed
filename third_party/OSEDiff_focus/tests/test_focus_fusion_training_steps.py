import contextlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_osediff_focus_fusion import TrainingProgress, run_training_loop, simulate_optimizer_update_schedule, validate_training_progress


class CountingAccelerator:
    def __init__(self, accumulation):
        self.accumulation = accumulation
        self.micro = 0
        self.sync_gradients = False

    @contextlib.contextmanager
    def accumulate(self, model):
        self.sync_gradients = (self.micro + 1) % self.accumulation == 0
        yield
        self.micro += 1

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)


class TailSyncAccelerator(CountingAccelerator):
    def __init__(self, accumulation, dataloader_length):
        super().__init__(accumulation)
        self.dataloader_length = dataloader_length

    @contextlib.contextmanager
    def accumulate(self, model):
        is_full_step = (self.micro + 1) % self.accumulation == 0
        is_epoch_tail = (self.micro + 1) == self.dataloader_length
        self.sync_gradients = is_full_step or is_epoch_tail
        yield
        self.micro += 1


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
        train_sampler=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        progress=TrainingProgress(),
        gradient_accumulation_steps=4,
        max_train_steps=3,
        checkpointing_steps=2,
        validation_steps=3,
        compute_loss_fn=lambda m, b: m(b),
        checkpoint_fn=lambda p, losses: checkpoints.append(p.global_step),
        validation_fn=lambda p: validations.append(p.global_step),
    )
    assert result.micro_batches == 12
    assert result.optimizer_updates == 3
    assert scheduler.count == 3
    assert result.global_step == 3
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
        train_sampler=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        max_train_steps=3,
        progress=TrainingProgress(global_step=1, optimizer_updates=1, scheduler_steps=1, micro_batches=4),
        gradient_accumulation_steps=4,
        compute_loss_fn=lambda m, b: m(b),
    )
    assert result.optimizer_updates == 2
    assert result.scheduler_steps == 2
    assert result.global_step == 3


def test_simulation_remains_auxiliary_only():
    s = simulate_optimizer_update_schedule(20, 4, 3)
    assert s["optimizer_steps"] == 3


def test_training_progress_from_trainer_state_restores_all_fields():
    state = {"global_step": 2, "current_epoch": 1, "completed_epochs": 1, "batches_consumed_in_current_epoch": 3,
             "micro_batches": 8, "optimizer_updates": 2, "scheduler_steps": 2, "sampler_epoch": 1}
    p = TrainingProgress.from_trainer_state(state)
    assert p.to_trainer_state_fields() == state


def test_accumulation_four_continuous_resume_4_8_12():
    loader = DataLoader(TensorDataset(torch.ones(20, 1)), batch_size=1)
    progress = TrainingProgress()
    for target, expected_micro in [(1, 4), (2, 8), (3, 12)]:
        model = Tiny()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = CountingScheduler()
        progress = run_training_loop(
            accelerator=CountingAccelerator(4),
            model=model,
            train_dataloader=loader,
            train_sampler=None,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            progress=progress,
            gradient_accumulation_steps=4,
            max_train_steps=target,
            compute_loss_fn=lambda m, b: m(b),
        )
        assert progress.global_step == target
        assert progress.optimizer_updates == target
        assert progress.scheduler_steps == target
        assert progress.micro_batches == expected_micro
        validate_training_progress(progress, gradient_accumulation_steps=4, dataloader_length=len(loader))


def test_partial_epoch_accumulation_uses_sync_events_not_formula():
    loader = DataLoader(TensorDataset(torch.ones(6, 1)), batch_size=1)
    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler()
    events = []
    progress = run_training_loop(
        accelerator=TailSyncAccelerator(4, len(loader)),
        model=model,
        train_dataloader=loader,
        train_sampler=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        progress=TrainingProgress(),
        gradient_accumulation_steps=4,
        max_train_steps=2,
        compute_loss_fn=lambda m, b: m(b),
        event_callback=events.append,
    )
    sync_events = [e for e in events if e["event"] == "micro_batch" and e["sync_gradients"]]
    assert [e["batch_index"] for e in sync_events] == [3, 5]
    assert progress.micro_batches == 6
    assert progress.global_step == len(sync_events)
    assert progress.optimizer_updates == len(sync_events)
    assert progress.scheduler_steps == len(sync_events)
    assert progress.current_epoch == 1
    assert progress.completed_epochs == 1
    assert progress.batches_consumed_in_current_epoch == 0
    validate_training_progress(progress, gradient_accumulation_steps=4, dataloader_length=len(loader))


def test_validate_training_progress_rejects_counter_mismatch():
    with pytest.raises(ValueError, match="optimizer_updates"):
        validate_training_progress(TrainingProgress(global_step=2, optimizer_updates=1, scheduler_steps=2), gradient_accumulation_steps=1)


def test_validate_training_progress_accepts_legal_partial_accumulation_state():
    progress = TrainingProgress(
        global_step=2,
        current_epoch=1,
        completed_epochs=1,
        batches_consumed_in_current_epoch=0,
        micro_batches=6,
        optimizer_updates=2,
        scheduler_steps=2,
        sampler_epoch=1,
    )
    validate_training_progress(progress, gradient_accumulation_steps=4, dataloader_length=6)


def test_production_code_does_not_assume_full_accumulation_per_update():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for rel in ("train_osediff_focus_fusion.py", "osediff_focus_fusion.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "micro_batches >=\n            global_step *" not in text
        assert "micro_batches ==\n            global_step *" not in text
        assert "expected_min_micro_batches" not in text
