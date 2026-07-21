from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

from train_osediff_focus_fusion import build_train_sampler, validate_sampler_resume_state


class IndexDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


class Acc:
    def __init__(self, rank=0, world=1):
        self.process_index = rank
        self.num_processes = world


def _order(n=32, rank=0, world=1, epoch=0, seed=123):
    sampler = build_train_sampler(IndexDataset(n), Acc(rank, world), SimpleNamespace(seed=seed), epoch)
    return list(iter(sampler))


def test_single_rank_resume_remaining_order_matches_uninterrupted():
    full = _order()
    consumed = 9
    resumed = _order()[consumed:]
    assert full[consumed:] == resumed


def test_two_rank_orders_are_deterministic():
    assert _order(rank=0, world=2) == _order(rank=0, world=2)
    assert _order(rank=1, world=2) == _order(rank=1, world=2)
    assert set(_order(rank=0, world=2)).isdisjoint(set(_order(rank=1, world=2)))


def test_epoch_boundary_changes_order_but_is_repeatable():
    assert _order(epoch=0) != _order(epoch=1)
    assert _order(epoch=1) == _order(epoch=1)


def test_resume_metadata_rejects_world_size_or_dataset_length_change():
    saved = {
        "world_size": 2,
        "dataset_length": 32,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sync_with_dataloader": True,
        "sampler_seed": 123,
        "drop_last": False,
        "sampler_epoch": 0,
        "batches_consumed_in_current_epoch": 0,
    }
    with pytest.raises(ValueError, match="world_size"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=1, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=False)
    with pytest.raises(ValueError, match="dataset_length"):
        validate_sampler_resume_state(saved, dataset_length=31, world_size=2, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=False)
    with pytest.raises(ValueError, match="sampler_seed"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=2, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=999, drop_last=False)
    with pytest.raises(ValueError, match="drop_last"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=2, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=True)


def test_validate_sampler_resume_state_rejects_batch_position_overflow():
    saved = {
        "world_size": 1,
        "dataset_length": 32,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sync_with_dataloader": True,
        "sampler_seed": 123,
        "drop_last": False,
        "sampler_epoch": 0,
        "batches_consumed_in_current_epoch": 99,
    }
    with pytest.raises(ValueError, match="batches_consumed_in_current_epoch"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=1, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=False,
                                      dataloader_length=32)


def test_validate_sampler_resume_state_rejects_progress_counter_mismatches():
    saved = {
        "trainer_state_version": 4,
        "world_size": 1,
        "dataset_length": 32,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sync_with_dataloader": True,
        "sampler_seed": 123,
        "drop_last": False,
        "current_epoch": 0,
        "sampler_epoch": 0,
        "completed_epochs": 0,
        "batches_consumed_in_current_epoch": 0,
        "global_step": 2,
        "optimizer_updates": 1,
        "scheduler_steps": 2,
        "micro_batches": 2,
    }
    with pytest.raises(ValueError, match="optimizer_updates"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=1, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=False)
    saved["optimizer_updates"] = 2
    saved["scheduler_steps"] = 1
    with pytest.raises(ValueError, match="scheduler_steps"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=1, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=False)
    saved["scheduler_steps"] = 2
    saved["micro_batches"] = 1
    with pytest.raises(ValueError, match="micro_batches"):
        validate_sampler_resume_state(saved, dataset_length=32, world_size=1, train_batch_size=1,
                                      gradient_accumulation_steps=1, sampler_seed=123, drop_last=False)


def test_validate_sampler_resume_state_accepts_epoch_length_six_partial_accumulation():
    saved = {
        "trainer_state_version": 4,
        "world_size": 1,
        "dataset_length": 6,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "sync_with_dataloader": True,
        "sampler_seed": 123,
        "drop_last": False,
        "current_epoch": 1,
        "sampler_epoch": 1,
        "completed_epochs": 1,
        "batches_consumed_in_current_epoch": 0,
        "global_step": 2,
        "optimizer_updates": 2,
        "scheduler_steps": 2,
        "micro_batches": 6,
    }
    validate_sampler_resume_state(
        saved,
        dataset_length=6,
        world_size=1,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        sync_with_dataloader=True,
        sampler_seed=123,
        drop_last=False,
        dataloader_length=6,
    )


def test_validate_sampler_resume_state_rejects_version4_epoch_tail_position():
    saved = {
        "trainer_state_version": 4,
        "world_size": 1,
        "dataset_length": 6,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "sync_with_dataloader": True,
        "sampler_seed": 123,
        "drop_last": False,
        "current_epoch": 0,
        "sampler_epoch": 0,
        "completed_epochs": 0,
        "batches_consumed_in_current_epoch": 6,
        "global_step": 2,
        "optimizer_updates": 2,
        "scheduler_steps": 2,
        "micro_batches": 6,
    }
    with pytest.raises(ValueError, match="version 4"):
        validate_sampler_resume_state(
            saved,
            dataset_length=6,
            world_size=1,
            train_batch_size=1,
            gradient_accumulation_steps=4,
            sync_with_dataloader=True,
            sampler_seed=123,
            drop_last=False,
            dataloader_length=6,
        )
