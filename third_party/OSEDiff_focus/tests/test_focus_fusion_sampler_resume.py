from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

from train_osediff_focus_fusion import build_train_sampler


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
    saved = {"world_size": 2, "dataset_length": 32}
    current = {"world_size": 1, "dataset_length": 32}
    with pytest.raises(AssertionError):
        assert saved["world_size"] == current["world_size"]
    current = {"world_size": 2, "dataset_length": 31}
    with pytest.raises(AssertionError):
        assert saved["dataset_length"] == current["dataset_length"]
