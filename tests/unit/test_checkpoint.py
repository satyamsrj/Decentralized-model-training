"""Tests for CheckpointManager (save, load, restore, cleanup)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from p2p_ml_swarm.checkpoint import CheckpointManager
from p2p_ml_swarm.exceptions import CheckpointError


@pytest.fixture
def simple_model():
    return nn.Linear(10, 5)


@pytest.fixture
def optimizer(simple_model):
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)


class TestCheckpointSave:
    def test_save_creates_file(self, simple_model, optimizer, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path, save_every_n_steps=1)
        path = mgr.save(simple_model, optimizer, epoch=0, batch_id=0)
        assert path.exists()
        assert "epoch0000_batch000000" in path.name

    def test_save_disabled_raises(self, simple_model, optimizer):
        mgr = CheckpointManager(checkpoint_dir=None)
        with pytest.raises(CheckpointError, match="disabled"):
            mgr.save(simple_model, optimizer, epoch=0, batch_id=0)


class TestCheckpointLoad:
    def test_load_restores_state(self, simple_model, optimizer, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        mgr.save(simple_model, optimizer, epoch=2, batch_id=42, loss=0.5)

        ckpt = mgr.load(mgr.find_latest())
        assert ckpt["epoch"] == 2
        assert ckpt["batch_id"] == 42
        assert ckpt["loss"] == 0.5
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt

    def test_load_missing_file(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(CheckpointError, match="not found"):
            mgr.load(tmp_path / "nonexistent.pt")


class TestCheckpointRestore:
    def test_restore_model(self, simple_model, optimizer, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        original_weight = simple_model.weight.data.clone()

        mgr.save(simple_model, optimizer, epoch=1, batch_id=10)

        # Corrupt model weights
        simple_model.weight.data.zero_()
        assert not torch.equal(simple_model.weight.data, original_weight)

        # Restore
        mgr.restore(simple_model, optimizer)
        assert torch.equal(simple_model.weight.data, original_weight)


class TestFindLatest:
    def test_find_latest_empty_dir(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        assert mgr.find_latest() is None

    def test_find_latest_returns_most_recent(self, simple_model, optimizer, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        mgr.save(simple_model, optimizer, epoch=0, batch_id=0)
        mgr.save(simple_model, optimizer, epoch=1, batch_id=50)
        mgr.save(simple_model, optimizer, epoch=2, batch_id=100)

        latest = mgr.find_latest()
        assert "epoch0002" in latest.name


class TestCleanup:
    def test_cleanup_keeps_last_n(self, simple_model, optimizer, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        for i in range(6):
            mgr.save(simple_model, optimizer, epoch=i, batch_id=0)

        # Should keep at most 3
        checkpoints = list(tmp_path.glob("checkpoint_*.pt"))
        assert len(checkpoints) <= 3


class TestShouldSave:
    def test_should_save_frequency(self):
        mgr = CheckpointManager(checkpoint_dir="./dummy", save_every_n_steps=5)
        results = [mgr.should_save() for _ in range(10)]
        assert results == [False, False, False, False, True, False, False, False, False, True]

    def test_should_save_disabled(self):
        mgr = CheckpointManager(checkpoint_dir=None)
        assert mgr.should_save() is False
