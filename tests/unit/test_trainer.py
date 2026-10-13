"""Tests for SwarmTrainer (training engine with mocked dependencies)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import NaNLossError, TrainingError
from p2p_ml_swarm.trainer import SwarmTrainer


@pytest.fixture
def config():
    return SwarmConfig(
        bootnode_addr="/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        run_id="test_run",
        bucket_url="https://test.r2.cloudflarestorage.com",
        bucket_name="test-bucket",
        batch_size_per_step=16,
        target_batch_size=64,
    )


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def mock_dht_manager():
    mgr = MagicMock()
    mgr.dht = MagicMock()
    # Simulate 3 shards available, then stop
    token_counter = iter([0, 1, 2, 3])
    mgr.checkout_token.side_effect = lambda: next(token_counter)
    mgr.is_batch_done.return_value = False
    mgr.get_peer_count.return_value = 2
    return mgr


@pytest.fixture
def mock_data_pipeline():
    pipeline = MagicMock()
    pipeline.get_shard_count.return_value = 3
    pipeline.load_shard.return_value = (
        torch.randn(32, 1, 28, 28),
        torch.randint(0, 10, (32,)),
    )
    return pipeline


@pytest.fixture
def mock_monitor():
    return MagicMock()


@pytest.fixture
def mock_checkpoint():
    mgr = MagicMock()
    mgr.enabled = False
    mgr.should_save.return_value = False
    return mgr


class TestTrainStep:
    def test_train_step_returns_loss(self, simple_model, config, mock_dht_manager,
                                     mock_data_pipeline, mock_monitor, mock_checkpoint):
        trainer = SwarmTrainer(
            model=simple_model, config=config, dht_manager=mock_dht_manager,
            data_pipeline=mock_data_pipeline, monitor=mock_monitor,
            checkpoint_mgr=mock_checkpoint, device="cpu",
        )
        # Manually set optimizer for standalone test
        trainer._optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        data = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        loss = trainer.train_step(data, labels)

        assert isinstance(loss, float)
        assert not math.isnan(loss)

    def test_nan_loss_raises(self, config, mock_dht_manager, mock_data_pipeline,
                              mock_monitor, mock_checkpoint):
        """A model that produces NaN should raise NaNLossError."""
        model = nn.Linear(10, 5)
        trainer = SwarmTrainer(
            model=model, config=config, dht_manager=mock_dht_manager,
            data_pipeline=mock_data_pipeline, monitor=mock_monitor,
            checkpoint_mgr=mock_checkpoint, device="cpu",
        )
        trainer._optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Verify normal forward works with correct label range
        data = torch.randn(4, 10)
        labels = torch.randint(0, 5, (4,))
        loss = trainer.train_step(data, labels)
        assert isinstance(loss, float)

        # Now force NaN loss by corrupting model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))
        data2 = torch.randn(4, 10)
        labels2 = torch.randint(0, 5, (4,))
        with pytest.raises(NaNLossError):
            trainer.train_step(data2, labels2)


class TestTrainEpoch:
    def test_train_epoch_processes_shards(self, simple_model, config, mock_dht_manager,
                                          mock_data_pipeline, mock_monitor, mock_checkpoint):
        trainer = SwarmTrainer(
            model=simple_model, config=config, dht_manager=mock_dht_manager,
            data_pipeline=mock_data_pipeline, monitor=mock_monitor,
            checkpoint_mgr=mock_checkpoint, device="cpu",
        )
        trainer._optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        avg_loss = trainer.train_epoch(epoch=0)

        assert isinstance(avg_loss, float)
        assert mock_dht_manager.checkout_token.call_count >= 1
        assert mock_dht_manager.mark_complete.call_count >= 1

    def test_skips_already_done_shards(self, simple_model, config, mock_data_pipeline,
                                       mock_monitor, mock_checkpoint):
        dht = MagicMock()
        dht.dht = MagicMock()
        token_counter = iter([0, 1, 2])
        dht.checkout_token.side_effect = lambda: next(token_counter)
        dht.is_batch_done.side_effect = [True, False, False]  # First shard already done
        dht.get_peer_count.return_value = 1
        dht.mark_complete.return_value = None

        mock_data_pipeline.get_shard_count.return_value = 2

        trainer = SwarmTrainer(
            model=simple_model, config=config, dht_manager=dht,
            data_pipeline=mock_data_pipeline, monitor=mock_monitor,
            checkpoint_mgr=mock_checkpoint, device="cpu",
        )
        trainer._optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        trainer.train_epoch(epoch=0)

        # Shard 0 was skipped (done), shard 1 processed
        assert dht.mark_complete.call_count == 1


class TestSetup:
    @patch("p2p_ml_swarm.trainer.hivemind", None)
    def test_setup_without_hivemind(self, simple_model, config, mock_dht_manager,
                                     mock_data_pipeline, mock_monitor, mock_checkpoint):
        trainer = SwarmTrainer(
            model=simple_model, config=config, dht_manager=mock_dht_manager,
            data_pipeline=mock_data_pipeline, monitor=mock_monitor,
            checkpoint_mgr=mock_checkpoint, device="cpu",
        )
        trainer.setup()
        assert trainer._optimizer is not None

    def test_train_without_setup_raises(self, simple_model, config, mock_dht_manager,
                                         mock_data_pipeline, mock_monitor, mock_checkpoint):
        trainer = SwarmTrainer(
            model=simple_model, config=config, dht_manager=mock_dht_manager,
            data_pipeline=mock_data_pipeline, monitor=mock_monitor,
            checkpoint_mgr=mock_checkpoint, device="cpu",
        )
        with pytest.raises(TrainingError, match="not set up"):
            trainer.train(1)
