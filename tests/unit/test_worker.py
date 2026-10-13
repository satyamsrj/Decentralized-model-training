"""Tests for SwarmWorker (lifecycle orchestrator)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn
import yaml

from p2p_ml_swarm.worker import SwarmWorker


@pytest.fixture
def config_file(tmp_path):
    data = {
        "bootnode_addr": "/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        "run_id": "test_run",
        "bucket_url": "https://test.r2.cloudflarestorage.com",
        "bucket_name": "test-bucket",
        "batch_size_per_step": 16,
        "target_batch_size": 64,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


class TestWorkerInit:
    def test_init_with_model(self, config_file, simple_model):
        worker = SwarmWorker(config_path=config_file, model=simple_model, num_epochs=1)
        assert worker._num_epochs == 1

    def test_init_without_model(self, config_file):
        worker = SwarmWorker(config_path=config_file, model=None, num_epochs=1)
        assert worker._model is None


class TestWorkerRun:
    @patch("p2p_ml_swarm.worker.SwarmTrainer")
    @patch("p2p_ml_swarm.worker.DataPipeline")
    @patch("p2p_ml_swarm.worker.StorageClient")
    @patch("p2p_ml_swarm.worker.DHTManager")
    def test_run_wires_components(self, MockDHT, MockStorage, MockPipeline,
                                   MockTrainer, config_file, simple_model):
        # Setup mocks
        mock_dht = MockDHT.return_value
        mock_dht.connect.return_value = MagicMock()
        mock_dht.get_peer_count.return_value = 2

        mock_pipeline = MockPipeline.return_value
        mock_pipeline.get_shard_count.return_value = 5

        mock_trainer = MockTrainer.return_value
        mock_trainer.setup.return_value = None
        mock_trainer.train.return_value = None

        worker = SwarmWorker(config_path=config_file, model=simple_model, num_epochs=1)
        worker.run()

        # Verify the lifecycle was called
        mock_dht.connect.assert_called_once()
        mock_trainer.setup.assert_called_once()
        mock_trainer.train.assert_called_once_with(1)
        mock_dht.shutdown.assert_called_once()

    def test_run_no_model_raises(self, config_file):
        worker = SwarmWorker(config_path=config_file, model=None, num_epochs=1)
        with pytest.raises(ValueError, match="No model provided"):
            worker.run()
