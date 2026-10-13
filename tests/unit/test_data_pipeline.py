"""Tests for the DataPipeline (sharding, upload, loading)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import TensorDataset

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.data_pipeline import DataPipeline
from p2p_ml_swarm.exceptions import ShardingError


@pytest.fixture
def config():
    return SwarmConfig(
        bootnode_addr="/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        run_id="test_run",
        bucket_url="https://test.r2.cloudflarestorage.com",
        bucket_name="test-bucket",
        shard_size_mb=1,
    )


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.get_shard_count.return_value = 3
    storage.download_shard_to_memory.return_value = {
        "data": torch.randn(32, 1, 28, 28),
        "labels": torch.randint(0, 10, (32,)),
    }
    return storage


@pytest.fixture
def small_dataset():
    """A small dataset for testing sharding."""
    data = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    return TensorDataset(data, labels)


class TestSharding:
    def test_shard_dataset_creates_files(self, config, mock_storage, small_dataset, tmp_path):
        pipeline = DataPipeline(config, mock_storage)
        shard_paths = pipeline.shard_dataset(small_dataset, tmp_path, shard_size_mb=1)
        assert len(shard_paths) > 0
        for p in shard_paths:
            assert p.exists()
            assert p.suffix == ".safetensors"

    def test_shard_dataset_empty_raises(self, config, mock_storage):
        empty_dataset = TensorDataset(torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long))
        pipeline = DataPipeline(config, mock_storage)
        with pytest.raises(ShardingError, match="empty dataset"):
            pipeline.shard_dataset(empty_dataset, "./output")

    def test_shard_files_are_loadable(self, config, mock_storage, small_dataset, tmp_path):
        pipeline = DataPipeline(config, mock_storage)
        shard_paths = pipeline.shard_dataset(small_dataset, tmp_path, shard_size_mb=1)
        data, labels = pipeline.load_shard_local(shard_paths[0])
        assert data.ndim == 4  # [N, C, H, W]
        assert labels.ndim == 1


class TestUploadAll:
    def test_upload_all_shards(self, config, mock_storage, small_dataset, tmp_path):
        pipeline = DataPipeline(config, mock_storage)
        pipeline.shard_dataset(small_dataset, tmp_path, shard_size_mb=1)
        count = pipeline.upload_all_shards(tmp_path)
        assert count > 0
        assert mock_storage.upload_shard.call_count == count

    def test_upload_empty_dir(self, config, mock_storage, tmp_path):
        pipeline = DataPipeline(config, mock_storage)
        count = pipeline.upload_all_shards(tmp_path)
        assert count == 0


class TestLoading:
    def test_get_shard_count(self, config, mock_storage):
        pipeline = DataPipeline(config, mock_storage)
        assert pipeline.get_shard_count() == 3

    def test_load_shard_from_cloud(self, config, mock_storage):
        pipeline = DataPipeline(config, mock_storage)
        data, labels = pipeline.load_shard(0)
        assert data.shape == (32, 1, 28, 28)
        assert labels.shape == (32,)

    def test_load_shard_missing_keys(self, config):
        mock_storage = MagicMock()
        mock_storage.download_shard_to_memory.return_value = {"wrong_key": torch.zeros(1)}
        pipeline = DataPipeline(config, mock_storage)
        with pytest.raises(ShardingError, match="missing required keys"):
            pipeline.load_shard(0)
