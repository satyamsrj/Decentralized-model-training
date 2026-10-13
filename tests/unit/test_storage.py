"""Tests for the StorageClient (S3/R2 shard management)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from botocore.exceptions import ClientError

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import ShardDownloadError, ShardUploadError, StorageError
from p2p_ml_swarm.storage import StorageClient


@pytest.fixture
def config():
    return SwarmConfig(
        bootnode_addr="/ip4/1.2.3.4/tcp/9000/p2p/QmTest",
        run_id="test_run",
        bucket_url="https://test.r2.cloudflarestorage.com",
        bucket_name="test-bucket",
        s3_endpoint_url="https://test.r2.cloudflarestorage.com",
        s3_access_key="test-key",
        s3_secret_key="test-secret",
    )


@pytest.fixture
def mock_s3():
    with patch("p2p_ml_swarm.storage.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client


class TestStorageClientInit:
    def test_creates_boto3_client(self, config, mock_s3):
        client = StorageClient(config)
        assert client._bucket == "test-bucket"


class TestUpload:
    def test_upload_shard_success(self, config, mock_s3, tmp_path):
        shard_file = tmp_path / "batch_0001.safetensors"
        shard_file.write_bytes(b"fake shard data")

        client = StorageClient(config)
        client.upload_shard(shard_file, "shards/batch_0001.safetensors")

        mock_s3.upload_file.assert_called_once()

    def test_upload_shard_file_not_found(self, config, mock_s3):
        client = StorageClient(config)
        with pytest.raises(ShardUploadError, match="not found"):
            client.upload_shard("/nonexistent/file.safetensors", "key")

    def test_upload_shard_client_error(self, config, mock_s3, tmp_path):
        shard_file = tmp_path / "batch.safetensors"
        shard_file.write_bytes(b"data")
        mock_s3.upload_file.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "upload_file"
        )
        client = StorageClient(config)
        with pytest.raises(ShardUploadError, match="Upload failed"):
            client.upload_shard(shard_file, "key")

    def test_upload_shard_bytes(self, config, mock_s3):
        client = StorageClient(config)
        client.upload_shard_bytes(b"raw data", "shards/test.safetensors")
        mock_s3.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="shards/test.safetensors", Body=b"raw data"
        )


class TestDownload:
    def test_download_to_memory_client_error(self, config, mock_s3):
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "get_object"
        )
        client = StorageClient(config)
        with pytest.raises(ShardDownloadError, match="Download failed"):
            client.download_shard_to_memory("missing_key")

    def test_download_to_file(self, config, mock_s3, tmp_path):
        client = StorageClient(config)
        local_path = tmp_path / "downloaded.safetensors"
        client.download_shard_to_file("shards/batch_0001.safetensors", local_path)
        mock_s3.download_file.assert_called_once()


class TestListing:
    def test_list_shards(self, config, mock_s3):
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [
                {"Key": "shards/batch_0000.safetensors"},
                {"Key": "shards/batch_0001.safetensors"},
            ]},
        ]
        mock_s3.get_paginator.return_value = paginator

        client = StorageClient(config)
        keys = client.list_shards("shards/")

        assert len(keys) == 2
        assert keys[0] == "shards/batch_0000.safetensors"

    def test_shard_exists_true(self, config, mock_s3):
        mock_s3.head_object.return_value = {}
        client = StorageClient(config)
        assert client.shard_exists("shards/batch_0000.safetensors") is True

    def test_shard_exists_false(self, config, mock_s3):
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "head_object"
        )
        client = StorageClient(config)
        assert client.shard_exists("shards/missing.safetensors") is False
