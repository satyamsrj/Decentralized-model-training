"""Cloud storage client for S3/R2 shard management."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from safetensors.torch import load as safetensors_load, save as safetensors_save
from tqdm import tqdm

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import ShardDownloadError, ShardUploadError, StorageError

logger = logging.getLogger(__name__)


class StorageClient:
    """Manages shard uploads and downloads to S3-compatible object storage (S3, R2, MinIO)."""

    def __init__(self, config: SwarmConfig) -> None:
        self._config = config
        self._bucket = config.bucket_name

        session_kwargs: dict = {}
        if config.s3_access_key and config.s3_secret_key:
            session_kwargs["aws_access_key_id"] = config.s3_access_key
            session_kwargs["aws_secret_access_key"] = config.s3_secret_key

        client_kwargs: dict = {"config": BotoConfig(max_pool_connections=10)}
        if config.s3_endpoint_url:
            client_kwargs["endpoint_url"] = config.s3_endpoint_url
        if config.s3_region and config.s3_region != "auto":
            client_kwargs["region_name"] = config.s3_region

        self._s3 = boto3.client("s3", **session_kwargs, **client_kwargs)
        logger.info(
            f"StorageClient initialized for bucket={self._bucket} "
            f"endpoint={config.s3_endpoint_url or 'default AWS'}"
        )

    # ── Upload ──────────────────────────────────────────────────────────

    def upload_shard(self, local_path: Path | str, remote_key: str) -> None:
        """Upload a local safetensors file to the bucket.

        Args:
            local_path: Path to the local .safetensors file.
            remote_key: Object key in the bucket (e.g. 'shards/batch_0001.safetensors').

        Raises:
            ShardUploadError: If the upload fails.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise ShardUploadError(f"Local file not found: {local_path}")

        file_size = local_path.stat().st_size
        logger.info(f"Uploading {local_path.name} ({file_size / 1e6:.1f} MB) → s3://{self._bucket}/{remote_key}")

        try:
            self._s3.upload_file(
                str(local_path),
                self._bucket,
                remote_key,
                Callback=_UploadProgress(local_path.name, file_size),
            )
        except ClientError as exc:
            raise ShardUploadError(f"Upload failed for {remote_key}: {exc}") from exc

    def upload_shard_bytes(self, data: bytes, remote_key: str) -> None:
        """Upload raw bytes (already serialized safetensor) to the bucket."""
        logger.info(f"Uploading {len(data) / 1e6:.1f} MB → s3://{self._bucket}/{remote_key}")
        try:
            self._s3.put_object(Bucket=self._bucket, Key=remote_key, Body=data)
        except ClientError as exc:
            raise ShardUploadError(f"Upload failed for {remote_key}: {exc}") from exc

    # ── Download ────────────────────────────────────────────────────────

    def download_shard_to_memory(self, remote_key: str) -> dict:
        """Stream a safetensors shard from cloud storage directly into CPU tensors.

        Returns:
            dict mapping tensor names to torch.Tensor objects.

        Raises:
            ShardDownloadError: If the download or deserialization fails.
        """
        logger.debug(f"Downloading shard s3://{self._bucket}/{remote_key} → memory")
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=remote_key)
            raw_bytes = response["Body"].read()
        except ClientError as exc:
            raise ShardDownloadError(f"Download failed for {remote_key}: {exc}") from exc

        try:
            tensors = safetensors_load(raw_bytes)
        except Exception as exc:
            raise ShardDownloadError(f"Failed to deserialize {remote_key}: {exc}") from exc

        logger.debug(f"Loaded {len(tensors)} tensors from {remote_key}")
        return tensors

    def download_shard_to_file(self, remote_key: str, local_path: Path | str) -> Path:
        """Download a shard to a local file."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self._bucket, remote_key, str(local_path))
        except ClientError as exc:
            raise ShardDownloadError(f"Download failed for {remote_key}: {exc}") from exc
        return local_path

    # ── Listing / Queries ───────────────────────────────────────────────

    def list_shards(self, prefix: str = "shards/") -> list[str]:
        """List all shard keys under a prefix in the bucket.

        Returns:
            Sorted list of object keys matching the prefix.
        """
        keys: list[str] = []
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".safetensors"):
                        keys.append(obj["Key"])
        except ClientError as exc:
            raise StorageError(f"Failed to list shards: {exc}") from exc

        keys.sort()
        logger.info(f"Found {len(keys)} shards under prefix '{prefix}'")
        return keys

    def shard_exists(self, remote_key: str) -> bool:
        """Check if a specific shard exists in the bucket."""
        try:
            self._s3.head_object(Bucket=self._bucket, Key=remote_key)
            return True
        except ClientError:
            return False

    def get_shard_count(self, prefix: str = "shards/") -> int:
        """Return the total number of shards available."""
        return len(self.list_shards(prefix))


class _UploadProgress:
    """Callback for tqdm progress bar during uploads."""

    def __init__(self, filename: str, total_size: int) -> None:
        self._bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"↑ {filename}")

    def __call__(self, bytes_transferred: int) -> None:
        self._bar.update(bytes_transferred)
