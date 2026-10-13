"""Data pipeline: chunk datasets into safetensor shards and manage cloud storage."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.exceptions import ShardingError
from p2p_ml_swarm.storage import StorageClient

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles dataset sharding, cloud upload, and on-demand shard loading."""

    def __init__(self, config: SwarmConfig, storage: StorageClient) -> None:
        self._config = config
        self._storage = storage
        self._shard_prefix = f"shards/{config.run_id}/"

    # ── Sharding ────────────────────────────────────────────────────────

    def shard_dataset(
        self,
        dataset: Dataset,
        output_dir: str | Path,
        shard_size_mb: Optional[int] = None,
    ) -> list[Path]:
        """Slice a PyTorch dataset into fixed-size safetensor shard files.

        Each shard contains:
          - "data":   stacked input tensors  [N, ...]
          - "labels": stacked target tensors  [N]

        Args:
            dataset: A PyTorch dataset that returns (input_tensor, label) tuples.
            output_dir: Local directory to write shard files.
            shard_size_mb: Max size per shard in MB (default from config).

        Returns:
            List of paths to the created shard files.

        Raises:
            ShardingError: If the dataset is empty or serialization fails.
        """
        if len(dataset) == 0:
            raise ShardingError("Cannot shard an empty dataset")

        shard_size_mb = shard_size_mb or self._config.shard_size_mb
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Estimate samples per shard by sampling one item
        sample_input, sample_label = dataset[0]
        if not isinstance(sample_input, torch.Tensor):
            sample_input = torch.tensor(sample_input)
        if not isinstance(sample_label, torch.Tensor):
            sample_label = torch.tensor(sample_label)

        bytes_per_sample = sample_input.nelement() * sample_input.element_size()
        bytes_per_sample += sample_label.nelement() * sample_label.element_size()
        samples_per_shard = max(1, int((shard_size_mb * 1024 * 1024) / bytes_per_sample))

        total_samples = len(dataset)
        num_shards = math.ceil(total_samples / samples_per_shard)
        logger.info(
            f"Sharding {total_samples} samples → {num_shards} shards "
            f"(~{samples_per_shard} samples/shard, ~{shard_size_mb}MB each)"
        )

        shard_paths: list[Path] = []
        for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
            start = shard_idx * samples_per_shard
            end = min(start + samples_per_shard, total_samples)

            inputs = []
            labels = []
            for i in range(start, end):
                inp, lbl = dataset[i]
                if not isinstance(inp, torch.Tensor):
                    inp = torch.tensor(inp)
                if not isinstance(lbl, torch.Tensor):
                    lbl = torch.tensor(lbl)
                inputs.append(inp)
                labels.append(lbl)

            shard_data = {
                "data": torch.stack(inputs),
                "labels": torch.stack(labels),
            }

            shard_name = f"batch_{shard_idx:04d}.safetensors"
            shard_path = output_dir / shard_name
            try:
                save_file(shard_data, str(shard_path))
            except Exception as exc:
                raise ShardingError(f"Failed to serialize shard {shard_idx}: {exc}") from exc

            shard_paths.append(shard_path)
            logger.debug(f"Created shard {shard_name} ({end - start} samples)")

        logger.info(f"Sharding complete: {len(shard_paths)} files in {output_dir}")
        return shard_paths

    # ── Upload ──────────────────────────────────────────────────────────

    def upload_all_shards(self, shard_dir: str | Path) -> int:
        """Upload all .safetensors files from a local directory to the cloud bucket.

        Args:
            shard_dir: Directory containing shard files.

        Returns:
            Number of shards uploaded.
        """
        shard_dir = Path(shard_dir)
        shard_files = sorted(shard_dir.glob("*.safetensors"))

        if not shard_files:
            logger.warning(f"No .safetensors files found in {shard_dir}")
            return 0

        logger.info(f"Uploading {len(shard_files)} shards to bucket")
        for shard_path in tqdm(shard_files, desc="Uploading shards"):
            remote_key = f"{self._shard_prefix}{shard_path.name}"
            self._storage.upload_shard(shard_path, remote_key)

        logger.info(f"Upload complete: {len(shard_files)} shards")
        return len(shard_files)

    # ── Loading ─────────────────────────────────────────────────────────

    def get_shard_count(self) -> int:
        """Return the total number of shards available in cloud storage."""
        return self._storage.get_shard_count(prefix=self._shard_prefix)

    def load_shard(self, batch_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single shard from cloud storage into CPU memory.

        Args:
            batch_id: Zero-based shard index.

        Returns:
            Tuple of (data_tensor, labels_tensor).
        """
        remote_key = f"{self._shard_prefix}batch_{batch_id:04d}.safetensors"
        tensors = self._storage.download_shard_to_memory(remote_key)

        data = tensors.get("data")
        labels = tensors.get("labels")

        if data is None or labels is None:
            raise ShardingError(
                f"Shard {remote_key} is missing required keys 'data' and/or 'labels'. "
                f"Found keys: {list(tensors.keys())}"
            )

        logger.debug(f"Loaded shard {batch_id}: data={data.shape}, labels={labels.shape}")
        return data, labels

    def load_shard_local(self, shard_path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a shard from a local file (for offline / testing use).

        Returns:
            Tuple of (data_tensor, labels_tensor).
        """
        tensors = load_file(str(shard_path))
        return tensors["data"], tensors["labels"]
