"""Checkpoint management for fault-tolerant training resumption."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from p2p_ml_swarm.exceptions import CheckpointError

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME_PATTERN = "checkpoint_epoch{epoch:04d}_batch{batch:06d}.pt"


class CheckpointManager:
    """Saves and restores training checkpoints for fault tolerance.

    Checkpoints include model weights, optimizer state, epoch, batch ID,
    and arbitrary metadata (loss, peer count, etc.).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        save_every_n_steps: int = 100,
    ) -> None:
        """
        Args:
            checkpoint_dir: Directory to save checkpoints. If None, checkpointing is disabled.
            save_every_n_steps: Auto-save frequency (number of training steps).
        """
        self._enabled = checkpoint_dir is not None
        self._dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._save_every = save_every_n_steps
        self._step_counter = 0

        if self._enabled and self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CheckpointManager: dir={self._dir}, save_every={save_every_n_steps} steps")
        else:
            logger.info("CheckpointManager: disabled (no checkpoint_dir configured)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_save(self) -> bool:
        """Check if a checkpoint should be saved at the current step."""
        if not self._enabled:
            return False
        self._step_counter += 1
        return self._step_counter % self._save_every == 0

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        batch_id: int,
        **extra_metadata: Any,
    ) -> Path:
        """Save a full training checkpoint.

        Args:
            model: The model to save.
            optimizer: The optimizer to save.
            epoch: Current epoch number.
            batch_id: Current batch/shard ID.
            **extra_metadata: Additional metadata (e.g. loss, lr, peer_count).

        Returns:
            Path to the saved checkpoint file.

        Raises:
            CheckpointError: If the save operation fails.
        """
        if not self._enabled or self._dir is None:
            raise CheckpointError("Checkpointing is disabled — no checkpoint_dir configured")

        filename = _CHECKPOINT_FILENAME_PATTERN.format(epoch=epoch, batch=batch_id)
        filepath = self._dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "batch_id": batch_id,
            "timestamp": time.time(),
            **extra_metadata,
        }

        try:
            torch.save(checkpoint, filepath)
        except Exception as exc:
            raise CheckpointError(f"Failed to save checkpoint to {filepath}: {exc}") from exc

        logger.info(f"Checkpoint saved: {filepath}")
        self._cleanup_old(keep_last=3)
        return filepath

    def load(self, path: str | Path) -> dict:
        """Load a specific checkpoint file.

        Returns:
            Dictionary with keys: model_state_dict, optimizer_state_dict, epoch,
            batch_id, timestamp, and any extra metadata.

        Raises:
            CheckpointError: If the file is missing or corrupt.
        """
        path = Path(path)
        if not path.exists():
            raise CheckpointError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise CheckpointError(f"Failed to load checkpoint {path}: {exc}") from exc

        logger.info(
            f"Loaded checkpoint: epoch={checkpoint.get('epoch')}, "
            f"batch_id={checkpoint.get('batch_id')}"
        )
        return checkpoint

    def find_latest(self) -> Optional[Path]:
        """Find the most recent checkpoint in the checkpoint directory.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        if not self._enabled or self._dir is None:
            return None

        checkpoints = sorted(self._dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            logger.info("No existing checkpoints found")
            return None

        latest = checkpoints[-1]
        logger.info(f"Found latest checkpoint: {latest.name}")
        return latest

    def restore(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[str | Path] = None,
    ) -> dict:
        """Restore model (and optionally optimizer) from a checkpoint.

        If no path is specified, uses the latest checkpoint.

        Returns:
            The full checkpoint dict (for extracting epoch, batch_id, etc.).

        Raises:
            CheckpointError: If no checkpoint is found or restoration fails.
        """
        if path is None:
            path = self.find_latest()
            if path is None:
                raise CheckpointError("No checkpoint found to restore from")

        checkpoint = self.load(path)

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            raise CheckpointError(f"Failed to restore model state: {exc}") from exc

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:
                logger.warning(f"Could not restore optimizer state: {exc}")

        return checkpoint

    def _cleanup_old(self, keep_last: int = 3) -> None:
        """Remove old checkpoints, keeping only the N most recent."""
        if not self._enabled or self._dir is None:
            return

        checkpoints = sorted(self._dir.glob("checkpoint_*.pt"))
        if len(checkpoints) <= keep_last:
            return

        for old_ckpt in checkpoints[:-keep_last]:
            try:
                old_ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {old_ckpt.name}")
            except OSError as exc:
                logger.warning(f"Failed to remove {old_ckpt.name}: {exc}")
