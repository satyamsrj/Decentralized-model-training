"""Core training engine — PyTorch training loop wrapped with Hivemind P2P optimizer."""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import torch
import torch.nn as nn

try:
    import hivemind
except ImportError:
    hivemind = None  # type: ignore[assignment]

from p2p_ml_swarm.checkpoint import CheckpointManager
from p2p_ml_swarm.compression import get_8bit_optimizer
from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.data_pipeline import DataPipeline
from p2p_ml_swarm.dht_manager import DHTManager
from p2p_ml_swarm.exceptions import NaNLossError, PeerSyncError, TrainingError
from p2p_ml_swarm.monitoring import SwarmMonitor

logger = logging.getLogger(__name__)


class SwarmTrainer:
    """Decentralized training engine using Hivemind for P2P gradient averaging.

    Orchestrates the training loop: checks out data shards from the DHT,
    runs forward/backward on the local model, and lets Hivemind's optimizer
    handle asynchronous gradient gossip with other peers.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SwarmConfig,
        dht_manager: DHTManager,
        data_pipeline: DataPipeline,
        monitor: SwarmMonitor,
        checkpoint_mgr: CheckpointManager,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model = model
        self._config = config
        self._dht = dht_manager
        self._data = data_pipeline
        self._monitor = monitor
        self._ckpt = checkpoint_mgr
        self._loss_fn = loss_fn or nn.CrossEntropyLoss()

        # Device selection
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model = self._model.to(self._device)
        self._loss_fn = self._loss_fn.to(self._device)
        logger.info(f"Training on device: {self._device}")

        # Optimizer will be created in setup()
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._hivemind_optimizer = None
        self._start_epoch = 0
        self._start_batch = 0

    def setup(self) -> None:
        """Initialize the Hivemind-wrapped optimizer and prepare for training."""
        # Create base 8-bit optimizer
        base_optimizer = get_8bit_optimizer(
            self._model.parameters(),
            lr=self._config.learning_rate,
        )

        if hivemind is not None:
            # Wrap with Hivemind for P2P gradient averaging
            logger.info(
                f"Setting up Hivemind optimizer: run_id={self._config.run_id}, "
                f"target_batch={self._config.target_batch_size}"
            )
            self._hivemind_optimizer = hivemind.Optimizer(
                dht=self._dht.dht,
                run_id=self._config.run_id,
                params=self._model.parameters(),
                optimizer=lambda params: get_8bit_optimizer(
                    params, lr=self._config.learning_rate
                ),
                batch_size_per_step=self._config.batch_size_per_step,
                target_batch_size=self._config.target_batch_size,
                matchmaking_time=self._config.matchmaking_time,
                averaging_timeout=self._config.averaging_timeout,
                offload_optimizer=self._config.offload_optimizer,
                auxiliary=self._config.auxiliary,
                verbose=True,
            )
            self._optimizer = self._hivemind_optimizer
            logger.info("Hivemind optimizer ready — P2P gossip enabled")
        else:
            logger.warning("Hivemind not available — running in standalone mode")
            self._optimizer = base_optimizer

    def resume_from_checkpoint(self) -> None:
        """Attempt to resume training from the latest checkpoint."""
        if not self._ckpt.enabled:
            return

        latest = self._ckpt.find_latest()
        if latest is None:
            return

        logger.info(f"Resuming from checkpoint: {latest}")
        ckpt = self._ckpt.restore(self._model, self._optimizer, latest)
        self._start_epoch = ckpt.get("epoch", 0)
        self._start_batch = ckpt.get("batch_id", 0)
        logger.info(f"Resumed at epoch={self._start_epoch}, batch={self._start_batch}")

    def train_step(self, data: torch.Tensor, labels: torch.Tensor) -> float:
        """Execute a single training step (forward + backward + optimizer step).

        Args:
            data: Input tensor batch.
            labels: Target labels tensor.

        Returns:
            The loss value for this step.

        Raises:
            NaNLossError: If the loss is NaN or Inf.
            TrainingError: If forward/backward computation fails.
        """
        self._model.train()
        data = data.to(self._device, non_blocking=True)
        labels = labels.to(self._device, non_blocking=True)

        try:
            output = self._model(data)
            loss = self._loss_fn(output, labels)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise TrainingError(
                    f"GPU OOM during forward pass. Try reducing batch_size_per_step "
                    f"(currently {self._config.batch_size_per_step}). Error: {exc}"
                ) from exc
            raise TrainingError(f"Forward pass failed: {exc}") from exc

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise NaNLossError(
                f"Loss is {loss_val} at step {self._monitor.step_count}. "
                "Check learning rate, data normalization, and model architecture."
            )

        try:
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise TrainingError(f"GPU OOM during backward pass: {exc}") from exc
            raise TrainingError(f"Backward/step failed: {exc}") from exc

        return loss_val

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch by dynamically checking out shards from the DHT.

        Each peer checks out the next available shard from the DHT,
        trains on it, marks it complete, and repeats. Faster peers
        naturally process more shards.

        Args:
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        epoch_start = time.time()
        total_shards = self._data.get_shard_count()
        logger.info(f"Starting epoch {epoch}: {total_shards} shards available")

        losses: list[float] = []
        shards_processed = 0

        while True:
            # Check out the next batch from the DHT
            batch_id = self._dht.checkout_token()
            if batch_id is None or batch_id >= total_shards:
                logger.info(f"Epoch {epoch}: no more shards to process (processed {shards_processed})")
                break

            # Skip if already completed
            if self._dht.is_batch_done(batch_id):
                logger.debug(f"Shard {batch_id} already done, skipping")
                continue

            # Load shard from cloud storage into memory
            try:
                data, labels = self._data.load_shard(batch_id)
            except Exception as exc:
                logger.error(f"Failed to load shard {batch_id}: {exc}")
                continue

            # Process the shard in mini-batches
            num_samples = data.size(0)
            bs = self._config.batch_size_per_step
            shard_losses: list[float] = []

            for start_idx in range(0, num_samples, bs):
                end_idx = min(start_idx + bs, num_samples)
                mini_data = data[start_idx:end_idx]
                mini_labels = labels[start_idx:end_idx]

                try:
                    loss = self.train_step(mini_data, mini_labels)
                    shard_losses.append(loss)

                    # Log step
                    peer_count = self._dht.get_peer_count()
                    self._monitor.log_step(
                        epoch=epoch,
                        batch_id=batch_id,
                        loss=loss,
                        lr=self._config.learning_rate,
                        peer_count=peer_count,
                        batch_size=end_idx - start_idx,
                    )
                except NaNLossError:
                    raise
                except TrainingError as exc:
                    logger.error(f"Training step failed: {exc}")
                    break

            # Mark shard as complete in DHT
            if shard_losses:
                self._dht.mark_complete(batch_id)
                avg_shard_loss = sum(shard_losses) / len(shard_losses)
                losses.append(avg_shard_loss)
                shards_processed += 1
                logger.info(
                    f"Shard {batch_id} complete: avg_loss={avg_shard_loss:.4f} "
                    f"({shards_processed}/{total_shards})"
                )

            # Auto-checkpoint
            if self._ckpt.should_save():
                self._ckpt.save(
                    self._model,
                    self._optimizer,
                    epoch,
                    batch_id,
                    loss=losses[-1] if losses else 0.0,
                )

        # Epoch summary
        elapsed = time.time() - epoch_start
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self._monitor.log_epoch_summary(epoch, avg_loss, elapsed)
        return avg_loss

    def train(self, num_epochs: int) -> None:
        """Run the full training loop for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
        """
        if self._optimizer is None:
            raise TrainingError("Trainer not set up. Call setup() first.")

        logger.info(
            f"Starting training: {num_epochs} epochs, "
            f"batch_size={self._config.batch_size_per_step}, "
            f"target_batch={self._config.target_batch_size}, "
            f"device={self._device}"
        )

        self._monitor.print_status_table()

        for epoch in range(self._start_epoch, self._start_epoch + num_epochs):
            avg_loss = self.train_epoch(epoch)

            # Save checkpoint at end of each epoch
            if self._ckpt.enabled:
                self._ckpt.save(
                    self._model,
                    self._optimizer,
                    epoch,
                    batch_id=-1,
                    avg_loss=avg_loss,
                )

        self._monitor.print_status_table()
        logger.info("Training complete!")
