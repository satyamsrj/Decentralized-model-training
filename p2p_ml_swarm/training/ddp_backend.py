"""PyTorch DistributedDataParallel (DDP) training backend for Local ML Cluster.

Provides robust, synchronized multi-worker training across LAN, USB-LAN,
or private overlay networks using TCP rendezvous.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from p2p_ml_swarm.exceptions import TrainingError
from p2p_ml_swarm.storage.integrity import compute_sha256
from p2p_ml_swarm.training.base import CheckpointMetadata, StepMetrics, TrainingBackend

logger = logging.getLogger(__name__)


class DDPBackend(TrainingBackend):
    """PyTorch DistributedDataParallel (DDP) backend.

    Synchronizes replicas over a TCP rendezvous point, handling process
    group setup, device assignment, gradient all-reduce, and atomic checkpointing.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize backend with an unwrapped PyTorch model definition.

        Args:
            model: PyTorch nn.Module to replicate and train.
        """
        self._raw_model = model
        self._ddp_model: Optional[DDP] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self._world_size: int = 1
        self._rank: int = 0
        self._device: torch.device = torch.device("cpu")
        self._is_initialized: bool = False
        self._backend_str: str = "gloo"

    def setup(
        self,
        world_size: int,
        rank: int,
        rendezvous_endpoint: str,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        """Initialize the distributed process group and replicate model.

        Args:
            world_size: Total count of participating nodes.
            rank: Unique 0-indexed rank of this node.
            rendezvous_endpoint: Network URI, e.g. 'tcp://192.168.1.50:29500'.
            config: Training hyperparameters (lr, batch_size, etc.).
            device: Target device string ('cpu', 'cuda', 'cuda:0').
        """
        self._world_size = world_size
        self._rank = rank
        self._device = torch.device(device)

        # Move raw model to device
        self._raw_model.to(self._device)

        # Backend selection: NCCL for CUDA on Linux, Gloo for Windows/CPU/cross-platform
        is_windows = platform.system() == "Windows"
        if self._device.type == "cuda" and not is_windows:
            self._backend_str = "nccl"
        else:
            self._backend_str = "gloo"

        logger.info(
            "Initializing DDP process group [rank %d/%d, backend=%s, rendezvous=%s, device=%s]",
            rank,
            world_size,
            self._backend_str,
            rendezvous_endpoint,
            self._device,
        )

        if world_size > 1:
            try:
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend=self._backend_str,
                        init_method=rendezvous_endpoint,
                        world_size=world_size,
                        rank=rank,
                    )
                self._is_initialized = True
            except Exception as e:
                msg = f"Failed to initialize DDP process group at '{rendezvous_endpoint}': {e}"
                logger.error(msg)
                raise TrainingError(msg) from e

            # Wrap model in DistributedDataParallel
            if self._device.type == "cuda":
                self._ddp_model = DDP(self._raw_model, device_ids=[self._device.index or 0])
            else:
                self._ddp_model = DDP(self._raw_model)
        else:
            # Single-worker local execution (no inter-process group needed)
            self._is_initialized = True
            self._ddp_model = self._raw_model
            logger.info("Single-worker mode (world_size=1): running model directly on %s", self._device)

        # Configure optimizer
        lr = config.get("learning_rate", 0.001)
        optimizer_type = config.get("optimizer", "adam").lower()
        if optimizer_type == "sgd":
            self._optimizer = torch.optim.SGD(self._ddp_model.parameters(), lr=lr, momentum=0.9)
        else:
            self._optimizer = torch.optim.Adam(self._ddp_model.parameters(), lr=lr)

        loss_type = config.get("loss_fn", "cross_entropy").lower()
        if loss_type == "mse":
            self._criterion = nn.MSELoss()
        else:
            self._criterion = nn.CrossEntropyLoss()

        logger.info("DDP process group established successfully for rank %d", rank)

    def train_step(self, batch: Any, step_idx: int) -> StepMetrics:
        """Run single forward/backward step with synchronized gradient all-reduce."""
        if not self._is_initialized or self._ddp_model is None or self._optimizer is None:
            raise TrainingError("DDP backend is not initialized. Call setup() first.")

        start_time = time.perf_counter()
        inputs, targets = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)
        batch_size = inputs.size(0)

        self._ddp_model.train()
        self._optimizer.zero_grad()

        outputs = self._ddp_model(inputs)
        loss = self._criterion(outputs, targets)

        loss_val = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            raise TrainingError(f"NaN or Inf loss encountered at step {step_idx}: {loss_val}")

        loss.backward()
        self._optimizer.step()

        duration = time.perf_counter() - start_time
        throughput = batch_size / max(duration, 1e-6)
        current_lr = self._optimizer.param_groups[0]["lr"]

        return StepMetrics(
            step=step_idx,
            epoch=0,
            train_loss=loss_val,
            samples_processed=batch_size,
            throughput_samples_per_sec=throughput,
            learning_rate=current_lr,
            step_duration_seconds=duration,
        )

    def evaluate(self, val_loader: Any) -> Dict[str, float]:
        """Synchronized evaluation pass across validation data."""
        if self._ddp_model is None or self._criterion is None:
            raise TrainingError("Backend not initialized")

        self._ddp_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                outputs = self._ddp_model(inputs)
                loss = self._criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                if outputs.dim() > 1 and outputs.size(1) > 1:
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)

        # Synchronize loss and sample counts across all ranks
        stats_tensor = torch.tensor(
            [total_loss, float(total_correct), float(total_samples)],
            device=self._device,
        )
        if dist.is_initialized():
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        global_loss = stats_tensor[0].item() / max(stats_tensor[2].item(), 1.0)
        global_acc = (
            (stats_tensor[1].item() / max(stats_tensor[2].item(), 1.0)) * 100.0
            if total_correct > 0
            else 0.0
        )

        return {"val_loss": round(global_loss, 4), "val_accuracy": round(global_acc, 2)}

    def save_checkpoint(self, output_path: Path, metadata: Dict[str, Any]) -> CheckpointMetadata:
        """Atomically persist rank 0 state to disk and compute SHA-256."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")

        # Barrier to ensure all workers are in sync before saving
        if dist.is_initialized():
            dist.barrier()

        # Rank 0 persists the checkpoint
        if self._rank == 0:
            state = {
                "model_state_dict": self._raw_model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else {},
                "metadata": metadata,
            }
            torch.save(state, tmp_path)
            tmp_path.replace(path)
            digest = compute_sha256(path)
            size = path.stat().st_size
        else:
            digest = ""
            size = 0

        if dist.is_initialized():
            dist.barrier()

        return CheckpointMetadata(
            checkpoint_path=path,
            sha256_hash=digest,
            file_size_bytes=size,
            epoch=metadata.get("epoch", 0),
            global_step=metadata.get("step", 0),
            val_loss=metadata.get("val_loss"),
            backend_name=self.backend_name,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Restore model and optimizer state from checkpoint."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at '{path}'")

        state = torch.load(path, map_location=self._device, weights_only=False)
        self._raw_model.load_state_dict(state["model_state_dict"])
        if self._optimizer and "optimizer_state_dict" in state:
            self._optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info("Restored model weights from checkpoint '%s'", path.name)
        return state.get("metadata", {})

    def teardown(self) -> None:
        """Release distributed process groups cleanly."""
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info("DDP process group destroyed cleanly")
            except Exception as e:
                logger.warning("Error destroying process group: %s", e)
        self._is_initialized = False

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "PyTorch DDP"
