"""Abstract training backend interface for Local ML Cluster.

Decouples the distributed training engine (PyTorch DDP, Hivemind, Local Baseline)
from GUI controls, run orchestration, and network adapters.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics produced by a single local training step."""

    step: int
    epoch: int
    train_loss: float
    samples_processed: int
    throughput_samples_per_sec: float
    learning_rate: float
    step_duration_seconds: float
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CheckpointMetadata:
    """Metadata describing a validated checkpoint artifact."""

    checkpoint_path: Path
    sha256_hash: str
    file_size_bytes: int
    epoch: int
    global_step: int
    val_loss: Optional[float]
    backend_name: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TrainingBackend(abc.ABC):
    """Abstract base class for all distributed training engines.

    Implementations must manage process synchronization, model replicas,
    optimizer step coordination, and checkpoint serialization without
    leaking UI or network topology concerns.
    """

    @abc.abstractmethod
    def setup(
        self,
        world_size: int,
        rank: int,
        rendezvous_endpoint: str,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        """Initialize the distributed process group and backend resources.

        Args:
            world_size: Total number of participating workers.
            rank: Unique 0-indexed rank of this worker.
            rendezvous_endpoint: Network address (e.g. 'tcp://192.168.1.10:29500') for coordination.
            config: Training hyperparameters (lr, batch_size, etc.).
            device: Target compute device ('cpu', 'cuda', 'cuda:0', etc.).
        """
        pass

    @abc.abstractmethod
    def train_step(self, batch: Any, step_idx: int) -> StepMetrics:
        """Execute forward pass, backward pass, gradient all-reduce, and optimizer step.

        Args:
            batch: Training data batch (inputs, targets).
            step_idx: Monotonically increasing step counter.

        Returns:
            StepMetrics capturing loss, timing, and local throughput.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, val_loader: Any) -> Dict[str, float]:
        """Run synchronized evaluation across validation data.

        Returns:
            Dict containing validation metrics (e.g. {'val_loss': 0.12, 'val_accuracy': 98.4}).
        """
        pass

    @abc.abstractmethod
    def save_checkpoint(self, output_path: Path, metadata: Dict[str, Any]) -> CheckpointMetadata:
        """Atomically persist model and optimizer state to disk with integrity validation.

        Args:
            output_path: Target path for the checkpoint artifact.
            metadata: Additional run/epoch information to store alongside weights.

        Returns:
            CheckpointMetadata with SHA-256 hash and file size.
        """
        pass

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Restore model weights and optimizer state from a validated checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file to load.

        Returns:
            Restored metadata dictionary.
        """
        pass

    @abc.abstractmethod
    def teardown(self) -> None:
        """Gracefully release distributed process groups and hardware devices."""
        pass

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier for this training backend."""
        pass
