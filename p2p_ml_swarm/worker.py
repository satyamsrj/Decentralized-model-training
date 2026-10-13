"""Top-level worker orchestrator — wires all components and runs the training lifecycle."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from p2p_ml_swarm.checkpoint import CheckpointManager
from p2p_ml_swarm.compression import estimate_compression_ratio, is_bnb_available
from p2p_ml_swarm.config import SwarmConfig
from p2p_ml_swarm.data_pipeline import DataPipeline
from p2p_ml_swarm.dht_manager import DHTManager
from p2p_ml_swarm.monitoring import SwarmMonitor
from p2p_ml_swarm.storage import StorageClient
from p2p_ml_swarm.trainer import SwarmTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SwarmWorker:
    """Full lifecycle manager for a single swarm worker node.

    Handles: config loading → DHT connection → storage setup → model creation →
    training loop → checkpointing → graceful shutdown.
    """

    def __init__(
        self,
        config_path: str | Path,
        model: Optional[nn.Module] = None,
        num_epochs: int = 1,
        resume: bool = False,
        device: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._model = model
        self._num_epochs = num_epochs
        self._resume = resume
        self._device = device
        self._tensorboard_dir = tensorboard_dir

        # Components (initialized in run())
        self._config: Optional[SwarmConfig] = None
        self._dht_manager: Optional[DHTManager] = None
        self._storage: Optional[StorageClient] = None
        self._data_pipeline: Optional[DataPipeline] = None
        self._monitor: Optional[SwarmMonitor] = None
        self._checkpoint_mgr: Optional[CheckpointManager] = None
        self._trainer: Optional[SwarmTrainer] = None
        self._shutdown_requested = False

    def _setup_signal_handlers(self) -> None:
        """Register handlers for graceful shutdown on SIGINT / SIGTERM."""

        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name} — initiating graceful shutdown")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    def run(self) -> None:
        """Execute the full worker lifecycle."""
        self._setup_signal_handlers()

        # ── 1. Load Configuration ──────────────────────────────────────
        logger.info(f"Loading config from {self._config_path}")
        self._config = SwarmConfig.from_yaml(self._config_path)
        logger.info(f"Run ID: {self._config.run_id}")
        logger.info(f"Bootnode: {self._config.bootnode_addr}")
        logger.info(f"Bucket: {self._config.bucket_name}")

        # ── 2. Validate Model ──────────────────────────────────────────
        if self._model is None:
            raise ValueError(
                "No model provided. Pass a model instance to SwarmWorker or "
                "use a pre-built example (e.g. examples/mnist/train.py)"
            )

        # ── 3. Connect to DHT ──────────────────────────────────────────
        logger.info("Connecting to DHT swarm...")
        self._dht_manager = DHTManager(self._config)
        self._dht_manager.connect()
        peer_count = self._dht_manager.get_peer_count()
        logger.info(f"Connected! {peer_count} peer(s) visible in the swarm")

        # ── 3. Initialize Storage Client ───────────────────────────────
        logger.info("Initializing cloud storage client...")
        self._storage = StorageClient(self._config)

        # ── 4. Initialize Data Pipeline ────────────────────────────────
        self._data_pipeline = DataPipeline(self._config, self._storage)
        shard_count = self._data_pipeline.get_shard_count()
        logger.info(f"Data pipeline ready: {shard_count} shards available in bucket")

        # ── 5. Model Info ──────────────────────────────────────────────

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

        compression_ratio = estimate_compression_ratio(self._model)
        logger.info(
            f"Estimated 8-bit compression ratio: {compression_ratio:.2f}x "
            f"(bnb available: {is_bnb_available()})"
        )

        # ── 6. Initialize Monitoring ───────────────────────────────────
        self._monitor = SwarmMonitor(
            run_id=self._config.run_id,
            tensorboard_dir=self._tensorboard_dir,
        )

        # ── 7. Initialize Checkpointing ────────────────────────────────
        self._checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self._config.checkpoint_dir,
        )

        # ── 8. Build Trainer ───────────────────────────────────────────
        self._trainer = SwarmTrainer(
            model=self._model,
            config=self._config,
            dht_manager=self._dht_manager,
            data_pipeline=self._data_pipeline,
            monitor=self._monitor,
            checkpoint_mgr=self._checkpoint_mgr,
            device=self._device,
        )
        self._trainer.setup()

        # ── 9. Resume from Checkpoint (if requested) ──────────────────
        if self._resume:
            self._trainer.resume_from_checkpoint()

        # ── 10. Train! ─────────────────────────────────────────────────
        try:
            self._trainer.train(self._num_epochs)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as exc:
            logger.error(f"Training failed: {exc}", exc_info=True)
            raise
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down worker...")

        if self._monitor:
            self._monitor.close()
        if self._dht_manager:
            self._dht_manager.shutdown()

        logger.info("Worker shutdown complete")


def main() -> None:
    """CLI entry point for launching a swarm worker."""
    parser = argparse.ArgumentParser(
        description="Launch a P2P ML Swarm worker node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  p2p-ml-worker --config config/swarm_config.yaml --epochs 5\n"
            "  p2p-ml-worker --config config/swarm_config.yaml --resume --epochs 10\n"
        ),
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/swarm_config.yaml",
        help="Path to swarm_config.yaml (default: config/swarm_config.yaml)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs (disabled if not set)",
    )

    args = parser.parse_args()

    # Note: The CLI doesn't know which model to use. Users should either:
    # 1. Use an example script (examples/mnist/train.py) that instantiates a model, or
    # 2. Write their own script that creates a SwarmWorker with a model.
    logger.error(
        "The generic p2p-ml-worker CLI requires a model. "
        "Use an example script (e.g. examples/mnist/train.py) or write a custom script "
        "that instantiates SwarmWorker with your model."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
