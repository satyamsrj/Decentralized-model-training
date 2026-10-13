"""Launch an MNIST swarm training worker.

This script defines a simple CNN and uses SwarmWorker to run decentralized
training across multiple peers.

Usage:
    python train.py --config ../../config/swarm_config.yaml --epochs 5
    python train.py --config ../../config/swarm_config.yaml --epochs 10 --resume
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2p_ml_swarm.worker import SwarmWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification.

    Architecture:
        Conv2d(1→32) → ReLU → Conv2d(32→64) → ReLU → MaxPool → Dropout →
        Flatten → Linear(9216→128) → ReLU → Dropout → Linear(128→10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST Swarm Training Worker")
    parser.add_argument("--config", "-c", type=str, default="../../config/swarm_config.yaml",
                        help="Path to swarm_config.yaml")
    parser.add_argument("--epochs", "-e", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device (cpu/cuda/cuda:0)")
    parser.add_argument("--tensorboard-dir", type=str, default="./runs",
                        help="TensorBoard log directory (default: ./runs)")
    args = parser.parse_args()

    # Create the model
    model = MNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MNISTNet: {total_params:,} parameters")

    # Create and run the worker
    worker = SwarmWorker(
        config_path=args.config,
        model=model,
        num_epochs=args.epochs,
        resume=args.resume,
        device=args.device,
        tensorboard_dir=args.tensorboard_dir,
    )
    worker.run()


if __name__ == "__main__":
    main()
