"""Curated model architecture templates and dataset loaders for Local ML Cluster.

Provides vetted model architectures and distributed data samplers,
preventing execution of arbitrary untrusted remote code.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset, TensorDataset

logger = logging.getLogger(__name__)


# ── Model Architectures ────────────────────────────────────────────────────────


class MnistCNN(nn.Module):
    """Vetted CNN architecture for MNIST / Fashion-MNIST digit classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class Cifar10ConvNet(nn.Module):
    """Small ConvNet architecture for 3-channel CIFAR-10 image classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class TabularMLP(nn.Module):
    """Multi-Layer Perceptron for tabular numerical classification."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


# ── Model Registry & Metadata ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelTemplateInfo:
    """Descriptor for a supported model architecture template."""

    name: str
    description: str
    parameter_count: int
    memory_footprint_mb: float
    supported_datasets: list[str]
    model_factory: Callable[[], nn.Module]

    def compute_architecture_hash(self) -> str:
        """Compute deterministic SHA-256 hash of layer names and parameter shapes."""
        model = self.model_factory()
        hash_input = []
        for name, param in sorted(model.named_parameters()):
            hash_input.append(f"{name}:{list(param.shape)}")
        raw = ";".join(hash_input).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


# Global Registry of vetted training templates
MODEL_REGISTRY: Dict[str, ModelTemplateInfo] = {
    "mnist_cnn": ModelTemplateInfo(
        name="mnist_cnn",
        description="Standard 2-layer CNN for MNIST handwritten digit recognition (28x28)",
        parameter_count=421642,
        memory_footprint_mb=1.6,
        supported_datasets=["mnist", "synthetic_digits"],
        model_factory=lambda: MnistCNN(num_classes=10),
    ),
    "fashion_mnist_cnn": ModelTemplateInfo(
        name="fashion_mnist_cnn",
        description="CNN classifier for Fashion-MNIST clothing articles (28x28)",
        parameter_count=421642,
        memory_footprint_mb=1.6,
        supported_datasets=["fashion_mnist", "synthetic_digits"],
        model_factory=lambda: MnistCNN(num_classes=10),
    ),
    "cifar10_convnet": ModelTemplateInfo(
        name="cifar10_convnet",
        description="3-layer ConvNet for CIFAR-10 RGB object classification (32x32x3)",
        parameter_count=621770,
        memory_footprint_mb=2.4,
        supported_datasets=["cifar10", "synthetic_cifar"],
        model_factory=lambda: Cifar10ConvNet(num_classes=10),
    ),
    "tabular_mlp": ModelTemplateInfo(
        name="tabular_mlp",
        description="3-layer MLP for structured tabular features and binary/multi-class classification",
        parameter_count=5762,
        memory_footprint_mb=0.1,
        supported_datasets=["synthetic_tabular"],
        model_factory=lambda: TabularMLP(input_dim=20, hidden_dim=64, num_classes=2),
    ),
}


def get_model_template(name: str) -> ModelTemplateInfo:
    """Retrieve template descriptor by name."""
    template = MODEL_REGISTRY.get(name.lower())
    if not template:
        raise KeyError(
            f"Model template '{name}' not found. Supported templates: {list(MODEL_REGISTRY.keys())}"
        )
    return template


def build_synthetic_dataset(
    dataset_type: str = "mnist",
    num_samples: int = 1000,
) -> Tuple[Dataset, Dataset]:
    """Generate deterministic synthetic train/val datasets for testing and offline runs.

    Guarantees that preflight and training can be run without live Internet downloads.
    """
    torch.manual_seed(42)
    if dataset_type in {"mnist", "fashion_mnist", "synthetic_digits"}:
        x = torch.randn(num_samples, 1, 28, 28)
        y = torch.randint(0, 10, (num_samples,))
    elif dataset_type in {"cifar10", "synthetic_cifar"}:
        x = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, 10, (num_samples,))
    else:
        x = torch.randn(num_samples, 20)
        y = torch.randint(0, 2, (num_samples,))

    split = int(num_samples * 0.8)
    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])
    return train_ds, val_ds


def create_distributed_loaders(
    dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create distributed PyTorch DataLoaders with DistributedSampler.

    Args:
        dataset: Training dataset.
        val_dataset: Validation dataset.
        batch_size: Batch size per worker.
        world_size: Total distributed worker count.
        rank: Rank index of this worker.
        num_workers: DataLoader multiprocessing worker count.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None
        val_sampler = None

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
    )

    return train_loader, val_loader
