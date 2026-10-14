"""Unit tests for curated model architectures, template registry, and synthetic data loaders."""

import pytest
import torch
from p2p_ml_swarm.training.model_registry import (
    MODEL_REGISTRY,
    MnistCNN,
    Cifar10ConvNet,
    TabularMLP,
    get_model_template,
    build_synthetic_dataset,
    create_distributed_loaders,
)


def test_registry_lookup():
    template = get_model_template("mnist_cnn")
    assert template.name == "mnist_cnn"
    assert template.parameter_count > 0
    assert template.memory_footprint_mb > 0.0

    with pytest.raises(KeyError):
        get_model_template("non_existent_transformer")


def test_model_forward_passes():
    # MNIST CNN
    mnist_model = MnistCNN(num_classes=10)
    x_mnist = torch.randn(4, 1, 28, 28)
    out_mnist = mnist_model(x_mnist)
    assert out_mnist.shape == (4, 10)

    # CIFAR10 ConvNet
    cifar_model = Cifar10ConvNet(num_classes=10)
    x_cifar = torch.randn(2, 3, 32, 32)
    out_cifar = cifar_model(x_cifar)
    assert out_cifar.shape == (2, 10)

    # Tabular MLP
    tab_model = TabularMLP(input_dim=20, hidden_dim=64, num_classes=2)
    x_tab = torch.randn(8, 20)
    out_tab = tab_model(x_tab)
    assert out_tab.shape == (8, 2)


def test_architecture_hash_deterministic():
    template = get_model_template("mnist_cnn")
    hash1 = template.compute_architecture_hash()
    hash2 = template.compute_architecture_hash()
    assert hash1 == hash2
    assert len(hash1) == 64


def test_synthetic_dataset_and_loaders():
    train_ds, val_ds = build_synthetic_dataset("mnist", num_samples=100)
    assert len(train_ds) == 80
    assert len(val_ds) == 20

    train_loader, val_loader = create_distributed_loaders(
        dataset=train_ds,
        val_dataset=val_ds,
        batch_size=16,
        world_size=1,
        rank=0,
    )
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (16, 1, 28, 28)
    assert batch_y.shape == (16,)
