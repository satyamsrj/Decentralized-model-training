"""Tests for 8-bit gradient compression utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from p2p_ml_swarm.compression import (
    compress_state_dict,
    decompress_state_dict,
    estimate_compression_ratio,
    get_8bit_optimizer,
    is_bnb_available,
)


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


class TestOptimizer:
    def test_get_8bit_optimizer_returns_optimizer(self, simple_model):
        optimizer = get_8bit_optimizer(simple_model.parameters(), lr=1e-3)
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_optimizer_can_step(self, simple_model):
        optimizer = get_8bit_optimizer(simple_model.parameters(), lr=1e-3)
        x = torch.randn(4, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()  # Should not raise


class TestCompressDecompress:
    def test_roundtrip_preserves_shape(self, simple_model):
        state = simple_model.state_dict()
        compressed = compress_state_dict(state)
        restored = decompress_state_dict(compressed)

        for key in state:
            assert key in restored, f"Key {key} missing after roundtrip"
            assert state[key].shape == restored[key].shape

    def test_compressed_uses_int8(self, simple_model):
        state = simple_model.state_dict()
        compressed = compress_state_dict(state)

        for key, tensor in compressed.items():
            if key.endswith(".__scale"):
                continue
            if state.get(key) is not None and state[key].is_floating_point():
                assert tensor.dtype == torch.int8

    def test_roundtrip_accuracy(self, simple_model):
        """Quantization should preserve values within ~1% for typical weights."""
        state = simple_model.state_dict()
        compressed = compress_state_dict(state)
        restored = decompress_state_dict(compressed)

        for key in state:
            if state[key].is_floating_point():
                original = state[key].float()
                recovered = restored[key].float()
                # Allow generous tolerance for 8-bit quantization
                max_error = (original - recovered).abs().max().item()
                max_val = original.abs().max().item()
                if max_val > 0:
                    relative_error = max_error / max_val
                    assert relative_error < 0.02, (
                        f"Key {key}: relative error {relative_error:.4f} > 2%"
                    )

    def test_empty_state_dict(self):
        compressed = compress_state_dict({})
        restored = decompress_state_dict(compressed)
        assert restored == {}


class TestEstimateCompression:
    def test_ratio_less_than_one(self, simple_model):
        ratio = estimate_compression_ratio(simple_model)
        assert 0 < ratio < 1.0, f"Expected ratio < 1.0, got {ratio}"

    def test_empty_model_ratio(self):
        model = nn.Module()
        ratio = estimate_compression_ratio(model)
        assert ratio == 1.0
