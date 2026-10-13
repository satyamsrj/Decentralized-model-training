"""8-bit gradient compression using bitsandbytes for bandwidth-efficient P2P sync."""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Attempt to import bitsandbytes; fall back gracefully if unavailable.
try:
    import bitsandbytes as bnb

    _BNB_AVAILABLE = True
except ImportError:
    bnb = None  # type: ignore[assignment]
    _BNB_AVAILABLE = False
    logger.warning(
        "bitsandbytes not available — falling back to standard (32-bit) optimizers. "
        "Install bitsandbytes for 8-bit compression: pip install bitsandbytes"
    )


def is_bnb_available() -> bool:
    """Check whether bitsandbytes is installed and importable."""
    return _BNB_AVAILABLE


def get_8bit_optimizer(
    params: Iterator[nn.Parameter],
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """Return an 8-bit Adam optimizer if bitsandbytes is available, else standard Adam.

    The 8-bit optimizer maintains near-32-bit accuracy while using ~4× less memory
    for optimizer states. This is critical for reducing the bandwidth of gradient
    gossip across the P2P mesh.

    Args:
        params: Model parameters to optimize.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
        betas: Adam beta coefficients.

    Returns:
        An optimizer instance (either ``bnb.optim.Adam8bit`` or ``torch.optim.Adam``).
    """
    if _BNB_AVAILABLE:
        logger.info("Using bitsandbytes 8-bit Adam optimizer")
        return bnb.optim.Adam8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        logger.info("Using standard PyTorch Adam optimizer (32-bit)")
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)


def compress_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Quantize a model state dict to 8-bit for bandwidth-efficient transfer.

    Converts float32/float16 parameter tensors to int8 with per-tensor scale factors.
    Non-floating-point tensors (e.g. batch-norm running counts) are left unchanged.

    Returns:
        A new dict with quantized tensors and their scale factors stored under
        ``<key>.__scale`` entries.
    """
    compressed: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if tensor.is_floating_point() and tensor.numel() > 0:
            abs_max = tensor.abs().max().clamp(min=1e-8)
            scale = abs_max / 127.0
            quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            compressed[key] = quantized
            compressed[f"{key}.__scale"] = scale.unsqueeze(0)
        else:
            compressed[key] = tensor
    return compressed


def decompress_state_dict(compressed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Restore a quantized state dict back to float32.

    Reads scale factors from ``<key>.__scale`` entries and dequantizes int8 tensors.
    """
    restored: dict[str, torch.Tensor] = {}
    scale_suffix = ".__scale"

    # Collect scale keys
    scale_keys = {k for k in compressed if k.endswith(scale_suffix)}
    for key, tensor in compressed.items():
        if key in scale_keys:
            continue  # skip scale entries themselves
        if tensor.dtype == torch.int8:
            scale_key = f"{key}{scale_suffix}"
            if scale_key in compressed:
                scale = compressed[scale_key].squeeze(0)
                restored[key] = tensor.float() * scale
            else:
                # No scale found, keep as-is (shouldn't happen in normal flow)
                restored[key] = tensor.float()
        else:
            restored[key] = tensor
    return restored


def estimate_compression_ratio(model: nn.Module) -> float:
    """Estimate the bandwidth savings from 8-bit compression.

    Returns:
        Approximate ratio of compressed / uncompressed size.
    """
    total_bytes = 0
    float_bytes = 0
    for param in model.parameters():
        size = param.nelement() * param.element_size()
        total_bytes += size
        if param.is_floating_point():
            float_bytes += size

    if total_bytes == 0:
        return 1.0

    # Float params → int8 (1 byte each) + 4 byte scale per tensor
    num_float_params = sum(p.nelement() for p in model.parameters() if p.is_floating_point())
    num_float_tensors = sum(1 for p in model.parameters() if p.is_floating_point())
    compressed_float_bytes = num_float_params * 1 + num_float_tensors * 4
    non_float_bytes = total_bytes - float_bytes
    compressed_total = compressed_float_bytes + non_float_bytes

    return compressed_total / total_bytes if total_bytes > 0 else 1.0
