"""LLaDA-MoE specific adapter.

Targets the fused expert tensors of ``TritonFusedMoEBlock``:

    block.w1: Parameter [local_experts, 2 * intermediate_dim, hidden_dim]
    block.w2: Parameter [local_experts, hidden_dim, intermediate_dim]

The adapter keeps the model code untouched: packed int8 weights and scales
are stored as extra persistent buffers on the block, and the live ``w1``/
``w2`` Parameters are replaced by their dequantized values. Router, norms,
attention and embeddings are never touched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..algorithms.symmetric import dequantize_tensor
from ..config import QuantConfig
from ..runtime.moe import QuantExpertWeights


def is_fused_expert_block(module: nn.Module) -> bool:
    w1 = getattr(module, "w1", None)
    w2 = getattr(module, "w2", None)
    return (
        isinstance(w1, nn.Parameter)
        and isinstance(w2, nn.Parameter)
        and w1.dim() == 3
        and w1.shape[1] % 2 == 0
    )


def quantize_llada_experts(model: nn.Module, config: QuantConfig) -> list[str]:
    """Quantize every fused expert block's ``w1``/``w2`` in place.

    Adds persistent buffers ``_qw1``/``_sw1``/``_qw2``/``_sw2`` (included in
    saved checkpoints) and materializes dequantized values back into the
    live ``w1``/``w2`` Parameters. Returns the list of quantized block names.
    """
    compute_dtype = getattr(torch, config.compute_dtype)
    scale_dtype = getattr(torch, config.scale_dtype)
    quantized: list[str] = []
    for name, block in model.named_modules():
        if not is_fused_expert_block(block):
            continue
        weights = QuantExpertWeights.quantize(
            block.w1.detach(),
            block.w2.detach(),
            bits=config.bits,
            group_size=config.group_size,
            scale_dtype=scale_dtype,
        )
        block.register_buffer("_qw1", weights.w1.q, persistent=True)
        block.register_buffer("_sw1", weights.w1.scale, persistent=True)
        block.register_buffer("_qw2", weights.w2.q, persistent=True)
        block.register_buffer("_sw2", weights.w2.scale, persistent=True)
        w1, w2 = weights.dequantize(dtype=compute_dtype)
        block.w1.data.copy_(w1)
        block.w2.data.copy_(w2)
        quantized.append(name)
    return quantized


def restore_llada_experts_from_buffers(model: nn.Module, config: QuantConfig) -> None:
    """Re-materialize ``w1``/``w2`` from the packed buffers (after a load)."""
    compute_dtype = getattr(torch, config.compute_dtype)
    for block in model.modules():
        if not all(hasattr(block, b) for b in ("_qw1", "_sw1", "_qw2", "_sw2")):
            continue
        w1 = dequantize_tensor(
            block._qw1, block._sw1, config.bits, config.group_size, dtype=compute_dtype
        )
        w2 = dequantize_tensor(
            block._qw2, block._sw2, config.bits, config.group_size, dtype=compute_dtype
        )
        block.w1.data.copy_(w1)
        block.w2.data.copy_(w2)