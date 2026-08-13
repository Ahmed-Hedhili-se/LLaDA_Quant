"""Interfaces for quantized MoE expert weights (``w1``/``w2``).

The quantized layout mirrors the fused tensors used by the Triton MoE path:

    w1: [local_experts, 2 * intermediate_dim, hidden_dim]   (Gate+Up stacked)
    w2: [local_experts, hidden_dim, intermediate_dim]       (Down)

Each expert is quantized independently; groups run along the last (K) axis,
so per-expert per-group scales fall out of the generic algorithm without
special-casing the expert dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..algorithms.symmetric import QuantResult, quantize_tensor

W1 = "w1"
W2 = "w2"


@dataclass
class QuantExpertWeights:
    """Packed + scales for one MoE layer's ``w1`` and ``w2`` fused tensors."""

    w1: QuantResult
    w2: QuantResult

    @classmethod
    def quantize(
        cls,
        w1: torch.Tensor,
        w2: torch.Tensor,
        bits: int = 8,
        group_size: int = 128,
        scale_dtype: torch.dtype = torch.float32,
    ) -> "QuantExpertWeights":
        return cls(
            w1=quantize_tensor(w1, bits=bits, group_size=group_size, scale_dtype=scale_dtype),
            w2=quantize_tensor(w2, bits=bits, group_size=group_size, scale_dtype=scale_dtype),
        )

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w1.dequantize(dtype=dtype), self.w2.dequantize(dtype=dtype)


def quantize_fused_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float32,
) -> QuantExpertWeights:
    """Convenience wrapper: quantize a fused ``w1``/``w2`` pair."""
    return QuantExpertWeights.quantize(w1, w2, bits=bits, group_size=group_size, scale_dtype=scale_dtype)


def materialize_expert_params(
    module: torch.nn.Module,
    weights: QuantExpertWeights,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Copy dequantized weights into an existing module's ``w1``/``w2`` params.

    This is the reference-runtime contract: the model keeps running BF16
    matmuls on the dequantized values, so router behavior is untouched.
    """
    w1, w2 = weights.dequantize(dtype=compute_dtype)
    module.w1.data.copy_(w1)
    module.w2.data.copy_(w2)