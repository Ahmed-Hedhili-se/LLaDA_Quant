"""Drop-in quantized linear module (reference path).

Stores packed integer weights + per-group scales and computes the output via
dequantize-then-matmul. This is the correctness reference the future Triton
kernel must reproduce, and it always *reduces resident memory* (the BF16
weight is gone) while *costing latency* (the weight is reconstructed on every
call). Both halves of that trade are real; neither is hidden.

At ``bits=4`` the weight is genuinely packed two values per byte, so
``qweight`` has half the columns of the logical weight.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithms.symmetric import (
    QuantResult,
    dequantize_tensor,
    quantize_tensor,
    validate_int4_layout,
)


class QuantLinear(nn.Module):
    """Quantized replacement for ``nn.Linear`` with identical call semantics.

    Internal layout mirrors ``nn.Linear``: ``qweight`` is
    ``[out_features, in_features]`` int8 at 8 bits, or
    ``[out_features, in_features // 2]`` at 4 bits; per-group scales
    ``scale`` are ``[out_features, num_groups]`` (groups along the K axis).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
        scale_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if bits not in (8, 4):
            raise ValueError(f"bits must be 8 or 4, got {bits}")
        if bits == 4:
            validate_int4_layout((out_features, in_features), group_size)
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self.packed = bits == 4
        stored_cols = in_features // 2 if self.packed else in_features
        self.register_buffer("qweight", torch.empty(out_features, stored_cols, dtype=torch.int8))
        self.register_buffer(
            "scale", torch.empty(out_features, self._num_groups(), dtype=scale_dtype)
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.register_buffer("bias", None)

    def _num_groups(self) -> int:
        if self.group_size == -1 or self.in_features % self.group_size != 0:
            return 1
        return self.in_features // self.group_size

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.bfloat16,
        scale_search: str = "amax",
        search_grid: int = 24,
    ) -> "QuantLinear":
        w = linear.weight.detach().to(torch.float32)
        q: QuantResult = quantize_tensor(
            w, bits=bits, group_size=group_size,
            scale_search=scale_search, search_grid=search_grid,
        )
        out = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            group_size=group_size,
            compute_dtype=compute_dtype,
            bias=linear.bias is not None,
        )
        out.qweight.copy_(q.q)
        out.scale.copy_(q.scale)
        if linear.bias is not None:
            out.bias.copy_(linear.bias.detach())
        return out

    def dequantize_weight(self) -> torch.Tensor:
        return dequantize_tensor(
            self.qweight,
            self.scale,
            self.bits,
            self.group_size,
            dtype=self.compute_dtype,
            packed=self.packed,
        )

    def storage_bytes(self) -> int:
        """Resident bytes of this module's weight representation."""
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scale.numel() * self.scale.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, packed={self.packed}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x.to(self.compute_dtype), self.dequantize_weight())
        if self.bias is not None:
            out = out + self.bias
        return out
