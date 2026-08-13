"""Drop-in quantized linear module (reference path).

Stores packed integer weights + per-group scales and computes the output via
dequantize-then-matmul. Correctness reference for the future Triton kernel;
the public ``QuantLinear`` API will not change when the fast path lands.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..algorithms.symmetric import QuantResult, dequantize_tensor, quantize_tensor


class QuantLinear(nn.Module):
    """Quantized replacement for ``nn.Linear`` with identical call semantics.

    Internal layout mirrors ``nn.Linear``: ``qweight`` is ``[out_features,
    in_features]`` int8 and per-group scales ``scale`` is ``[out_features,
    num_groups]`` (groups along the K / input dimension).
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
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "qweight", torch.empty(out_features, in_features, dtype=torch.int8)
        )
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
    ) -> "QuantLinear":
        w = linear.weight.detach().to(torch.float32)
        q: QuantResult = quantize_tensor(w, bits=bits, group_size=group_size)
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
            self.qweight, self.scale, self.bits, self.group_size, dtype=self.compute_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x.to(self.compute_dtype), self.dequantize_weight())
        if self.bias is not None:
            out = out + self.bias
        return out