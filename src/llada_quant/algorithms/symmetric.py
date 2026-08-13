"""Symmetric (zero-point-free) groupwise integer quantization.

Reference implementation on CPU/GPU via torch. Establishes the numerical
contract the future Triton kernels must reproduce:

    W_q = clamp(round(W / s), -Qmax - 1, Qmax)   bit-exact per group
    s   = max(|W_group|, eps) / Qmax

Groups run along the last (K) axis. An expert-stacked weight of shape
``[E, N, K]`` is therefore quantized per expert and per group without any
reshape of the expert dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def qmax_for_bits(bits: int) -> int:
    """Largest representable positive integer for signed symmetric quant."""
    return 2 ** (bits - 1) - 1


@dataclass
class QuantResult:
    """Packed integer weights plus per-group scales.

    Attributes:
        q: Integer tensor with the same shape as ``w``; dtype ``int8`` for
            bits <= 8 (values fit in int8 up to 127).
        scale: Per-group scales, shape ``w.shape[:-1] + (num_groups,)``.
        bits: Bit-width used.
        group_size: Group size used (``-1`` means per-tensor).
    """

    q: torch.Tensor
    scale: torch.Tensor
    bits: int
    group_size: int

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_tensor(self.q, self.scale, self.bits, self.group_size, dtype=dtype)


def _as_groups(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    if group_size == -1 or w.shape[-1] % group_size != 0:
        return w, 1
    shape = w.shape
    w_g = w.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    return w_g, shape[-1] // group_size


def quantize_tensor(
    w: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float32,
) -> QuantResult:
    """Quantize ``w`` symmetrically per group along its last axis.

    Falls back to per-tensor scaling when ``group_size`` does not divide the
    last dimension (or is ``-1``).
    """
    if bits not in (8, 4):
        raise ValueError(f"bits must be 8 or 4, got {bits}")
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        w = w.to(torch.float32)

    w_g, num_groups = _as_groups(w, group_size)
    qmax = qmax_for_bits(bits)
    amax = w_g.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, (amax / qmax).to(scale_dtype), torch.zeros_like(amax, dtype=scale_dtype))
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(w_g / safe.to(w.dtype)).clamp(-qmax - 1, qmax).to(torch.int8)

    if num_groups == 1:
        scale = scale.reshape(*w.shape[:-1], 1)
    else:
        scale = scale.reshape(*w.shape[:-1], num_groups)
    return QuantResult(q.reshape_as(w), scale, bits, group_size if num_groups > 1 else -1)


def dequantize_tensor(
    q: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct ``w ~= q * scale`` with the inverse reshape of ``quantize_tensor``.

    Arithmetic runs in float32 regardless of ``dtype``, then the result is
    cast; this keeps scale precision intact when the target dtype is BF16.
    """
    if group_size == -1 or q.shape[-1] % group_size != 0:
        return (q.to(torch.float32) * scale.to(torch.float32)).to(dtype)
    shape = q.shape
    q_g = q.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    s = scale.reshape(*shape[:-1], shape[-1] // group_size, 1)
    return (q_g.to(torch.float32) * s.to(torch.float32)).reshape_as(q).to(dtype)


def pack_int4(q8: torch.Tensor) -> torch.Tensor:
    """Pack adjacent int4 values along the last axis into one int8 byte.

    Input values must be in [-8, 7]. The low nibble holds the first element;
    nibbles are stored as two's-complement 4-bit values (8..15 mean -8..-1).
    Groups along the last axis stay aligned because 2 divides any group size.
    """
    if q8.shape[-1] % 2 != 0:
        raise ValueError("last dim must be even for int4 packing")
    q = q8.to(torch.int16)
    lo = q[..., 0::2] & 0x0F
    hi = q[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_int4`` (sign-extends nibbles 8..15 to -8..-1)."""
    shape = packed.shape
    p = packed.to(torch.int16)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*shape[:-1], shape[-1] * 2).to(torch.int8)