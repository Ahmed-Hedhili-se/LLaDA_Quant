"""Symmetric (zero-point-free) groupwise integer quantization.

Reference implementation on CPU/GPU via torch. Establishes the numerical
contract the future Triton kernels must reproduce:

    W_q = clamp(round(W / s), -Qmax - 1, Qmax)   bit-exact per group
    s   = max(|W_group|, eps) / Qmax

Groups run along the last (K) axis. An expert-stacked weight of shape
``[E, N, K]`` is therefore quantized per expert and per group without any
reshape of the expert dimension.

**Storage contract.** ``bits=8`` stores one int8 per weight. ``bits=4``
stores *two* weights per int8 byte: element ``2i`` in the low nibble,
element ``2i+1`` in the high nibble, two's complement. INT4 storage is
therefore genuinely half of INT8 — see :func:`storage_bytes`. Because 2
divides every legal group size, group boundaries always land on byte
boundaries, so a kernel can address a group without splitting a byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch


def qmax_for_bits(bits: int) -> int:
    """Largest representable positive integer for signed symmetric quant."""
    return 2 ** (bits - 1) - 1


def qmin_for_bits(bits: int) -> int:
    """Most negative representable integer (``-Qmax - 1``)."""
    return -(2 ** (bits - 1))


@dataclass
class QuantResult:
    """Integer weights plus per-group scales.

    Attributes:
        q: Storage tensor, always ``int8``. When ``packed`` is True this holds
            two 4-bit values per byte and its last dimension is *half* of
            ``logical_shape``'s.
        scale: Per-group scales, shape ``logical_shape[:-1] + (num_groups,)``.
        bits: Bit-width used (8 or 4).
        group_size: Group size used (``-1`` means per-tensor).
        packed: True when ``q`` holds two int4 values per byte.
        logical_shape: Shape of the original weight tensor. Needed to
            reconstruct when ``packed``; also makes a checkpoint
            self-describing.
    """

    q: torch.Tensor
    scale: torch.Tensor
    bits: int
    group_size: int
    packed: bool = False
    logical_shape: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.logical_shape:
            shape = list(self.q.shape)
            if self.packed:
                shape[-1] *= 2
            self.logical_shape = tuple(shape)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return dequantize_tensor(
            self.q,
            self.scale,
            self.bits,
            self.group_size,
            dtype=dtype,
            packed=self.packed,
        )

    def storage_bytes(self) -> int:
        """Bytes actually occupied by this representation (values + scales)."""
        return self.q.numel() * self.q.element_size() + self.scale.numel() * self.scale.element_size()


def _as_groups(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    if group_size == -1 or w.shape[-1] % group_size != 0:
        return w, 1
    shape = w.shape
    w_g = w.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    return w_g, shape[-1] // group_size


def validate_int4_layout(shape: Sequence[int], group_size: int) -> None:
    """Raise unless ``shape``/``group_size`` admit a byte-aligned int4 packing.

    Two conditions, both about never splitting a value or a group across a
    byte boundary: the packed axis must be even, and so must the group size
    (``-1``/per-tensor is fine — the whole axis is one group).
    """
    if shape[-1] % 2 != 0:
        raise ValueError(
            f"int4 packing needs an even last dimension, got {tuple(shape)}; "
            "the last axis is the K axis and two values share one byte"
        )
    if group_size != -1 and group_size % 2 != 0:
        raise ValueError(
            f"int4 packing needs an even group_size so groups stay byte-aligned, got {group_size}"
        )



SCALE_SEARCH_AMAX = "amax"
SCALE_SEARCH_MSE = "mse"
KNOWN_SCALE_SEARCH = (SCALE_SEARCH_AMAX, SCALE_SEARCH_MSE)


def _amax_scale(w_g: torch.Tensor, bits: int) -> torch.Tensor:
    """The textbook scale: the largest magnitude in the group maps to Qmax."""
    return w_g.abs().amax(dim=-1, keepdim=True).float() / qmax_for_bits(bits)


def search_group_scale(
    w_g: torch.Tensor,
    bits: int,
    grid: int = 24,
    max_shrink: float = 0.5,
) -> torch.Tensor:
    """Per-group scale minimising squared reconstruction error.

    ``s = amax / Qmax`` guarantees nothing is clipped, which is the wrong
    objective: it spends the whole grid accommodating the single largest
    weight, and every other weight in the group pays for it in step size. At 8
    bits that trade is nearly free (256 levels absorb it), at 4 bits it is not
    — 16 levels stretched over an outlier is why INT4 lands around 12%
    relative error while INT8 sits at 0.65%.

    This searches shrink ratios in ``[1 - max_shrink, 1]``, quantizes with
    each, and keeps the one with the lowest per-group L2 error. Deliberately
    clips outliers when the group as a whole comes out ahead.

    **The kernel contract is untouched.** The dequantize formula is still
    ``W ~= W_q * s``, zero-point-free, one scale per group — only the choice
    of ``s`` changes. Nothing downstream (packing, checkpoints, a future
    Triton kernel) needs to know this ran.

    Cost is ``grid`` passes over the tensor, one-off at quantization time.

    Args:
        w_g: Grouped weights, ``[..., num_groups, group_size]``.
        bits: Bit width.
        grid: Candidate ratios to try. More is finer and slower.
        max_shrink: Largest fraction of ``amax`` to clip away. 0.5 means
            scales as small as ``0.5 * amax / Qmax`` are considered.

    Returns:
        Scales shaped ``[..., num_groups, 1]``, float32. Zero groups get 0.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if not 0.0 <= max_shrink < 1.0:
        raise ValueError(f"max_shrink must lie in [0, 1), got {max_shrink}")

    qmax, qmin = qmax_for_bits(bits), qmin_for_bits(bits)
    wf = w_g.float()
    base = _amax_scale(w_g, bits)

    best_scale = base.clone()
    best_err = torch.full_like(base, float("inf"))
    for step in range(grid):
        ratio = 1.0 - max_shrink * step / grid
        candidate = base * ratio
        safe = torch.where(candidate > 0, candidate, torch.ones_like(candidate))
        q = torch.round(wf / safe).clamp(qmin, qmax)
        err = ((q * safe - wf) ** 2).sum(dim=-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, candidate, best_scale)

    return torch.where(base > 0, best_scale, torch.zeros_like(best_scale))


def quantize_tensor(
    w: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float32,
    pack: bool = True,
    scale_search: str = SCALE_SEARCH_AMAX,
    search_grid: int = 24,
) -> QuantResult:
    """Quantize ``w`` symmetrically per group along its last axis.

    Falls back to per-tensor scaling when ``group_size`` does not divide the
    last dimension (or is ``-1``). With ``bits=4`` and ``pack=True`` (the
    default) the result is genuinely half the size of INT8; ``pack=False``
    keeps one int4 value per int8 byte, which is only useful for debugging.
    """
    if bits not in (8, 4):
        raise ValueError(f"bits must be 8 or 4, got {bits}")
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if bits == 4 and pack:
        validate_int4_layout(w.shape, group_size)
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        w = w.to(torch.float32)

    if scale_search not in KNOWN_SCALE_SEARCH:
        raise ValueError(
            f"scale_search must be one of {list(KNOWN_SCALE_SEARCH)}, got {scale_search!r}"
        )

    logical_shape = tuple(w.shape)
    w_g, num_groups = _as_groups(w, group_size)
    qmax = qmax_for_bits(bits)
    if scale_search == SCALE_SEARCH_MSE:
        chosen = search_group_scale(w_g, bits, grid=search_grid)
    else:
        chosen = _amax_scale(w_g, bits)
    scale = chosen.to(scale_dtype)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (
        torch.round(w_g.float() / safe.float())
        .clamp(qmin_for_bits(bits), qmax)
        .to(torch.int8)
    )

    if num_groups == 1:
        scale = scale.reshape(*logical_shape[:-1], 1)
    else:
        scale = scale.reshape(*logical_shape[:-1], num_groups)

    q = q.reshape(logical_shape)
    packed = False
    if bits == 4 and pack:
        q = pack_int4(q)
        packed = True
    return QuantResult(
        q=q,
        scale=scale,
        bits=bits,
        group_size=group_size if num_groups > 1 else -1,
        packed=packed,
        logical_shape=logical_shape,
    )


def dequantize_tensor(
    q: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    group_size: int,
    dtype: torch.dtype = torch.float32,
    packed: bool = False,
) -> torch.Tensor:
    """Reconstruct ``w ~= q * scale`` with the inverse reshape of ``quantize_tensor``.

    Arithmetic runs in float32 regardless of ``dtype``, then the result is
    cast; this keeps scale precision intact when the target dtype is BF16.
    Set ``packed`` when ``q`` holds two int4 values per byte.
    """
    if packed:
        q = unpack_int4(q)
    if group_size == -1 or q.shape[-1] % group_size != 0:
        return (q.to(torch.float32) * scale.to(torch.float32)).to(dtype)
    shape = q.shape
    q_g = q.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    s = scale.reshape(*shape[:-1], shape[-1] // group_size, 1)
    return (q_g.to(torch.float32) * s.to(torch.float32)).reshape(shape).to(dtype)


def pack_int4(q8: torch.Tensor) -> torch.Tensor:
    """Pack adjacent int4 values along the last axis into one int8 byte.

    Input values must be in [-8, 7]. The low nibble holds the even-indexed
    element, the high nibble the odd-indexed one; nibbles are two's
    complement (8..15 decode to -8..-1). Groups along the last axis stay
    byte-aligned because 2 divides any legal group size.
    """
    if q8.shape[-1] % 2 != 0:
        raise ValueError("last dim must be even for int4 packing")
    if q8.numel() and (int(q8.max()) > 7 or int(q8.min()) < -8):
        raise ValueError(
            f"int4 packing needs values in [-8, 7], got [{int(q8.min())}, {int(q8.max())}]"
        )
    q = q8.to(torch.int16)
    lo = q[..., 0::2] & 0x0F
    hi = q[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8).to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_int4`` (sign-extends nibbles 8..15 to -8..-1)."""
    shape = packed.shape
    p = packed.to(torch.int16) & 0xFF
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*shape[:-1], shape[-1] * 2).to(torch.int8)


def storage_bytes(numel: int, bits: int, group_size: int, scale_bytes: int = 4) -> int:
    """Bytes a packed representation of ``numel`` weights occupies.

    Includes per-group scales, which are *not* negligible at INT4: with
    ``group_size=128`` and fp32 scales they add 6.25% on top of the values.
    """
    if bits == 8:
        value_bytes = numel
    elif bits == 4:
        value_bytes = numel // 2
    else:
        raise ValueError(f"bits must be 8 or 4, got {bits}")
    groups = 1 if group_size == -1 else max(1, numel // group_size)
    return value_bytes + groups * scale_bytes


def block_k_is_scale_aligned(block_k: int, group_size: int) -> bool:
    """Can a kernel with K-tile ``block_k`` index scales without splitting a group?

    Scales are per group along K, so a kernel computes
    ``scale_idx = (k_block * block_k) // group_size``. That is only exact when
    a K-tile either spans whole groups or sits entirely inside one — i.e. when
    the two divide each other. ``group_size=-1`` (per-tensor) is always fine:
    there is one scale for the whole axis.
    """
    if group_size == -1:
        return True
    if block_k <= 0 or group_size <= 0:
        return False
    return block_k % group_size == 0 or group_size % block_k == 0


def validate_block_k_alignment(block_k: int, group_size: int) -> None:
    """Raise unless ``block_k`` can address groups exactly.

    Worth calling from an autotuner's search-space pruning. Nothing in a
    Triton kernel enforces this: a tuner free to pick, say, ``BLOCK_SIZE_K=96``
    against ``group_size=128`` produces a kernel that reads the wrong scale for
    part of every tile and returns plausible, wrong numbers.
    """
    if not block_k_is_scale_aligned(block_k, group_size):
        raise ValueError(
            f"BLOCK_SIZE_K={block_k} cannot address group_size={group_size} scales "
            "exactly: one must divide the other, or a K-tile straddles a group "
            "boundary and half of it is dequantized with the wrong scale"
        )


def aligned_block_k_values(group_size: int, candidates: Sequence[int]) -> list[int]:
    """The subset of ``candidates`` a kernel may legally use with these groups."""
    return [k for k in candidates if block_k_is_scale_aligned(k, group_size)]
