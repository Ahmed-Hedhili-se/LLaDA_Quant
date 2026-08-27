"""FP8 (E4M3) weight-only quantization.

Same per-group symmetric contract as :mod:`algorithms.symmetric` -- one
scale per group, zero-point-free -- with the integer round/clamp step
replaced by a direct cast to ``torch.float8_e4m3fn``:

    W_q = (W / s).to(float8_e4m3fn)      hardware rounds to nearest fp8 value
    s   = max(|W_group|, eps) / FP8_E4M3_MAX

This is deliberately **not** the same numeric contract as the INT8 path.
INT8 has a uniform step size within a group (``amax / 127``, the same gap
between every representable value); E4M3 is a floating format, so its step
size shrinks near zero and widens near ``amax`` -- more of its 256
codepoints resolve small weights, fewer resolve large ones. Same 1
byte/weight footprint as INT8, different error distribution: whichever wins
is a question for measurement (see the project's H100 benchmark log), not
something derivable from the format alone.

Hardware motivation: E4M3 exists as a *storage* format here (weight-only,
activations stay BF16, no FP8 GEMM yet -- see the module docstring in
``runtime/moe.py`` for why PACKED without a fused kernel is a deliberate,
precedented intermediate state, not a shortcut). The reason to reach for it
at all is that unlike INT8, native FP8 tensor cores exist on sm_90+ (H100)
-- a future fused kernel here has a real hardware target that INT8-on-H100
does not uniquely have (H100 also has fast INT8 tensor cores; FP8 additionally
skips a dequantize-to-BF16 step if the *activations* are ever cast to FP8
too, which this module does not do).
"""

from __future__ import annotations

from typing import Tuple

import torch

from .symmetric import QuantResult, _as_groups

#: Largest finite magnitude ``torch.float8_e4m3fn`` can represent. Values
#: with |x| > this after scaling clamp to +-448 on cast (E4M3 has no Inf;
#: it trades that exponent pattern for one extra representable magnitude).
FP8_E4M3_MAX = 448.0

FP8_DTYPE = torch.float8_e4m3fn
QTYPE = "fp8_e4m3"


def _amax_scale_fp8(w_g: torch.Tensor) -> torch.Tensor:
    """Map the group's largest magnitude onto E4M3's largest magnitude."""
    return w_g.abs().amax(dim=-1, keepdim=True).float() / FP8_E4M3_MAX


def quantize_tensor_fp8(
    w: torch.Tensor,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float32,
) -> QuantResult:
    """Quantize ``w`` to E4M3 symmetrically per group along its last axis.

    Falls back to per-tensor scaling when ``group_size`` does not divide the
    last dimension (or is ``-1``), matching ``quantize_tensor``'s contract.
    Always unpacked (1 byte/value) -- there is no sub-byte FP8 packing here,
    unlike INT4.
    """
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        w = w.to(torch.float32)

    logical_shape = tuple(w.shape)
    w_g, num_groups = _as_groups(w, group_size)
    scale = _amax_scale_fp8(w_g).to(scale_dtype)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    # The cast to float8_e4m3fn does the rounding; clamp first so an
    # out-of-range value (shouldn't occur since s is derived from this
    # group's own amax, but a different group's tie-broken scale reuse
    # could in principle push one) saturates predictably instead of
    # relying on cast behavior at the format's edge.
    q = (w_g.float() / safe.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_DTYPE)

    if num_groups == 1:
        scale = scale.reshape(*logical_shape[:-1], 1)
    else:
        scale = scale.reshape(*logical_shape[:-1], num_groups)
    q = q.reshape(logical_shape)

    return QuantResult(
        q=q,
        scale=scale,
        bits=8,
        group_size=group_size if num_groups > 1 else -1,
        packed=False,
        logical_shape=logical_shape,
        qtype=QTYPE,
    )


def dequantize_tensor_fp8(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct ``w ~= q.float() * scale``, inverse of ``quantize_tensor_fp8``.

    ``q.float()`` upcasts the fp8 storage tensor via its own dtype's exact
    cast (no precision loss beyond what the fp8 value already carries);
    arithmetic then runs in float32 before the final cast to ``dtype``, same
    as ``dequantize_tensor``.
    """
    if q.dtype != FP8_DTYPE:
        raise ValueError(f"expected {FP8_DTYPE} storage, got {q.dtype}")
    if group_size == -1 or q.shape[-1] % group_size != 0:
        return (q.float() * scale.to(torch.float32)).to(dtype)
    shape = q.shape
    q_g = q.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    s = scale.reshape(*shape[:-1], shape[-1] // group_size, 1)
    return (q_g.float() * s.to(torch.float32)).reshape(shape).to(dtype)


def storage_bytes(numel: int, group_size: int, scale_bytes: int = 4) -> int:
    """Bytes an FP8 representation of ``numel`` weights occupies.

    Always 1 byte/value (no packing), plus per-group scales -- same formula
    shape as ``symmetric.storage_bytes(bits=8, ...)``.
    """
    groups = 1 if group_size == -1 else max(1, numel // group_size)
    return numel + groups * scale_bytes
