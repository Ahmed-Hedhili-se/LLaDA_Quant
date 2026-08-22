"""Quantized MoE expert weights (``w1``/``w2``) and their two residency modes.

The quantized layout mirrors the fused tensors used by the Triton MoE path:

    w1: [local_experts, 2 * intermediate_dim, hidden_dim]   (Gate+Up stacked)
    w2: [local_experts, hidden_dim, intermediate_dim]       (Down)

Each expert is quantized independently; groups run along the last (K) axis,
so per-expert per-group scales fall out of the generic algorithm without
special-casing the expert dimension.

Residency
---------
``ExecutionMode.REFERENCE`` keeps dequantized BF16 Parameters next to the
packed buffers. Nothing shrinks — it is ~1.5x the unquantized model — and it
exists only to pin the numerical contract during validation.

``ExecutionMode.PACKED`` deletes the ``w1``/``w2`` Parameters and serves them
from the packed buffers through a property, so ``block.w1`` keeps working
inside an untouched ``fused_moe(x, self.w1, self.w2, ...)`` call while the
BF16 copy no longer exists between calls. Resident memory really drops; the
cost is a dequantize per access, which is why the packed-consuming kernel is
the next milestone rather than a nice-to-have.

The property is installed by pointing the block at a generated subclass of
its own class. That keeps the model repository untouched, which is the whole
point of the adapter, at the price of one piece of controlled magic — see
:func:`install_packed_expert_access`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..algorithms.symmetric import QuantResult, dequantize_tensor, quantize_tensor

W1 = "w1"
W2 = "w2"
EXPERT_PARAM_NAMES = (W1, W2)
PACKED_BUFFERS = ("_qw1", "_sw1", "_qw2", "_sw2")
QUANT_META_ATTR = "_llada_quant_meta"


@dataclass
class QuantExpertWeights:
    """Packed values + scales for one MoE layer's ``w1`` and ``w2``."""

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
        scale_search: str = "amax",
        search_grid: int = 24,
    ) -> "QuantExpertWeights":
        kwargs = dict(
            bits=bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            scale_search=scale_search,
            search_grid=search_grid,
        )
        return cls(w1=quantize_tensor(w1, **kwargs), w2=quantize_tensor(w2, **kwargs))

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w1.dequantize(dtype=dtype), self.w2.dequantize(dtype=dtype)

    def storage_bytes(self) -> int:
        return self.w1.storage_bytes() + self.w2.storage_bytes()


def quant_result_from_buffers(q: torch.Tensor, scale: torch.Tensor, bits: int) -> QuantResult:
    """Rebuild a :class:`QuantResult` from stored tensors alone.

    The effective group size is *derived* from the scale tensor rather than
    taken from the config, because ``quantize_tensor`` silently falls back to
    per-tensor scaling when the group size does not divide K. Trusting the
    config here would dequantize with the wrong grouping and produce garbage
    that still has the right shape.
    """
    packed = bits == 4
    k_logical = q.shape[-1] * (2 if packed else 1)
    num_groups = scale.shape[-1] if scale.dim() else 1
    group_size = -1 if num_groups <= 1 else k_logical // num_groups
    return QuantResult(
        q=q,
        scale=scale,
        bits=bits,
        group_size=group_size,
        packed=packed,
        logical_shape=tuple(list(q.shape[:-1]) + [k_logical]),
    )


def quantize_fused_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float32,
    scale_search: str = "amax",
    search_grid: int = 24,
) -> QuantExpertWeights:
    """Convenience wrapper: quantize a fused ``w1``/``w2`` pair."""
    return QuantExpertWeights.quantize(
        w1, w2, bits=bits, group_size=group_size, scale_dtype=scale_dtype,
        scale_search=scale_search, search_grid=search_grid,
    )


def materialize_expert_params(
    module: nn.Module,
    weights: QuantExpertWeights,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Copy dequantized weights into a module's live ``w1``/``w2`` Parameters.

    REFERENCE mode only. In PACKED mode the Parameters no longer exist and
    this raises rather than silently writing into a temporary.
    """
    if is_packed_expert_block(module):
        raise RuntimeError(
            "materialize_expert_params() called on a PACKED block: w1/w2 are "
            "derived from the packed buffers and cannot be assigned. Use "
            "ExecutionMode.REFERENCE if you need writable BF16 Parameters."
        )
    w1, w2 = weights.dequantize(dtype=compute_dtype)
    module.w1.data.copy_(w1)
    module.w2.data.copy_(w2)


# ---------------------------------------------------------------------------
# PACKED residency
# ---------------------------------------------------------------------------


def attach_packed_buffers(
    block: nn.Module,
    weights: QuantExpertWeights,
    compute_dtype: torch.dtype,
    compile_dequant: bool = False,
) -> None:
    """Register ``_qw1``/``_sw1``/``_qw2``/``_sw2`` and the metadata to read them."""
    block.register_buffer("_qw1", weights.w1.q, persistent=True)
    block.register_buffer("_sw1", weights.w1.scale, persistent=True)
    block.register_buffer("_qw2", weights.w2.q, persistent=True)
    block.register_buffer("_sw2", weights.w2.scale, persistent=True)
    setattr(
        block,
        QUANT_META_ATTR,
        {
            "bits": weights.w1.bits,
            "group_size": weights.w1.group_size,
            "packed": weights.w1.packed,
            "compute_dtype": compute_dtype,
            "compile_dequant": compile_dequant,
            "logical_shape": {W1: weights.w1.logical_shape, W2: weights.w2.logical_shape},
        },
    )


_COMPILED_DEQUANT = None


def compiled_dequantize():
    """``dequantize_tensor`` fused into a single elementwise kernel.

    Eager dequantization runs four kernels — unpack, upcast, scale, downcast —
    each making its own round trip through HBM. On one 268M-element expert
    tensor that is 7.49 ms against the 0.77 ms it costs merely to read the BF16
    weight. Compiled it is **1.22 ms**, against a 1.19 ms floor for reading the
    packed bytes and writing BF16, so the fusion is essentially optimal for an
    out-of-kernel dequantize.

    It cannot reach BF16 parity, and no amount of compilation will: the
    expanded weight still has to land in HBM for ``fused_moe`` to read back.
    Removing *that* round trip needs the dequantize inside the GEMM's inner
    loop. This is the cheap 6x, not the fix.

    Compiled lazily and cached: ``torch.compile`` costs seconds on first call
    and the guards then specialise per shape.
    """
    global _COMPILED_DEQUANT
    if _COMPILED_DEQUANT is None:
        _COMPILED_DEQUANT = torch.compile(dequantize_tensor, dynamic=False)
    return _COMPILED_DEQUANT


def _packed_expert_weight(self: nn.Module, which: str) -> torch.Tensor:
    meta = getattr(self, QUANT_META_ATTR, None)
    if meta is None:
        raise RuntimeError(
            f"{type(self).__name__}.{which} is served from packed buffers but "
            f"{QUANT_META_ATTR} is missing. Reload through "
            "load_quantized_weights() so the quantization config is applied."
        )
    suffix = which[-1]
    dequantize = compiled_dequantize() if meta.get("compile_dequant") else dequantize_tensor
    return dequantize(
        getattr(self, f"_qw{suffix}"),
        getattr(self, f"_sw{suffix}"),
        meta["bits"],
        meta["group_size"],
        dtype=meta["compute_dtype"],
        packed=meta["packed"],
    )


#: Methods that write into ``w1``/``w2`` in place. In PACKED mode those
#: writes would land in a freshly dequantized temporary and vanish without
#: error, so the generated subclass shadows them with a loud failure.
#: ``TritonFusedMoEBlock.load_state_dict_from_unfused`` does exactly this
#: (``self.w1[i].copy_(...)``) and runs at model build time, which is why
#: quantization must come after weight loading, not before.
WEIGHT_MUTATING_METHODS = ("load_state_dict_from_unfused",)


def _blocked_method(name: str):
    def blocked(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__mro__[1].__name__}.{name}() writes into w1/w2 in place, "
            "but this block is quantized in PACKED mode where those are derived "
            "from packed buffers — the write would be silently discarded.\n"
            "Load the weights first, then quantize:\n"
            f"    block.{name}(...)\n"
            "    quantize_model(model, config)"
        )

    blocked.__name__ = name
    return blocked


_PACKED_CLASS_CACHE: Dict[type, type] = {}


def _packed_class(cls: type) -> type:
    """Subclass of ``cls`` whose ``w1``/``w2`` dequantize from packed buffers.

    Cached per base class so repeated quantization does not leak classes and
    so ``deepcopy`` of a quantized model stays cheap.
    """
    if getattr(cls, "_llada_quant_packed", False):
        return cls
    cached = _PACKED_CLASS_CACHE.get(cls)
    if cached is not None:
        return cached
    namespace = {
        "_llada_quant_packed": True,
        "_packed_expert_weight": _packed_expert_weight,
        W1: property(lambda self: self._packed_expert_weight(W1)),
        W2: property(lambda self: self._packed_expert_weight(W2)),
        "__doc__": f"{cls.__name__} with w1/w2 served from packed integer buffers.",
    }
    for method in WEIGHT_MUTATING_METHODS:
        if hasattr(cls, method):
            namespace[method] = _blocked_method(method)
    generated = type(f"PackedQuant{cls.__name__}", (cls,), namespace)
    _PACKED_CLASS_CACHE[cls] = generated
    return generated


def install_packed_expert_access(block: nn.Module) -> None:
    """Drop the BF16 ``w1``/``w2`` Parameters and serve them from packed buffers.

    After this call ``block.w1`` returns a freshly dequantized tensor on every
    access and ``state_dict()`` no longer contains a BF16 copy. Assignment to
    ``block.w1`` raises (``property`` without a setter), which is deliberate:
    a silent write into a temporary is exactly the failure this mode exists to
    prevent.
    """
    missing = [name for name in PACKED_BUFFERS if not hasattr(block, name)]
    if missing:
        raise RuntimeError(f"cannot install packed access, missing buffers: {missing}")
    for name in EXPERT_PARAM_NAMES:
        block._parameters.pop(name, None)
    block.__class__ = _packed_class(type(block))


def is_packed_expert_block(module: nn.Module) -> bool:
    """True when ``w1``/``w2`` are derived from packed buffers, not stored."""
    return getattr(type(module), "_llada_quant_packed", False)


def expert_storage_bytes(block: nn.Module) -> int:
    """Bytes the packed representation occupies on this block."""
    total = 0
    for name in PACKED_BUFFERS:
        buf = getattr(block, name, None)
        if buf is not None:
            total += buf.numel() * buf.element_size()
    return total
