"""Route a PACKED expert block through the fused W8A16 kernel.

``ExecutionMode.PACKED`` makes ``block.w1`` a property that dequantizes on every
access, so an unmodified ``fused_moe(x, self.w1, self.w2, ...)`` keeps working
and the BF16 copy no longer sits in memory between calls. That is what makes the
capacity win possible -- and it is also what makes PACKED ~6x slower than BF16,
because the property expands 768 MiB into HBM for a GEMM that immediately reads
it back.

This module removes that round trip for INT8 blocks. It installs a ``forward``
that never touches ``.w1``/``.w2`` and hands the packed buffers straight to
:func:`LLaDA_Quant.runtime.kernels.w8a16_moe.fused_moe_w8a16`.

Why the routing is duplicated here
----------------------------------
The block's routing (gate -> fp32 softmax -> top-k -> optional TP masking) is a
dozen lines inside ``TritonFusedMoEBlock.forward``, with no seam to call it
separately, so overriding ``forward`` means restating it. That is a real
divergence risk and this package avoids duplication elsewhere on principle --
so it is covered by a test rather than a comment: ``test_fused_block.py``
asserts a packed-fused block and a REFERENCE-mode block produce matching output
on the same input, which fails if the restated routing drifts from the original
in any way that changes which experts run.

INT8 only. Packed INT4 needs an unpack inside the K-loop; :func:`install` leaves
those blocks on the dequantize-per-access path rather than silently producing
wrong numbers.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import QUANT_META_ATTR, is_packed_expert_block

FUSED_FORWARD_ATTR = "_llada_quant_fused_forward"
_FUSED_CLASS_CACHE: dict = {}


def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``TritonFusedMoEBlock.forward`` with the dequantize round trip removed.

    Mirrors the original exactly up to the expert GEMM; only the call changes.
    """
    from .kernels.w8a16_moe import fused_moe_w8a16

    B, T, H = x.shape
    x_flat = x.view(B * T, H)
    router_logits = self.gate(x_flat)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, self.cfg.TOPK, dim=-1)

    if getattr(self, "tp_size", 1) > 1:
        local_expert_end = self.local_expert_start + self.num_local_experts
        mask = (topk_ids >= self.local_expert_start) & (topk_ids < local_expert_end)
        topk_ids = (topk_ids - self.local_expert_start).clamp(
            min=0, max=self.num_local_experts - 1
        )
        topk_weights = topk_weights * mask.float()

    meta = getattr(self, QUANT_META_ATTR, {}) or {}
    out = fused_moe_w8a16(
        x_flat,
        self._qw1, self._sw1,
        self._qw2, self._sw2,
        topk_weights.to(x.dtype),
        topk_ids.to(torch.int32),
        quant_group=int(meta.get("group_size", 128)),
    )
    out = out.view(B, T, H)

    # The block's own all-reduce, resolved through the class the block came
    # from rather than re-imported, so a TP deployment keeps working.
    from model_update.distributed import tp_all_reduce_

    tp_all_reduce_(out)
    return out


def _fused_class(cls: type) -> type:
    if getattr(cls, FUSED_FORWARD_ATTR, False):
        return cls
    cached = _FUSED_CLASS_CACHE.get(cls)
    if cached is not None:
        return cached
    generated = type(
        f"FusedW8A16{cls.__name__}",
        (cls,),
        {
            FUSED_FORWARD_ATTR: True,
            "forward": _fused_forward,
            "__doc__": f"{cls.__name__} whose forward consumes packed INT8 experts directly.",
        },
    )
    _FUSED_CLASS_CACHE[cls] = generated
    return generated


def _is_int8(block: nn.Module) -> bool:
    meta = getattr(block, QUANT_META_ATTR, {}) or {}
    if int(meta.get("bits", 0)) != 8:
        return False
    # Packing halves the K extent; the kernel reads whole bytes as values.
    return getattr(block, "_qw1", None) is not None and not meta.get("packed", False)


def install(model: nn.Module, strict: bool = False) -> List[str]:
    """Switch every eligible PACKED INT8 expert block to the fused kernel.

    Returns the module paths that were switched. Blocks that are not packed, not
    INT8, or hold packed INT4 are left alone -- with ``strict=True`` an
    ineligible *quantized* block raises instead, for callers that would rather
    fail than silently serve the slow path.
    """
    switched: List[str] = []
    for name, module in model.named_modules():
        if not is_packed_expert_block(module):
            continue
        if not _is_int8(module):
            if strict:
                meta = getattr(module, QUANT_META_ATTR, {}) or {}
                raise ValueError(
                    f"{name}: fused W8A16 needs unpacked INT8 experts, got "
                    f"bits={meta.get('bits')} packed={meta.get('packed')}"
                )
            continue
        module.__class__ = _fused_class(type(module))
        switched.append(name)
    return switched


def is_fused_block(module: nn.Module) -> bool:
    return bool(getattr(module, FUSED_FORWARD_ATTR, False))
