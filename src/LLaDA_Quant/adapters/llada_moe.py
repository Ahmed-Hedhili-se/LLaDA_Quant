"""LLaDA-MoE adapter: structural detection of fused expert blocks.

Targets the fused expert tensors of ``TritonFusedMoEBlock``:

    block.w1: Parameter [local_experts, 2 * intermediate_dim, hidden_dim]
    block.w2: Parameter [local_experts, hidden_dim, intermediate_dim]

Detection is **structural**, never name-based. The four shape relations below
pin the layout exactly, so a module that merely happens to be called
``expert`` or ``mlp`` is not touched, and a correctly shaped block with an
unexpected name is not missed:

    w1.shape[0] == w2.shape[0]        same expert count
    w1.shape[1] == 2 * w2.shape[2]    w1 is Gate+Up stacked over I
    w1.shape[2] == w2.shape[1]        both agree on the hidden dim H

The model repository stays untouched: either the dequantized values are
written back into the live Parameters (REFERENCE) or the Parameters are
replaced by packed buffers behind a property (PACKED).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ExecutionMode, QuantConfig, is_excluded
from ..formats.manifest import TargetedModule
from ..runtime.moe import (
    PACKED_BUFFERS,
    QuantExpertWeights,
    attach_packed_buffers,
    install_packed_expert_access,
    is_packed_expert_block,
    materialize_expert_params,
    quant_result_from_buffers,
)


@dataclass(frozen=True)
class ExpertBlockShape:
    """The four numbers that identify a fused LLaDA expert block."""

    num_experts: int
    hidden: int
    intermediate: int

    @property
    def numel(self) -> int:
        """Total expert weight elements: w1 [E, 2I, H] plus w2 [E, H, I]."""
        return self.num_experts * (2 * self.intermediate * self.hidden + self.hidden * self.intermediate)

    def describe(self) -> str:
        return (
            f"E={self.num_experts} H={self.hidden} I={self.intermediate} "
            f"(w1 [{self.num_experts}, {2 * self.intermediate}, {self.hidden}], "
            f"w2 [{self.num_experts}, {self.hidden}, {self.intermediate}])"
        )


def describe_fused_expert_block(module: nn.Module) -> Optional[ExpertBlockShape]:
    """Return the block's shape if it is a fused expert block, else ``None``.

    Accepts both an unquantized block (``w1``/``w2`` Parameters) and one
    already in PACKED mode (``w1``/``w2`` served from buffers).
    """
    w1 = getattr(module, "w1", None)
    w2 = getattr(module, "w2", None)
    if not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor):
        return None
    if not is_packed_expert_block(module) and not (
        isinstance(w1, nn.Parameter) and isinstance(w2, nn.Parameter)
    ):
        return None
    if w1.dim() != 3 or w2.dim() != 3:
        return None
    experts, two_i, hidden = w1.shape
    experts2, hidden2, intermediate = w2.shape
    if experts != experts2 or hidden != hidden2:
        return None
    if two_i != 2 * intermediate:
        return None
    return ExpertBlockShape(num_experts=experts, hidden=hidden, intermediate=intermediate)


def is_fused_expert_block(module: nn.Module) -> bool:
    """True when ``module`` matches the fused LLaDA expert layout exactly."""
    return describe_fused_expert_block(module) is not None


def find_expert_blocks(
    model: nn.Module, config: QuantConfig
) -> List[Tuple[str, nn.Module, ExpertBlockShape]]:
    """Every structurally matching, non-excluded fused expert block."""
    found: List[Tuple[str, nn.Module, ExpertBlockShape]] = []
    for name, block in model.named_modules():
        if is_excluded(name, config):
            continue
        shape = describe_fused_expert_block(block)
        if shape is not None:
            found.append((name, block, shape))
    return found


def quantize_llada_experts(model: nn.Module, config: QuantConfig) -> List[TargetedModule]:
    """Quantize every fused expert block in place, honouring ``execution_mode``.

    Returns one :class:`TargetedModule` record per block — the audit trail that
    makes a run reproducible and lets the caller assert what was touched.
    """
    compute_dtype = getattr(torch, config.compute_dtype)
    scale_dtype = getattr(torch, config.scale_dtype)
    mode = config.mode
    records: List[TargetedModule] = []

    for name, block, shape in find_expert_blocks(model, config):
        if is_packed_expert_block(block):
            raise RuntimeError(f"{name!r} is already quantized in PACKED mode")
        bf16_bytes = sum(
            p.numel() * p.element_size() for p in (block.w1, block.w2)
        )
        weights = QuantExpertWeights.quantize(
            block.w1.detach(),
            block.w2.detach(),
            bits=config.bits,
            group_size=config.group_size,
            scale_dtype=scale_dtype,
            scale_search=config.scale_search,
            search_grid=config.search_grid,
            dtype=config.dtype,
        )
        attach_packed_buffers(block, weights, compute_dtype, config.compile_dequant)

        if mode is ExecutionMode.PACKED:
            install_packed_expert_access(block)
        else:
            materialize_expert_params(block, weights, compute_dtype=compute_dtype)

        records.append(
            TargetedModule(
                name=name,
                kind="expert",
                module_type=type(block).__mro__[1].__name__
                if mode is ExecutionMode.PACKED
                else type(block).__name__,
                shapes={
                    "w1": list(weights.w1.logical_shape),
                    "w2": list(weights.w2.logical_shape),
                },
                bits=config.bits,
                group_size=weights.w1.group_size,
                packed=weights.w1.packed,
                execution_mode=mode.value,
                source_bytes=bf16_bytes,
                quantized_bytes=weights.storage_bytes(),
            )
        )
    return records


def restore_llada_experts_from_buffers(model: nn.Module, config: QuantConfig) -> int:
    """Re-establish expert weight access after loading a quantized checkpoint.

    PACKED blocks get their metadata and property re-installed; REFERENCE
    blocks get their BF16 Parameters rewritten from the packed buffers.
    Returns the number of blocks touched.
    """
    compute_dtype = getattr(torch, config.compute_dtype)
    touched = 0
    for block in model.modules():
        if not all(hasattr(block, b) for b in PACKED_BUFFERS):
            continue
        weights = QuantExpertWeights(
            w1=quant_result_from_buffers(block._qw1, block._sw1, config.bits, qtype=config.dtype),
            w2=quant_result_from_buffers(block._qw2, block._sw2, config.bits, qtype=config.dtype),
        )
        attach_packed_buffers(block, weights, compute_dtype, config.compile_dequant)
        if config.mode is ExecutionMode.PACKED:
            if not is_packed_expert_block(block):
                install_packed_expert_access(block)
        else:
            materialize_expert_params(block, weights, compute_dtype=compute_dtype)
        touched += 1
    return touched
