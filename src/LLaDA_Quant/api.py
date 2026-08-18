"""Public API: quantize a model, then save / load the artifact.

A run is only useful if you can tell afterwards what it touched, so
:func:`quantize_model` returns a :class:`QuantizationResult` audit trail
rather than a bare list of names, and refuses to succeed quietly when it
matched nothing.

Typical use::

    config = QuantConfig(bits=8, group_size=128, targets=("expert",),
                         execution_mode="packed", expect_expert_blocks=16)
    result = quantize_model(model, config)
    print(result.summary())
    save_quantized_checkpoint(model, QuantizationManifest(config=config,
                                                          targets=result.targets),
                              "llada-moe-int8")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn as nn

from .adapters.llada_moe import quantize_llada_experts
from .adapters.torch import replace_linears
from .config import COMPONENT_EXPERT, COMPONENT_LINEAR, ExecutionMode, QuantConfig
from .formats.manifest import QuantizationManifest, TargetedModule
from .formats.safetensors import load_quantized_weights, save_quantized_checkpoint
from .memory import MemoryComparison, compare_resident_memory, resident_memory

__all__ = [
    "QuantConfig",
    "QuantizationManifest",
    "QuantizationResult",
    "TargetingError",
    "quantize_model",
    "quantized_model",
    "save_quantized_checkpoint",
    "load_quantized_weights",
]


class TargetingError(RuntimeError):
    """Raised when a run touched a different set of modules than intended."""


@dataclass
class QuantizationResult:
    config: QuantConfig
    targets: List[TargetedModule] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [t.name for t in self.targets]

    @property
    def expert_blocks(self) -> List[TargetedModule]:
        return [t for t in self.targets if t.kind == "expert"]

    @property
    def linears(self) -> List[TargetedModule]:
        return [t for t in self.targets if t.kind == "linear"]

    @property
    def source_bytes(self) -> int:
        return sum(t.source_bytes for t in self.targets)

    @property
    def quantized_bytes(self) -> int:
        return sum(t.quantized_bytes for t in self.targets)

    @property
    def weight_ratio(self) -> float:
        return self.quantized_bytes / self.source_bytes if self.source_bytes else 1.0

    def summary(self) -> str:
        mode = self.config.mode
        lines = [
            f"quantized {len(self.targets)} module(s): "
            f"{len(self.expert_blocks)} expert block(s), {len(self.linears)} linear(s)",
            f"bits={self.config.bits} group_size={self.config.group_size} "
            f"mode={mode.value} packed={any(t.packed for t in self.targets)}",
            f"converted weights: {self.source_bytes / 2**20:.2f} MiB -> "
            f"{self.quantized_bytes / 2**20:.2f} MiB ({self.weight_ratio:.3f}x)",
        ]
        if mode is ExecutionMode.REFERENCE:
            lines.append(
                "REFERENCE mode: dequantized BF16 weights are still resident next to "
                "the packed ones. The model is LARGER than unquantized. Validation only."
            )
        else:
            lines.append(
                "PACKED mode: BF16 weights are not resident; they are reconstructed per "
                "access. Resident memory drops, per-call latency rises until a packed "
                "kernel lands."
            )
        return "\n".join(lines)


def _validate_targeting(result: QuantizationResult, config: QuantConfig) -> None:
    if not result.targets and not config.allow_no_matches:
        raise TargetingError(
            f"quantization matched no modules for targets={list(config.targets)}. "
            "A silent no-op is indistinguishable from success, so this is an error. "
            "For experts, check the block really has w1 [E, 2I, H] / w2 [E, H, I] "
            "Parameters; for linears, populate linear_include (it has no implicit "
            "default). Pass allow_no_matches=True only in tests."
        )
    if config.expect_expert_blocks is not None:
        found = len(result.expert_blocks)
        if found != config.expect_expert_blocks:
            raise TargetingError(
                f"expected {config.expect_expert_blocks} expert block(s), matched {found}: "
                f"{[t.name for t in result.expert_blocks]}"
            )
    if config.expect_linears is not None:
        found = len(result.linears)
        if found != config.expect_linears:
            raise TargetingError(
                f"expected {config.expect_linears} linear(s), matched {found}: "
                f"{[t.name for t in result.linears]}"
            )


def quantize_model(model: nn.Module, config: QuantConfig) -> QuantizationResult:
    targets: List[TargetedModule] = []
    if COMPONENT_EXPERT in config.targets:
        targets += quantize_llada_experts(model, config)
    if COMPONENT_LINEAR in config.targets:
        targets += replace_linears(model, config)
    result = QuantizationResult(config=config, targets=targets)
    _validate_targeting(result, config)
    return result


def quantized_model(model: nn.Module, config: QuantConfig) -> nn.Module:
    import copy
    clone = copy.deepcopy(model)
    quantize_model(clone, config)
    return clone


def quantize_and_measure(
    model: nn.Module, config: QuantConfig
) -> tuple[nn.Module, QuantizationResult, MemoryComparison]:
    """Quantize a copy and measure the *resident* memory delta against the original.

    The one-call way to get a memory number that cannot be wrong about which
    tensors are alive: both models exist simultaneously and both are measured
    by walking their module trees.
    """
    import copy

    clone = copy.deepcopy(model)
    result = quantize_model(clone, config)
    comparison = compare_resident_memory(
        model, clone, label=f"INT{config.bits} g{config.group_size} {config.execution_mode}"
    )
    return clone, result, comparison
