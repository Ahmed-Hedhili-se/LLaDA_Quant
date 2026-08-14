"""Public API: quantize_model / save_quantized_checkpoint / load_quantized_weights.

Typical use::

    config = QuantConfig(
        bits=8, group_size=128,
        targets=("expert", "linear"),
        exclude=("router", "norm", "embed_tokens", "lm_head"),
    )
    quantize_model(model, config)
    save_quantized_checkpoint(model, QuantizationManifest(config=config, ...), "llada-moe-int8")

Later (or in another process)::

    load_quantized_weights(model, "llada-moe-int8")
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .adapters.llada_moe import quantize_llada_experts
from .adapters.torch import replace_linears
from .config import COMPONENT_EXPERT, COMPONENT_LINEAR, QuantConfig
from .formats.manifest import QuantizationManifest
from .formats.safetensors import load_quantized_weights, save_quantized_checkpoint

__all__ = [
    "QuantConfig",
    "QuantizationManifest",
    "quantize_model",
    "save_quantized_checkpoint",
    "load_quantized_weights",
]


def quantize_model(model: nn.Module, config: QuantConfig) -> list[str]:
    """Quantize ``model`` in place according to ``config``.

    Returns the list of quantized module names.

    - ``targets=("expert",)``: LLaDA fused expert ``w1``/``w2`` tensors
      (runs through ``TritonFusedMoEBlock`` unchanged, router untouched).
    - ``targets=("linear",)``: every matching non-expert ``nn.Linear`` is
      swapped for a ``QuantLinear``.
    - ``compute`` stays BF16: activations and write-back dtype are untouched.
    """
    quantized: list[str] = []
    if COMPONENT_EXPERT in config.targets:
        quantized += quantize_llada_experts(model, config)
    if COMPONENT_LINEAR in config.targets:
        quantized += replace_linears(model, config)
    return quantized


def quantized_model(model: nn.Module, config: QuantConfig) -> nn.Module:
    """Non-destructive variant: clone ``model`` first, then ``quantize_model``."""
    import copy

    clone = copy.deepcopy(model)
    quantize_model(clone, config)
    return clone