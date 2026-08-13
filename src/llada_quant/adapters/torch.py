"""Generic PyTorch adapter: pattern-based ``nn.Linear`` replacement."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import QuantConfig, matches
from ..runtime.linear import QuantLinear


def replace_linears(model: nn.Module, config: QuantConfig) -> list[str]:
    """Replace every non-excluded ``nn.Linear`` whose name matches ``config.targets``.

    Returns the list of replaced module names.
    """
    compute_dtype = getattr(torch, config.compute_dtype)
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not name:
            continue
        if not matches(name, config):
            continue
        *parts, leaf = name.split(".")
        parent = model
        for part in parts:
            parent = getattr(parent, part)
        quantized = QuantLinear.from_linear(
            module, bits=config.bits, group_size=config.group_size, compute_dtype=compute_dtype
        )
        setattr(parent, leaf, quantized)
        replaced.append(name)
    return replaced