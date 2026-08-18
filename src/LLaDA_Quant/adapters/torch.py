"""Generic PyTorch adapter: explicitly named ``nn.Linear`` replacement."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..config import QuantConfig, matches_linear
from ..formats.manifest import TargetedModule
from ..runtime.linear import QuantLinear


def find_linears(model: nn.Module, config: QuantConfig) -> List[tuple[str, nn.Linear]]:
    """Every ``nn.Linear`` explicitly named by ``config.linear_include``."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and matches_linear(name, config)
    ]


def replace_linears(model: nn.Module, config: QuantConfig) -> List[TargetedModule]:
    """Replace every explicitly named ``nn.Linear`` with a :class:`QuantLinear`.

    Returns one record per replaced module. Nothing is matched implicitly: an
    empty ``linear_include`` replaces nothing, by design.
    """
    compute_dtype = getattr(torch, config.compute_dtype)
    records: List[TargetedModule] = []
    for name, module in find_linears(model, config):
        source_bytes = module.weight.numel() * module.weight.element_size()
        if module.bias is not None:
            source_bytes += module.bias.numel() * module.bias.element_size()
        quantized = QuantLinear.from_linear(
            module,
            bits=config.bits,
            group_size=config.group_size,
            compute_dtype=compute_dtype,
            scale_search=config.scale_search,
            search_grid=config.search_grid,
        )
        *parts, leaf = name.split(".")
        parent = model
        for part in parts:
            parent = getattr(parent, part)
        setattr(parent, leaf, quantized)
        records.append(
            TargetedModule(
                name=name,
                kind="linear",
                module_type="Linear",
                shapes={"weight": [module.out_features, module.in_features]},
                bits=config.bits,
                group_size=quantized.group_size if quantized._num_groups() > 1 else -1,
                packed=quantized.packed,
                execution_mode=config.execution_mode,
                source_bytes=source_bytes,
                quantized_bytes=quantized.storage_bytes(),
            )
        )
    return records
