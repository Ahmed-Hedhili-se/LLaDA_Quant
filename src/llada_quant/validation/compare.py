"""Model-component comparison harness: reference vs quantized, layer by layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .metrics import router_overlap, summarize_metrics


@dataclass
class ComponentReport:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    router_overlap: float = 1.0


def compare_models(
    reference: torch.nn.Module,
    quantized: torch.nn.Module,
    components: list[str],
    get_inputs_fn: Callable[[str], tuple],
    forward_fn: Callable[[torch.nn.Module, tuple], torch.Tensor],
    router_fn: Callable[[torch.nn.Module, tuple], torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, ComponentReport]:
    """Compare ``components`` (module names) of two models.

    ``forward_fn(model, inputs)`` returns the module output used for metric
    comparison; ``router_fn(model, inputs)`` returns routing ids, if the
    component has a router (returns ``None`` to skip).
    """
    reports: dict[str, ComponentReport] = {}
    for name in components:
        ref_mod = reference.get_submodule(name)
        qnt_mod = quantized.get_submodule(name)
        ref_mod = ref_mod.to(dtype).eval()
        qnt_mod = qnt_mod.to(dtype).eval()
        inputs = get_inputs_fn(name)

        ref_router = router_fn(ref_mod, inputs)
        qnt_router = router_fn(qnt_mod, inputs)
        with torch.no_grad():
            ref_out = forward_fn(ref_mod, inputs)
            qnt_out = forward_fn(qnt_mod, inputs)
        metrics = summarize_metrics(ref_out, qnt_out)
        overlap = 1.0 if ref_router is None else router_overlap(ref_router, qnt_router)
        reports[name] = ComponentReport(name=name, metrics=metrics, router_overlap=overlap)
    return reports