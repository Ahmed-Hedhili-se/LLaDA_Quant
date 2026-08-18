"""Resident memory accounting.

Every memory number this project reports comes from here, and every number
here is derived from tensors that are *actually alive on the module tree* —
never from the theoretical size of a packed representation.

The distinction matters because it is exactly where the previous version was
wrong: it reported the size of the packed buffers while the dequantized BF16
copies were still resident next to them, so a run that grew the model by 52%
was reported as a 47% saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class MemoryReport:
    """Resident bytes on a module tree, split by role.

    Attributes:
        parameters: Bytes held by ``nn.Parameter`` objects.
        buffers: Bytes held by registered buffers (packed ints, scales, ...).
        by_dtype: Resident bytes per dtype name, over params and buffers.
        tensor_count: Number of distinct resident tensors counted.
    """

    parameters: int = 0
    buffers: int = 0
    by_dtype: Dict[str, int] = field(default_factory=dict)
    tensor_count: int = 0

    @property
    def total(self) -> int:
        return self.parameters + self.buffers

    def to_dict(self) -> dict:
        return {
            "parameters_bytes": self.parameters,
            "buffers_bytes": self.buffers,
            "total_bytes": self.total,
            "total_mib": round(self.total / 2**20, 3),
            "by_dtype_bytes": dict(sorted(self.by_dtype.items())),
            "tensor_count": self.tensor_count,
        }


def resident_memory(module: nn.Module) -> MemoryReport:
    """Measure what is actually resident on ``module``.

    Shared storages are counted once (keyed by ``data_ptr``), so a tied weight
    is not double-billed.
    """
    report = MemoryReport()
    seen: set[int] = set()

    def account(tensor: torch.Tensor, is_param: bool) -> None:
        if tensor is None:
            return
        key = tensor.untyped_storage().data_ptr() if tensor.numel() else id(tensor)
        if key in seen:
            return
        seen.add(key)
        nbytes = tensor.numel() * tensor.element_size()
        if is_param:
            report.parameters += nbytes
        else:
            report.buffers += nbytes
        name = str(tensor.dtype).replace("torch.", "")
        report.by_dtype[name] = report.by_dtype.get(name, 0) + nbytes
        report.tensor_count += 1

    for param in module.parameters(recurse=True):
        account(param.data, True)
    for buf in module.buffers(recurse=True):
        account(buf, False)
    return report


@dataclass
class MemoryComparison:
    """Before/after resident memory for one quantization run."""

    baseline: MemoryReport
    quantized: MemoryReport
    label: str = ""

    @property
    def ratio(self) -> float:
        """Quantized total divided by baseline total. Below 1.0 is a saving."""
        return self.quantized.total / self.baseline.total if self.baseline.total else 1.0

    @property
    def saved_bytes(self) -> int:
        """Positive when memory was saved, negative when it grew."""
        return self.baseline.total - self.quantized.total

    @property
    def is_saving(self) -> bool:
        return self.saved_bytes > 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "baseline": self.baseline.to_dict(),
            "quantized": self.quantized.to_dict(),
            "ratio": round(self.ratio, 4),
            "saved_bytes": self.saved_bytes,
            "saved_pct": round(100 * (1 - self.ratio), 2),
            "is_saving": self.is_saving,
        }

    def describe(self) -> str:
        verdict = "saving" if self.is_saving else "REGRESSION"
        return (
            f"{self.label or 'resident memory'}: "
            f"{self.baseline.total / 2**20:.2f} MiB -> {self.quantized.total / 2**20:.2f} MiB "
            f"({self.ratio:.3f}x, {100 * (1 - self.ratio):+.1f}% — {verdict})"
        )


def compare_resident_memory(
    baseline: nn.Module, quantized: nn.Module, label: str = ""
) -> MemoryComparison:
    """Resident-memory delta between an unquantized and a quantized model."""
    return MemoryComparison(
        baseline=resident_memory(baseline), quantized=resident_memory(quantized), label=label
    )
