"""Numerical and routing metrics for quantization validation."""

from __future__ import annotations

import torch


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.double() - b.double()).abs().max()


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.double() - b.double()).abs().mean()


def max_rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = a.double().abs().clamp_min(eps)
    return ((a.double() - b.double()).abs() / denom).max()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_f, b_f = a.double().flatten(), b.double().flatten()
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0)


def router_overlap(topk_ids_a: torch.Tensor, topk_ids_b: torch.Tensor) -> float:
    """Fraction of (token, rank) slots where the two routings agree."""
    a = topk_ids_a.reshape(-1)
    b = topk_ids_b.reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return (a == b).float().mean().item()


def summarize_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    return {
        "max_abs_error": float(max_abs_error(a, b)),
        "mean_abs_error": float(mean_abs_error(a, b)),
        "max_rel_error": float(max_rel_error(a, b)),
        "cosine_similarity": float(cosine_similarity(a, b)),
    }