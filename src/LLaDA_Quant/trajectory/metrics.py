"""Metrics for diffusion trajectories.

Tensor-level metrics live in :mod:`LLaDA_Quant.validation.metrics` and are
re-exported here so callers have one import. Added below are the ones that
only make sense for a masked diffusion MoE: how decisive the router was, and
how much of the distribution is actually in play at a masked position.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..validation.metrics import (
    cosine_similarity,
    kl_divergence,
    max_abs_error,
    max_rel_error,
    mean_abs_error,
    router_overlap,
    summarize_metrics,
    tie_fraction,
    top1_agreement,
    top2_margin,
    unmask_selection_agreement,
)

__all__ = [
    "cosine_similarity",
    "kl_divergence",
    "max_abs_error",
    "max_rel_error",
    "mean_abs_error",
    "router_overlap",
    "summarize_metrics",
    "tie_fraction",
    "top1_agreement",
    "top2_margin",
    "unmask_selection_agreement",
    "predictive_entropy",
    "router_margin",
    "router_gate_entropy",
    "topk_kl_lower_bound",
    "commit_order_agreement",
]


def predictive_entropy(logits: torch.Tensor, positions: Optional[torch.Tensor] = None) -> float:
    """Mean Shannon entropy (nats) of the next-token distribution.

    High entropy at masked slots means the model has not decided yet, so a
    disagreement there costs less than the same disagreement at a confident
    slot. Read alongside ``tie_fraction``.
    """
    x = logits if positions is None else logits[positions.bool()]
    x = x.reshape(-1, logits.shape[-1]).float()
    if x.numel() == 0:
        return 0.0
    logp = F.log_softmax(x, dim=-1)
    return float(-(logp.exp() * logp).sum(dim=-1).mean())


def router_margin(gates: torch.Tensor, top_k: int) -> float:
    """Mean gap between the k-th and (k+1)-th router score.

    The quantity that decides whether quantization noise can flip expert
    selection: if the margin is below the perturbation the top-k set is a
    coin toss, exactly as ``tie_fraction`` measures for token logits.
    """
    if gates.dim() < 2:
        raise ValueError(f"gates must be [..., num_experts], got {tuple(gates.shape)}")
    num_experts = gates.shape[-1]
    if top_k >= num_experts:
        return 0.0
    top = gates.float().topk(top_k + 1, dim=-1).values
    return float((top[..., top_k - 1] - top[..., top_k]).mean())


def router_gate_entropy(gates: torch.Tensor) -> float:
    """Mean entropy of the router distribution — how spread the routing is."""
    g = gates.reshape(-1, gates.shape[-1]).float()
    if g.numel() == 0:
        return 0.0
    logp = F.log_softmax(g, dim=-1)
    return float(-(logp.exp() * logp).sum(dim=-1).mean())


def topk_kl_lower_bound(
    ref_topk_logprobs: torch.Tensor,
    other_topk_logprobs: torch.Tensor,
) -> float:
    """KL restricted to the stored top-k support.

    **Not** the KL divergence. Truncating to k terms drops all mass outside
    the reference's top-k, so this is a lower bound on the true value. It is
    what :mod:`~LLaDA_Quant.trajectory.replay` can recompute offline; the
    exact figure has to be reduced on device during capture.
    """
    p = ref_topk_logprobs.float()
    q = other_topk_logprobs.float()
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {tuple(p.shape)} vs {tuple(q.shape)}")
    if p.numel() == 0:
        return 0.0
    return float((p.exp() * (p - q)).sum(dim=-1).mean())


def commit_order_agreement(
    ref_committed: list[list[int]], other_committed: list[list[int]]
) -> float:
    """Fraction of steps where both trajectories committed the same position set.

    Distinct from token agreement: two decodes can write identical tokens in a
    different order, and the order is what later steps condition on.
    """
    if not ref_committed:
        return 1.0
    steps = min(len(ref_committed), len(other_committed))
    if steps == 0:
        return 0.0
    same = sum(
        1 for i in range(steps) if set(ref_committed[i]) == set(other_committed[i])
    )
    return same / max(len(ref_committed), len(other_committed))
