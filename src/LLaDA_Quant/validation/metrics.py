"""Numerical, routing and masked-token metrics for quantization validation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


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


# --------------------------------------------------------------------------
# Masked-token metrics (diffusion LMs)
#
# A masked diffusion LM emits logits for every position, but only the masked
# ones matter: they decide which token is written and which position is
# unmasked next. Metrics below therefore take an optional boolean
# ``positions`` mask and compare only those slots.
# --------------------------------------------------------------------------


def _select(logits: torch.Tensor, positions: Optional[torch.Tensor]) -> torch.Tensor:
    """Flatten ``logits`` to ``[N, vocab]``, keeping only ``positions`` if given."""
    if positions is None:
        return logits.reshape(-1, logits.shape[-1])
    if tuple(positions.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            f"positions shape {tuple(positions.shape)} does not match "
            f"logits leading shape {tuple(logits.shape[:-1])}"
        )
    return logits[positions.bool()]


def top1_agreement(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> float:
    """Fraction of positions where both models argmax to the same token id.

    Returns 1.0 when there is nothing to compare (no selected positions).
    """
    a = _select(logits_a, positions)
    b = _select(logits_b, positions)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 1.0
    return (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item()


def kl_divergence(
    logits_ref: torch.Tensor,
    logits_other: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> float:
    """Mean ``KL(reference || other)`` in nats over the selected positions.

    Asymmetric on purpose: the BF16 reference is the target distribution.
    """
    a = _select(logits_ref, positions).float()
    b = _select(logits_other, positions).float()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0:
        return 0.0
    log_p = F.log_softmax(a, dim=-1)
    log_q = F.log_softmax(b, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean().item()


def top2_margin(
    logits: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> float:
    """Mean gap between the best and second-best logit over selected positions.

    How decisive the model is. Read it next to the logit error: a disagreement
    where the margin is far below the perturbation is a coin-toss, not damage.
    """
    a = _select(logits, positions).float()
    if a.numel() == 0 or a.shape[-1] < 2:
        return 0.0
    top2 = a.topk(2, dim=-1).values
    return (top2[..., 0] - top2[..., 1]).mean().item()


def tie_fraction(
    logits_ref: torch.Tensor,
    logits_other: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> float:
    """Fraction of positions where the reference was undecided anyway.

    A position counts as tied when the reference's own top-2 margin is smaller
    than the largest logit shift quantization introduced there: the argmax may
    flip, but the reference had no real preference to destroy. A high
    ``top1_agreement`` drop paired with a high tie fraction is noise; the same
    drop at a low tie fraction is genuine degradation. Report both or the
    disagreement number cannot be interpreted.
    """
    a = _select(logits_ref, positions).float()
    b = _select(logits_other, positions).float()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() == 0 or a.shape[-1] < 2:
        return 0.0
    top2 = a.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    perturbation = (a - b).abs().max(dim=-1).values
    return (margin < perturbation).float().mean().item()


def unmask_selection_agreement(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask_positions: torch.Tensor,
    k: int = 1,
) -> float:
    """Overlap of the *next positions each model would unmask*.

    Confidence-based remasking decoders (LLaDA's default) unmask the ``k``
    masked positions with the highest max-softmax probability. Two models can
    agree on every predicted token yet still unmask in a different order,
    which changes the context all later steps condition on. This measures that
    ordering directly: ``|A ∩ B| / |A|``, averaged over batch rows.

    Rows with no masked position are skipped; 1.0 if every row is skipped.
    """
    if logits_a.dim() == 2:  # [L, V] -> [1, L, V]
        logits_a = logits_a.unsqueeze(0)
        logits_b = logits_b.unsqueeze(0)
        mask_positions = mask_positions.unsqueeze(0)
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"shape mismatch: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    mask = mask_positions.bool()
    conf_a = F.softmax(logits_a.float(), dim=-1).max(dim=-1).values  # [B, L]
    conf_b = F.softmax(logits_b.float(), dim=-1).max(dim=-1).values

    scores: list[float] = []
    for row in range(conf_a.shape[0]):
        idx = mask[row].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        take = min(k, int(idx.numel()))
        top_a = idx[conf_a[row, idx].topk(take).indices]
        top_b = idx[conf_b[row, idx].topk(take).indices]
        shared = len(set(top_a.tolist()) & set(top_b.tolist()))
        scores.append(shared / take)
    if not scores:
        return 1.0
    return sum(scores) / len(scores)