"""Tensor- and component-level comparison.

Trajectory-level validation moved to :mod:`LLaDA_Quant.trajectory`, which
separates capture from metrics so results can be replayed offline.
"""

from .compare import ComponentReport, compare_models
from .metrics import (
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
    "ComponentReport",
    "compare_models",
    "max_abs_error",
    "mean_abs_error",
    "max_rel_error",
    "cosine_similarity",
    "router_overlap",
    "summarize_metrics",
    "top1_agreement",
    "kl_divergence",
    "top2_margin",
    "tie_fraction",
    "unmask_selection_agreement",
]
