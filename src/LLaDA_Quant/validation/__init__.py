"""Validation: component-level comparison and diffusion-trajectory divergence."""

from .compare import ComponentReport, compare_models
from .diffusion import (
    AdvanceFn,
    DiffusionState,
    LogitsFn,
    RouterFn,
    fully_masked_state,
    make_masked_states,
    mask_positions_from_ids,
)
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
from .trajectory import (
    FreeRunReport,
    FreeRunStep,
    StateReport,
    TrajectoryReport,
    compare_free_running,
    compare_trajectory,
)

__all__ = [
    "ComponentReport",
    "compare_models",
    "DiffusionState",
    "LogitsFn",
    "RouterFn",
    "AdvanceFn",
    "make_masked_states",
    "fully_masked_state",
    "mask_positions_from_ids",
    "StateReport",
    "TrajectoryReport",
    "FreeRunStep",
    "FreeRunReport",
    "compare_trajectory",
    "compare_free_running",
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
