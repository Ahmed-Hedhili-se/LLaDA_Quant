"""Trajectory validation: MODEL -> CAPTURE -> TRACE -> METRICS -> REPORT.

Only :mod:`~LLaDA_Quant.trajectory.capture` touches a model or a GPU.
Everything downstream works from a JSON trace, so metrics can be recomputed,
re-cut and unit-tested without rerunning anything.
"""

from .capture import (
    DEFAULT_TOP_K,
    FreeRunCapture,
    GatesFn,
    SharedCapture,
    capture_free_running,
    capture_shared,
)
from .llada import (
    LLADA_MASK_ID,
    LLaDADecoder,
    RouterCapture,
    assert_matches_production_decoder,
    attach_router_capture,
    gates_fn_for,
    load_llada_decoder,
    make_llada_advance_fn,
    router_fn_for,
)
from .metrics import (
    commit_order_agreement,
    predictive_entropy,
    router_gate_entropy,
    router_margin,
    router_overlap,
    tie_fraction,
    top1_agreement,
    top2_margin,
    topk_kl_lower_bound,
    unmask_selection_agreement,
)
from .replay import ReplayReport, ReplayedStep, replay_free_running, replay_shared, verify_replay
from .report import TrajectoryReport
from .state import (
    AdvanceFn,
    DiffusionState,
    LogitsFn,
    RouterFn,
    fully_masked_state,
    make_masked_states,
    mask_positions_from_ids,
)
from .trace import LayerStats, MetricPrecision, ScalarMetric, Trace, TraceStep

__all__ = [
    "DiffusionState",
    "LogitsFn",
    "RouterFn",
    "AdvanceFn",
    "GatesFn",
    "make_masked_states",
    "fully_masked_state",
    "mask_positions_from_ids",
    "Trace",
    "TraceStep",
    "LayerStats",
    "ScalarMetric",
    "MetricPrecision",
    "capture_shared",
    "capture_free_running",
    "SharedCapture",
    "FreeRunCapture",
    "DEFAULT_TOP_K",
    "replay_shared",
    "replay_free_running",
    "verify_replay",
    "ReplayReport",
    "ReplayedStep",
    "TrajectoryReport",
    "LLaDADecoder",
    "load_llada_decoder",
    "make_llada_advance_fn",
    "assert_matches_production_decoder",
    "RouterCapture",
    "attach_router_capture",
    "router_fn_for",
    "gates_fn_for",
    "LLADA_MASK_ID",
    "top1_agreement",
    "top2_margin",
    "tie_fraction",
    "unmask_selection_agreement",
    "router_overlap",
    "router_margin",
    "router_gate_entropy",
    "predictive_entropy",
    "topk_kl_lower_bound",
    "commit_order_agreement",
]
