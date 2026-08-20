"""The only part of the trajectory stack that touches a model or a GPU.

Capture runs the models, reduces everything that needs full tensors to
scalars *on device*, and writes a compact :class:`~.trace.Trace`. Metrics,
replay and reporting all work from those traces, so they run anywhere and
never need the model again.

Two modes, deliberately separate (see the module docstring of
:mod:`~LLaDA_Quant.trajectory.report` for why conflating them is the classic
error):

``capture_shared``
    Mode A. Both models are fed byte-identical states. Every difference is
    injected quantization error at that step. Cannot show amplification.

``capture_free_running``
    Mode B. Each model advances itself through the caller's commit rule.
    Shows amplification. Cannot separate per-step error from accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import (
    kl_divergence,
    predictive_entropy,
    router_overlap,
    tie_fraction,
    top1_agreement,
    top2_margin,
    unmask_selection_agreement,
)
from .state import AdvanceFn, DiffusionState, LogitsFn, RouterFn
from .trace import LayerStats, MetricPrecision, ScalarMetric, Trace, TraceStep

DEFAULT_TOP_K = 8

#: ``(model, state) -> {layer_name: gates}`` where gates are the raw router
#: scores ``[tokens, num_experts]``. Optional; enables router-margin capture.
GatesFn = Callable[[nn.Module, DiffusionState], Optional[Mapping[str, torch.Tensor]]]


def _exact(value: float, note: str = "") -> ScalarMetric:
    return ScalarMetric(value=float(value), precision=MetricPrecision.EXACT.value, note=note)


def _as_mapping(
    value: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    if value is None:
        return {}
    if isinstance(value, torch.Tensor):
        return {"router": value}
    return dict(value)


def _topk_slice(
    logits: torch.Tensor, mask: torch.Tensor, k: int
) -> tuple[List[List[int]], List[List[float]]]:
    """Top-k ids and log-probs at masked positions, as plain Python lists."""
    selected = logits[mask]
    if selected.numel() == 0:
        return [], []
    k = min(k, selected.shape[-1])
    logprobs = F.log_softmax(selected.float(), dim=-1)
    values, indices = logprobs.topk(k, dim=-1)
    return indices.tolist(), values.tolist()


def _layer_stats(
    routers: Mapping[str, torch.Tensor],
    gates: Mapping[str, torch.Tensor],
    top_k: int,
    store_router_ids: bool,
) -> Dict[str, LayerStats]:
    from .metrics import router_gate_entropy, router_margin

    stats: Dict[str, LayerStats] = {}
    for name in sorted(set(routers) | set(gates)):
        entry = LayerStats()
        if name in gates and gates[name] is not None:
            entry.router_margin = router_margin(gates[name], top_k)
            entry.router_gate_entropy = router_gate_entropy(gates[name])
        if store_router_ids and name in routers:
            ids = routers[name]
            entry.router_topk_ids = ids.reshape(ids.shape[0], -1).tolist() if ids.dim() >= 2 else [
                ids.tolist()
            ]
        stats[name] = entry
    return stats


def _step_record(
    state: DiffusionState,
    logits: torch.Tensor,
    mask: torch.Tensor,
    routers: Mapping[str, torch.Tensor],
    gates: Mapping[str, torch.Tensor],
    top_k: int,
    store_router_ids: bool,
) -> TraceStep:
    ids, logprobs = _topk_slice(logits, mask, top_k)
    record = TraceStep(
        step=state.step,
        num_masked=state.num_masked,
        mask_ratio=state.mask_ratio,
        masked_positions=mask.reshape(-1).nonzero(as_tuple=True)[0].tolist(),
        topk_ids=ids,
        topk_logprobs=logprobs,
        layers=_layer_stats(routers, gates, top_k, store_router_ids),
    )
    record.scalars["entropy_masked"] = _exact(
        predictive_entropy(logits, mask if bool(mask.any()) else None)
    )
    record.scalars["top2_margin"] = _exact(
        top2_margin(logits, mask if bool(mask.any()) else None)
    )
    hidden_absmax = float(logits.abs().max())
    record.scalars["logit_absmax"] = _exact(hidden_absmax)
    return record


@dataclass
class SharedCapture:
    """Mode A result: one trace per model, pairwise scalars on the quantized one."""

    reference: Trace
    quantized: Trace

    def save(self, directory: str, prefix: str = "modeA") -> None:
        import os

        os.makedirs(directory, exist_ok=True)
        self.reference.save(os.path.join(directory, f"{prefix}-reference.json"))
        self.quantized.save(os.path.join(directory, f"{prefix}-quantized.json"))


def capture_shared(
    reference: nn.Module,
    quantized: nn.Module,
    states: Sequence[DiffusionState],
    logits_fn: LogitsFn,
    router_fn: Optional[RouterFn] = None,
    gates_fn: Optional[GatesFn] = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    unmask_k: int = 1,
    store_router_ids: bool = False,
    labels: tuple[str, str] = ("reference", "quantized"),
) -> SharedCapture:
    """Mode A: probe both models on identical denoising states.

    Pairwise quantities (cosine, KL, agreement, tie fraction, router overlap)
    need the full logits of both models at once, so they are reduced here and
    stored as EXACT scalars on the quantized trace under ``pair.*``. Replay
    can recompute truncated versions and check them against these.
    """
    ref_trace = Trace(label=labels[0], mode="shared", top_k_stored=top_k)
    qnt_trace = Trace(label=labels[1], mode="shared", top_k_stored=top_k)

    for state in states:
        with torch.no_grad():
            ref_logits = logits_fn(reference, state)
            ref_routers = _as_mapping(router_fn(reference, state)) if router_fn else {}
            ref_gates = _as_mapping(gates_fn(reference, state)) if gates_fn else {}
            qnt_logits = logits_fn(quantized, state)
            qnt_routers = _as_mapping(router_fn(quantized, state)) if router_fn else {}
            qnt_gates = _as_mapping(gates_fn(quantized, state)) if gates_fn else {}

        if ref_logits.shape != qnt_logits.shape:
            raise ValueError(
                f"logits shape mismatch at step {state.step}: "
                f"{tuple(ref_logits.shape)} vs {tuple(qnt_logits.shape)}"
            )
        mask = state.mask_positions.to(device=ref_logits.device, dtype=torch.bool)
        positions = mask if bool(mask.any()) else None

        ref_step = _step_record(state, ref_logits, mask, ref_routers, ref_gates, top_k, store_router_ids)
        qnt_step = _step_record(state, qnt_logits, mask, qnt_routers, qnt_gates, top_k, store_router_ids)

        sel_ref = ref_logits[positions] if positions is not None else ref_logits
        sel_qnt = qnt_logits[positions] if positions is not None else qnt_logits
        cos = F.cosine_similarity(sel_ref.double().flatten(), sel_qnt.double().flatten(), dim=0)

        qnt_step.scalars.update(
            {
                "pair.logit_cosine": _exact(cos, "full-vocab reduction on device"),
                "pair.max_abs_error": _exact((sel_ref - sel_qnt).abs().max()),
                "pair.kl_masked": _exact(
                    kl_divergence(ref_logits, qnt_logits, positions), "KL(reference || quantized)"
                ),
                "pair.top1_agreement": _exact(top1_agreement(ref_logits, qnt_logits, positions)),
                "pair.tie_fraction": _exact(
                    tie_fraction(ref_logits, qnt_logits, positions),
                    "share of positions where the reference was undecided anyway",
                ),
                "pair.unmask_agreement": _exact(
                    unmask_selection_agreement(ref_logits, qnt_logits, mask, k=unmask_k)
                ),
            }
        )
        for name, ref_ids in ref_routers.items():
            if name in qnt_routers:
                qnt_step.scalars[f"pair.router_overlap.{name}"] = _exact(
                    router_overlap(ref_ids, qnt_routers[name])
                )
        overlaps = [
            m.value for k, m in qnt_step.scalars.items() if k.startswith("pair.router_overlap.")
        ]
        if overlaps:
            qnt_step.scalars["pair.router_overlap_mean"] = _exact(sum(overlaps) / len(overlaps))

        ref_trace.steps.append(ref_step)
        qnt_trace.steps.append(qnt_step)
    return SharedCapture(reference=ref_trace, quantized=qnt_trace)


@dataclass
class FreeRunCapture:
    """Mode B result: one independently evolved trace per model."""

    reference: Trace
    quantized: Trace

    def save(self, directory: str, prefix: str = "modeB") -> None:
        import os

        os.makedirs(directory, exist_ok=True)
        self.reference.save(os.path.join(directory, f"{prefix}-reference.json"))
        self.quantized.save(os.path.join(directory, f"{prefix}-quantized.json"))


def capture_free_running(
    reference: nn.Module,
    quantized: nn.Module,
    initial_state: DiffusionState,
    logits_fn: LogitsFn,
    advance_fn: AdvanceFn,
    *,
    max_steps: int = 64,
    top_k: int = DEFAULT_TOP_K,
    router_fn: Optional[RouterFn] = None,
    labels: tuple[str, str] = ("reference", "quantized"),
    seed: Optional[int] = None,
) -> FreeRunCapture:
    """Mode B: each model denoises itself through the caller's commit rule.

    ``advance_fn`` is *your* decoder's rule — this module supplies none. Both
    models are driven by the same callable, so any divergence comes from the
    logits, not from two different decoding implementations.

    No pairwise logit scalars are recorded: once the trajectories differ, the
    two models are looking at different inputs and a logit distance conflates
    quantization error with input drift. What is recorded is what actually
    diverged — committed positions and committed tokens per step.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    traces = {
        labels[0]: Trace(label=labels[0], mode="free_running", top_k_stored=top_k, seed=seed),
        labels[1]: Trace(label=labels[1], mode="free_running", top_k_stored=top_k, seed=seed),
    }
    for trace in traces.values():
        trace.gen_length = int(initial_state.mask_positions.sum())
        trace.prompt_length = int(initial_state.input_ids.shape[1]) - trace.gen_length

    runners = [
        [labels[0], reference, initial_state, False],
        [labels[1], quantized, initial_state, False],
    ]

    for _ in range(max_steps):
        if all(done for *_, done in runners):
            break
        for runner in runners:
            label, model, state, done = runner
            if done:
                continue
            with torch.no_grad():
                logits = logits_fn(model, state)
                routers = _as_mapping(router_fn(model, state)) if router_fn else {}
            mask = state.mask_positions.to(device=logits.device, dtype=torch.bool)
            record = _step_record(state, logits, mask, routers, {}, top_k, False)

            with torch.no_grad():
                nxt = advance_fn(state, logits)
            if nxt is None:
                runner[3] = True
                traces[label].steps.append(record)
                continue

            newly = (mask & ~nxt.mask_positions.to(mask.device).bool()).reshape(-1)
            committed = newly.nonzero(as_tuple=True)[0]
            flat_ids = nxt.input_ids.reshape(-1)
            record.committed_positions = committed.tolist()
            record.committed_tokens = flat_ids[committed.to(flat_ids.device)].tolist()
            traces[label].steps.append(record)

            runner[2] = nxt
            runner[3] = nxt.num_masked == 0

    return FreeRunCapture(reference=traces[labels[0]], quantized=traces[labels[1]])
