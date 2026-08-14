"""Trajectory-level validation: reference vs quantized across denoising states.

`compare_models` in :mod:`~LLaDA_Quant.validation.compare` answers "how wrong
is this layer on one forward pass". That is the standard PTQ question and it
is structurally blind to the failure mode that matters for a masked diffusion
LM: a small logit shift changes *which position gets unmasked*, that position
becomes context for every later step, and the error compounds along the
schedule. Two entry points here:

``compare_trajectory``
    Teacher-forced. Both models see byte-identical inputs at each denoising
    state, so every difference is attributable to quantization alone. This
    isolates per-step sensitivity — including whether it worsens as the
    sequence fills in — but by construction it cannot show compounding.

``compare_free_running``
    Each model advances its own state through a caller-supplied unmasking
    rule. Inputs drift apart, which is the point: this is where compounding
    becomes visible. Use it for the headline number, and
    ``compare_trajectory`` to explain it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from .diffusion import AdvanceFn, DiffusionState, LogitsFn, RouterFn
from .metrics import (
    kl_divergence,
    router_overlap,
    summarize_metrics,
    tie_fraction,
    top1_agreement,
    top2_margin,
    unmask_selection_agreement,
)

__all__ = [
    "StateReport",
    "TrajectoryReport",
    "FreeRunStep",
    "FreeRunReport",
    "compare_trajectory",
    "compare_free_running",
]


def _as_router_mapping(
    value: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]]
) -> dict[str, torch.Tensor]:
    if value is None:
        return {}
    if isinstance(value, torch.Tensor):
        return {"router": value}
    return dict(value)


@dataclass
class StateReport:
    """Quantization error at one denoising state, teacher-forced."""

    step: int
    label: str
    mask_ratio: float
    num_masked: int
    logit_metrics: dict[str, float] = field(default_factory=dict)
    top1_agreement: float = 1.0
    unmask_agreement: float = 1.0
    kl_masked: float = 0.0
    router_overlap: dict[str, float] = field(default_factory=dict)
    reference_top2_margin: float = 0.0
    tie_fraction: float = 0.0

    @property
    def mean_router_overlap(self) -> float:
        if not self.router_overlap:
            return 1.0
        return sum(self.router_overlap.values()) / len(self.router_overlap)

    @property
    def min_router_overlap(self) -> float:
        return min(self.router_overlap.values()) if self.router_overlap else 1.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "label": self.label,
            "mask_ratio": self.mask_ratio,
            "num_masked": self.num_masked,
            "logit_metrics": self.logit_metrics,
            "top1_agreement": self.top1_agreement,
            "unmask_agreement": self.unmask_agreement,
            "kl_masked": self.kl_masked,
            "router_overlap": self.router_overlap,
            "mean_router_overlap": self.mean_router_overlap,
            "reference_top2_margin": self.reference_top2_margin,
            "tie_fraction": self.tie_fraction,
        }


@dataclass
class TrajectoryReport:
    """Teacher-forced sweep over a denoising schedule."""

    states: list[StateReport] = field(default_factory=list)

    def series(self, key: str) -> list[float]:
        """Values of ``key`` in step order — the shape of the degradation."""
        out = []
        for s in self.states:
            if key in s.logit_metrics:
                out.append(s.logit_metrics[key])
            else:
                out.append(float(getattr(s, key)))
        return out

    @property
    def worst_state(self) -> Optional[StateReport]:
        if not self.states:
            return None
        return min(self.states, key=lambda s: s.top1_agreement)

    @property
    def min_router_overlap(self) -> float:
        return min((s.min_router_overlap for s in self.states), default=1.0)

    def to_dict(self) -> dict:
        return {
            "kind": "teacher_forced",
            "states": [s.to_dict() for s in self.states],
            "min_router_overlap": self.min_router_overlap,
            "min_top1_agreement": min((s.top1_agreement for s in self.states), default=1.0),
        }

    def to_table(self) -> str:
        header = (
            f"{'step':>4}  {'label':<14}{'masked':>8}{'top1':>9}{'unmask':>9}"
            f"{'KL':>10}{'cos':>9}{'router':>9}{'margin':>10}{'tied':>8}"
        )
        rows = [
            header,
            "-" * len(header),
        ]
        for s in self.states:
            rows.append(
                f"{s.step:>4}  {s.label:<14}{s.num_masked:>8}"
                f"{s.top1_agreement:>9.4f}{s.unmask_agreement:>9.4f}"
                f"{s.kl_masked:>10.2e}"
                f"{s.logit_metrics.get('cosine_similarity', float('nan')):>9.5f}"
                f"{s.mean_router_overlap:>9.4f}"
                f"{s.reference_top2_margin:>10.2e}{s.tie_fraction:>8.2f}"
            )
        rows.append("")
        rows.append("'tied' = share of masked positions where the reference's own top-2")
        rows.append("margin is below the quantization shift; disagreement there is noise.")
        return "\n".join(rows)


def compare_trajectory(
    reference: nn.Module,
    quantized: nn.Module,
    states: Sequence[DiffusionState],
    logits_fn: LogitsFn,
    router_fn: Optional[RouterFn] = None,
    *,
    unmask_k: int = 1,
    masked_positions_only: bool = True,
) -> TrajectoryReport:
    """Probe both models on identical denoising states.

    Args:
        reference: BF16 model.
        quantized: Same architecture, quantized weights.
        states: Denoising states, typically from
            :func:`~LLaDA_Quant.validation.diffusion.make_masked_states`.
        logits_fn: ``(model, state) -> [B, L, vocab]``.
        router_fn: ``(model, state) -> ids | {layer: ids} | None``. Top-k
            expert ids per token; compared slot-by-slot. If your model exposes
            these only through a hook, have ``router_fn`` read what the
            preceding ``logits_fn`` call cached.
        unmask_k: How many positions a decode step unmasks — used for the
            next-unmask overlap metric. Match your decoder's block size.
        masked_positions_only: Restrict logit metrics to masked slots. True is
            almost always what you want: predictions at resolved positions are
            discarded by the decoder.

    Returns:
        A :class:`TrajectoryReport`; read ``to_table()`` for a quick look and
        ``to_dict()`` to persist alongside a benchmark run.
    """
    report = TrajectoryReport()
    for state in states:
        with torch.no_grad():
            ref_logits = logits_fn(reference, state)
            ref_router = _as_router_mapping(router_fn(reference, state)) if router_fn else {}
            qnt_logits = logits_fn(quantized, state)
            qnt_router = _as_router_mapping(router_fn(quantized, state)) if router_fn else {}

        if ref_logits.shape != qnt_logits.shape:
            raise ValueError(
                f"logits shape mismatch at step {state.step}: "
                f"{tuple(ref_logits.shape)} vs {tuple(qnt_logits.shape)}"
            )

        mask = state.mask_positions.to(device=ref_logits.device, dtype=torch.bool)
        positions = mask if (masked_positions_only and bool(mask.any())) else None
        if positions is not None:
            ref_sel, qnt_sel = ref_logits[positions], qnt_logits[positions]
        else:
            ref_sel, qnt_sel = ref_logits, qnt_logits

        overlaps = {}
        for name, ref_ids in ref_router.items():
            if name not in qnt_router:
                raise ValueError(f"router_fn returned '{name}' for reference but not quantized")
            overlaps[name] = router_overlap(ref_ids, qnt_router[name])

        report.states.append(
            StateReport(
                step=state.step,
                label=state.describe(),
                mask_ratio=state.mask_ratio,
                num_masked=state.num_masked,
                logit_metrics=summarize_metrics(ref_sel, qnt_sel),
                top1_agreement=top1_agreement(ref_logits, qnt_logits, positions),
                unmask_agreement=unmask_selection_agreement(
                    ref_logits, qnt_logits, mask, k=unmask_k
                ),
                kl_masked=kl_divergence(ref_logits, qnt_logits, positions),
                router_overlap=overlaps,
                reference_top2_margin=top2_margin(ref_logits, positions),
                tie_fraction=tie_fraction(ref_logits, qnt_logits, positions),
            )
        )
    return report


@dataclass
class FreeRunStep:
    """Divergence between two independently advancing trajectories."""

    step: int
    num_masked_reference: int
    num_masked_quantized: int
    token_agreement: float
    resolved_set_agreement: float
    resolved_disagreements: int

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "num_masked_reference": self.num_masked_reference,
            "num_masked_quantized": self.num_masked_quantized,
            "token_agreement": self.token_agreement,
            "resolved_set_agreement": self.resolved_set_agreement,
            "resolved_disagreements": self.resolved_disagreements,
        }


@dataclass
class FreeRunReport:
    """Where two self-driven trajectories part company, and how far."""

    steps: list[FreeRunStep] = field(default_factory=list)
    first_divergence_step: Optional[int] = None
    final_token_agreement: float = 1.0
    reference_ids: Optional[torch.Tensor] = None
    quantized_ids: Optional[torch.Tensor] = None

    def to_dict(self) -> dict:
        return {
            "kind": "free_running",
            "steps": [s.to_dict() for s in self.steps],
            "first_divergence_step": self.first_divergence_step,
            "final_token_agreement": self.final_token_agreement,
        }

    def to_table(self) -> str:
        header = f"{'step':>4}{'masked(ref)':>13}{'masked(q)':>11}{'tokens=':>10}{'same unmask':>13}"
        rows = [header, "-" * len(header)]
        for s in self.steps:
            rows.append(
                f"{s.step:>4}{s.num_masked_reference:>13}{s.num_masked_quantized:>11}"
                f"{s.token_agreement:>10.4f}{s.resolved_set_agreement:>13.4f}"
            )
        return "\n".join(rows)


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum().item()
    if union == 0:
        return 1.0
    return (a & b).sum().item() / union


def compare_free_running(
    reference: nn.Module,
    quantized: nn.Module,
    initial_state: DiffusionState,
    logits_fn: LogitsFn,
    advance_fn: AdvanceFn,
    *,
    max_steps: int = 64,
) -> FreeRunReport:
    """Let each model denoise on its own and measure how far apart they drift.

    ``advance_fn(state, logits) -> next state | None`` is your unmasking rule
    (confidence remasking, block decoding, whatever the model repo does) —
    LLaDA_Quant supplies no decoding policy of its own. It is called once per
    model per step with that model's own logits. A model is frozen once it has
    no masked positions left or ``advance_fn`` returns None.

    ``token_agreement`` counts a generation-region position as agreeing when
    both trajectories hold the same id, mask token included — so it starts at
    1.0 and falls as the two decodes make different choices. Text equality is
    deliberately not the gate: read the curve, not the final string.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    gen_region = initial_state.mask_positions.bool()
    ref_state = initial_state
    qnt_state = initial_state
    ref_done = qnt_done = False
    report = FreeRunReport()

    for step in range(1, max_steps + 1):
        if ref_done and qnt_done:
            break

        with torch.no_grad():
            if not ref_done:
                next_ref = advance_fn(ref_state, logits_fn(reference, ref_state))
                ref_done = next_ref is None
                ref_state = next_ref or ref_state
            if not qnt_done:
                next_qnt = advance_fn(qnt_state, logits_fn(quantized, qnt_state))
                qnt_done = next_qnt is None
                qnt_state = next_qnt or qnt_state

        if ref_state.input_ids.shape != qnt_state.input_ids.shape:
            raise ValueError(
                f"advance_fn produced diverging shapes at step {step}: "
                f"{tuple(ref_state.input_ids.shape)} vs {tuple(qnt_state.input_ids.shape)}"
            )

        ref_ids, qnt_ids = ref_state.input_ids, qnt_state.input_ids
        agreement = (ref_ids[gen_region] == qnt_ids[gen_region]).float().mean().item()

        ref_resolved = gen_region & ~ref_state.mask_positions.bool()
        qnt_resolved = gen_region & ~qnt_state.mask_positions.bool()
        both = ref_resolved & qnt_resolved
        disagreements = int((ref_ids[both] != qnt_ids[both]).sum().item())

        report.steps.append(
            FreeRunStep(
                step=step,
                num_masked_reference=ref_state.num_masked,
                num_masked_quantized=qnt_state.num_masked,
                token_agreement=agreement,
                resolved_set_agreement=_jaccard(ref_resolved, qnt_resolved),
                resolved_disagreements=disagreements,
            )
        )
        if agreement < 1.0 and report.first_divergence_step is None:
            report.first_divergence_step = step

        ref_done = ref_done or ref_state.num_masked == 0
        qnt_done = qnt_done or qnt_state.num_masked == 0

    if report.steps:
        report.final_token_agreement = report.steps[-1].token_agreement
    report.reference_ids = ref_state.input_ids
    report.quantized_ids = qnt_state.input_ids
    return report
