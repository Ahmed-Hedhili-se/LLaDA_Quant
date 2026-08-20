"""Recompute trajectory metrics from stored traces, without the model.

Everything here runs on a laptop from JSON. That is the point: capture is
expensive and needs the GPU, so it should happen once, and every later
question should be answerable offline.

Two classes of number come out:

* **Replayed** — derived from the stored top-k slice. Honest but truncated;
  each carries ``MetricPrecision.TOPK``.
* **Stored exact** — reduced over full tensors during capture and carried in
  the trace. Read these when you need the real value.

:func:`verify_replay` cross-checks the two so a silent drift between what
capture measured and what replay reconstructs cannot go unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import commit_order_agreement
from .trace import MetricPrecision, ScalarMetric, Trace


def _topk(value: float, note: str) -> ScalarMetric:
    return ScalarMetric(value=float(value), precision=MetricPrecision.TOPK.value, note=note)


@dataclass
class ReplayedStep:
    step: int
    replayed: Dict[str, ScalarMetric] = field(default_factory=dict)
    stored_exact: Dict[str, ScalarMetric] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "replayed": {k: v.to_dict() for k, v in self.replayed.items()},
            "stored_exact": {k: v.to_dict() for k, v in self.stored_exact.items()},
        }


@dataclass
class ReplayReport:
    mode: str
    steps: List[ReplayedStep] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)

    def series(self, key: str) -> List[float]:
        out = []
        for step in self.steps:
            metric = step.replayed.get(key) or step.stored_exact.get(key)
            if metric is not None:
                out.append(metric.value)
        return out

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
        }


def _jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def replay_shared(reference: Trace, quantized: Trace) -> ReplayReport:
    """Mode A metrics, recomputed offline from two shared-state traces."""
    if reference.mode != "shared" or quantized.mode != "shared":
        raise ValueError(
            f"replay_shared needs mode='shared' traces, got "
            f"{reference.mode!r} and {quantized.mode!r}"
        )
    if len(reference) != len(quantized):
        raise ValueError(f"step count mismatch: {len(reference)} vs {len(quantized)}")

    report = ReplayReport(mode="shared")
    for ref_step, qnt_step in zip(reference.steps, quantized.steps):
        entry = ReplayedStep(step=ref_step.step)
        if ref_step.topk_ids and qnt_step.topk_ids:
            pairs = list(zip(ref_step.topk_ids, qnt_step.topk_ids))
            overlap = sum(_jaccard(a, b) for a, b in pairs) / len(pairs)
            # Use the stored argmax, never topk[0]: they disagree on exactly
            # tied logits, and the decoder uses argmax.
            if ref_step.argmax_ids and qnt_step.argmax_ids:
                argmax_pairs = list(zip(ref_step.argmax_ids, qnt_step.argmax_ids))
                top1 = sum(1 for a, b in argmax_pairs if a == b) / len(argmax_pairs)
                note = "stored argmax per position; equals exact top-1 agreement"
            else:
                top1 = sum(1 for a, b in pairs if a and b and a[0] == b[0]) / len(pairs)
                note = "argmax of the stored top-k; differs on exact ties"
            entry.replayed["top1_agreement"] = _topk(top1, note)
            entry.replayed["topk_set_overlap"] = _topk(
                overlap, f"Jaccard over the stored top-{reference.top_k_stored} ids"
            )
        entry.stored_exact = {
            key: metric
            for key, metric in qnt_step.scalars.items()
            if metric.is_exact and key.startswith("pair.")
        }
        report.steps.append(entry)

    exact_top1 = [
        s.stored_exact["pair.top1_agreement"].value
        for s in report.steps
        if "pair.top1_agreement" in s.stored_exact
    ]
    ties = [
        s.stored_exact["pair.tie_fraction"].value
        for s in report.steps
        if "pair.tie_fraction" in s.stored_exact
    ]
    report.summary = {
        "steps": len(report.steps),
        "min_top1_agreement": min(exact_top1) if exact_top1 else 1.0,
        "mean_top1_agreement": sum(exact_top1) / len(exact_top1) if exact_top1 else 1.0,
        "mean_tie_fraction": sum(ties) / len(ties) if ties else 0.0,
    }
    return report


def replay_free_running(reference: Trace, quantized: Trace) -> ReplayReport:
    """Mode B metrics: what the two independent trajectories actually committed."""
    for trace in (reference, quantized):
        if trace.mode != "free_running":
            raise ValueError(f"replay_free_running needs mode='free_running', got {trace.mode!r}")

    ref_commits = [s.committed_positions for s in reference.steps]
    qnt_commits = [s.committed_positions for s in quantized.steps]

    ref_final: Dict[int, int] = {}
    qnt_final: Dict[int, int] = {}
    report = ReplayReport(mode="free_running")
    first_divergence: Optional[int] = None

    for index in range(max(len(reference.steps), len(quantized.steps))):
        entry = ReplayedStep(step=index + 1)
        if index < len(reference.steps):
            step = reference.steps[index]
            ref_final.update(dict(zip(step.committed_positions, step.committed_tokens)))
        if index < len(quantized.steps):
            step = quantized.steps[index]
            qnt_final.update(dict(zip(step.committed_positions, step.committed_tokens)))

        shared = set(ref_final) & set(qnt_final)
        disagreements = sum(1 for p in shared if ref_final[p] != qnt_final[p])
        agreement = 1.0 - disagreements / len(shared) if shared else 1.0
        same_positions = (
            index < len(ref_commits)
            and index < len(qnt_commits)
            and set(ref_commits[index]) == set(qnt_commits[index])
        )
        entry.replayed["token_agreement_on_committed"] = _topk(
            agreement, "over positions committed by both trajectories so far"
        )
        entry.replayed["same_commit_set"] = _topk(
            float(same_positions), "did both commit the same positions this step"
        )
        entry.replayed["resolved_disagreements"] = _topk(float(disagreements), "cumulative count")
        if agreement < 1.0 and first_divergence is None:
            first_divergence = index + 1
        report.steps.append(entry)

    shared = set(ref_final) & set(qnt_final)
    final_agreement = (
        1.0 - sum(1 for p in shared if ref_final[p] != qnt_final[p]) / len(shared)
        if shared
        else 1.0
    )
    report.summary = {
        "steps": len(report.steps),
        "first_divergence_step": float(first_divergence) if first_divergence else -1.0,
        "final_token_agreement": final_agreement,
        "commit_order_agreement": commit_order_agreement(ref_commits, qnt_commits),
        "positions_committed_by_both": float(len(shared)),
    }
    return report


def verify_replay(report: ReplayReport, tolerance: float = 1e-6) -> List[str]:
    """Check replayed numbers against the exact ones captured on device.

    Returns a list of human-readable discrepancies; empty means the offline
    path reproduces what the GPU measured. Run this in CI: it is what stops
    the trace format and the capture code drifting apart unnoticed.
    """
    problems: List[str] = []
    for step in report.steps:
        replayed = step.replayed.get("top1_agreement")
        exact = step.stored_exact.get("pair.top1_agreement")
        if replayed is None or exact is None:
            continue
        if abs(replayed.value - exact.value) > tolerance:
            problems.append(
                f"step {step.step}: replayed top1_agreement={replayed.value:.6f} "
                f"but capture measured {exact.value:.6f}"
            )
    return problems
