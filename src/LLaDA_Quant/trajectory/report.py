"""Mode A, Mode B, and the noise floor — reported together or not at all.

Three numbers are routinely confused, and each confusion produces a
confident wrong conclusion:

* **Mode A** (shared state) measures error *injected per step*. Reading it as
  end-to-end damage understates amplification.
* **Mode B** (free running) measures *amplified* divergence. Reading it as
  per-step error overstates the damage, because one early coin-toss flip
  drags every later step with it.
* **The noise floor** (BF16 vs BF16) is what both modes report when nothing
  is wrong. Non-determinism, batch composition and kernel selection all move
  it off zero. A quantization result that is not compared against it is not a
  result.

:class:`TrajectoryReport` refuses to render a table without space for all
three, which is the cheapest available guard against quoting one as another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .replay import ReplayReport


def _get(report: Optional[ReplayReport], key: str, default: float = float("nan")) -> float:
    if report is None:
        return default
    return report.summary.get(key, default)


@dataclass
class TrajectoryReport:
    """Everything needed to state what quantization did to generation."""

    mode_a: Optional[ReplayReport] = None
    mode_b: Optional[ReplayReport] = None
    noise_floor_a: Optional[ReplayReport] = None
    noise_floor_b: Optional[ReplayReport] = None
    label: str = ""

    @property
    def per_step_signal(self) -> float:
        """Mode A disagreement in excess of the BF16-vs-BF16 floor."""
        quantized = 1.0 - _get(self.mode_a, "mean_top1_agreement", 1.0)
        floor = 1.0 - _get(self.noise_floor_a, "mean_top1_agreement", 1.0)
        return quantized - (0.0 if floor != floor else floor)

    @property
    def amplification(self) -> float:
        """How much larger end-to-end divergence is than per-step divergence.

        Above 1 means errors compound along the schedule; near 1 means each
        step's error stays local. Undefined (nan) without both modes.
        """
        per_step = 1.0 - _get(self.mode_a, "mean_top1_agreement", float("nan"))
        end_to_end = 1.0 - _get(self.mode_b, "final_token_agreement", float("nan"))
        if per_step != per_step or end_to_end != end_to_end or per_step <= 0:
            return float("nan")
        return end_to_end / per_step

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "mode_a_shared_state": self.mode_a.to_dict() if self.mode_a else None,
            "mode_b_free_running": self.mode_b.to_dict() if self.mode_b else None,
            "noise_floor_a": self.noise_floor_a.summary if self.noise_floor_a else None,
            "noise_floor_b": self.noise_floor_b.summary if self.noise_floor_b else None,
            "derived": {
                "per_step_signal_above_floor": self.per_step_signal,
                "amplification_ratio": self.amplification,
            },
            "interpretation": (
                "Mode A = error injected per step (no amplification). "
                "Mode B = amplified end-to-end divergence (cannot isolate per-step). "
                "Both are meaningless without the BF16-vs-BF16 floor."
            ),
        }

    def to_table(self) -> str:
        rows = [
            f"{'quantity':<34}{'quantized':>14}{'BF16 floor':>14}",
            "-" * 62,
            f"{'Mode A  mean top-1 agreement':<34}"
            f"{_get(self.mode_a, 'mean_top1_agreement'):>14.4f}"
            f"{_get(self.noise_floor_a, 'mean_top1_agreement'):>14.4f}",
            f"{'Mode A  min top-1 agreement':<34}"
            f"{_get(self.mode_a, 'min_top1_agreement'):>14.4f}"
            f"{_get(self.noise_floor_a, 'min_top1_agreement'):>14.4f}",
            f"{'Mode A  mean tie fraction':<34}"
            f"{_get(self.mode_a, 'mean_tie_fraction'):>14.4f}"
            f"{_get(self.noise_floor_a, 'mean_tie_fraction'):>14.4f}",
            f"{'Mode B  final token agreement':<34}"
            f"{_get(self.mode_b, 'final_token_agreement'):>14.4f}"
            f"{_get(self.noise_floor_b, 'final_token_agreement'):>14.4f}",
            f"{'Mode B  first divergence step':<34}"
            f"{_get(self.mode_b, 'first_divergence_step'):>14.0f}"
            f"{_get(self.noise_floor_b, 'first_divergence_step'):>14.0f}",
            f"{'Mode B  commit-order agreement':<34}"
            f"{_get(self.mode_b, 'commit_order_agreement'):>14.4f}"
            f"{_get(self.noise_floor_b, 'commit_order_agreement'):>14.4f}",
            "",
            f"per-step signal above floor : {self.per_step_signal:.4f}",
            f"amplification (B/A)         : {self.amplification:.2f}x",
            "",
            "Mode A is per-step injected error; Mode B is amplified end-to-end",
            "divergence. Neither substitutes for the other, and neither means",
            "anything without the BF16-vs-BF16 floor column.",
        ]
        return "\n".join(rows)
