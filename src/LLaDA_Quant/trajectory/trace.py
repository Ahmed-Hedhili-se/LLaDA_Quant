"""Compact, replayable record of one denoising trajectory.

A trace is what the GPU-side capture leaves behind so every metric after it
can be recomputed on a laptop. That forces two rules:

**Never store full tensors.** Logits are ``[B, L, 157184]`` for LLaDA; one
step of one model is ~40 MB. Anything needing the full vocabulary is reduced
to a scalar *online, on device*, during capture. Everything else is stored
top-k truncated.

**Never call an approximation exact.** Every stored scalar carries a
:class:`MetricPrecision`. ``kl_topk`` computed from 8 stored logprobs is not
``KL``; it is a lower bound on it, and the trace says so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

TRACE_FORMAT_VERSION = 1


class MetricPrecision(str, Enum):
    """How a stored scalar relates to the quantity it is named after."""

    EXACT = "exact"
    """Reduced on device over the full tensor. Equals the true value."""

    TOPK = "topk"
    """Computed from the stored top-k slice only. A truncation, not the truth."""

    SAMPLED = "sampled"
    """Estimated from a subset of positions or steps."""


@dataclass
class ScalarMetric:
    """One number plus an honest label for what it actually is."""

    value: float
    precision: str = MetricPrecision.EXACT.value
    note: str = ""

    @property
    def is_exact(self) -> bool:
        return self.precision == MetricPrecision.EXACT.value

    def to_dict(self) -> dict:
        return {"value": self.value, "precision": self.precision, "note": self.note}


@dataclass
class LayerStats:
    """Per-layer summaries small enough to keep for every step."""

    hidden_norm: float = 0.0
    hidden_absmax: float = 0.0
    router_margin: float = 0.0
    """Mean gap between the k-th and (k+1)-th router gate — how close routing
    came to flipping. A small margin means quantization noise decides."""
    router_gate_entropy: float = 0.0
    router_topk_ids: Optional[List[List[int]]] = None
    """``[tokens, top_k]``. Optional: this is the largest thing in a trace."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceStep:
    """One denoising step, compactly."""

    step: int
    num_masked: int
    mask_ratio: float
    masked_positions: List[int] = field(default_factory=list)
    committed_positions: List[int] = field(default_factory=list)
    committed_tokens: List[int] = field(default_factory=list)
    topk_ids: List[List[int]] = field(default_factory=list)
    """``[masked_positions, k]`` predicted token ids at masked slots."""
    topk_logprobs: List[List[float]] = field(default_factory=list)
    """Matching log-probabilities. Enough to recompute agreement and a
    truncated KL offline, not enough to recompute the exact one."""
    layers: Dict[str, LayerStats] = field(default_factory=dict)
    scalars: Dict[str, ScalarMetric] = field(default_factory=dict)
    """Online reductions over full tensors, computed during capture."""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "num_masked": self.num_masked,
            "mask_ratio": self.mask_ratio,
            "masked_positions": self.masked_positions,
            "committed_positions": self.committed_positions,
            "committed_tokens": self.committed_tokens,
            "topk_ids": self.topk_ids,
            "topk_logprobs": self.topk_logprobs,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "scalars": {k: v.to_dict() for k, v in self.scalars.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraceStep":
        return cls(
            step=data["step"],
            num_masked=data["num_masked"],
            mask_ratio=data["mask_ratio"],
            masked_positions=data.get("masked_positions", []),
            committed_positions=data.get("committed_positions", []),
            committed_tokens=data.get("committed_tokens", []),
            topk_ids=data.get("topk_ids", []),
            topk_logprobs=data.get("topk_logprobs", []),
            layers={k: LayerStats(**v) for k, v in data.get("layers", {}).items()},
            scalars={k: ScalarMetric(**v) for k, v in data.get("scalars", {}).items()},
        )


@dataclass
class Trace:
    """A whole trajectory: steps plus the context needed to interpret them."""

    label: str = ""
    model_id: str = ""
    mode: str = ""
    """``"shared"`` (teacher-forced) or ``"free_running"``."""
    seed: Optional[int] = None
    mask_token_id: Optional[int] = None
    prompt_length: int = 0
    gen_length: int = 0
    top_k_stored: int = 0
    steps: List[TraceStep] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    format_version: int = TRACE_FORMAT_VERSION

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def final_tokens(self) -> List[int]:
        """Committed tokens in commit order across the whole trajectory."""
        out: List[int] = []
        for step in self.steps:
            out.extend(step.committed_tokens)
        return out

    def scalar_series(self, key: str) -> List[float]:
        return [s.scalars[key].value for s in self.steps if key in s.scalars]

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "label": self.label,
            "model_id": self.model_id,
            "mode": self.mode,
            "seed": self.seed,
            "mask_token_id": self.mask_token_id,
            "prompt_length": self.prompt_length,
            "gen_length": self.gen_length,
            "top_k_stored": self.top_k_stored,
            "meta": self.meta,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trace":
        version = data.get("format_version", TRACE_FORMAT_VERSION)
        if version != TRACE_FORMAT_VERSION:
            raise ValueError(
                f"trace format version {version} != {TRACE_FORMAT_VERSION}; "
                "recapture or write a migration rather than guessing"
            )
        trace = cls(
            label=data.get("label", ""),
            model_id=data.get("model_id", ""),
            mode=data.get("mode", ""),
            seed=data.get("seed"),
            mask_token_id=data.get("mask_token_id"),
            prompt_length=data.get("prompt_length", 0),
            gen_length=data.get("gen_length", 0),
            top_k_stored=data.get("top_k_stored", 0),
            meta=data.get("meta", {}),
            format_version=version,
        )
        trace.steps = [TraceStep.from_dict(s) for s in data.get("steps", [])]
        return trace

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def size_estimate_bytes(self) -> int:
        """Rough serialized size — a guard against traces quietly exploding."""
        return len(json.dumps(self.to_dict()))
