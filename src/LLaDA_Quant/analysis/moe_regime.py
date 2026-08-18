"""Is the expert GEMM bandwidth-bound? The question that decides the kernel.

Weight-only quantization (W8A16 / W4A16) buys latency **only where the GEMM
is memory-bandwidth-bound**. Past the roofline crossover the bottleneck is
arithmetic, dequantize-then-matmul adds work, and the only remaining benefit
is capacity. Building a fused low-bit kernel before knowing which side of
that line the workload sits on is how months disappear.

For a top-k MoE the deciding quantity is **tokens per expert per step**, not
tokens per step: routing scatters M tokens over E experts, so each expert's
GEMM sees roughly ``M * top_k / E`` rows. For LLaDA-MoE that is ``M / 8``,
which is why batch size — not sequence length — dominates the verdict.

Two ways in:

* :func:`regime_sweep` — analytic, needs only the config. Gives the crossover
  and the ideal-balance operating point.
* :func:`expert_token_stats` — measured, takes real ``topk_ids`` captured
  from a running model. Ideal balance is optimistic; real routing is skewed,
  and the slowest expert sets the step time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class MoEShape:
    """Static MoE geometry."""

    num_experts: int
    top_k: int
    hidden: int
    intermediate: int
    num_layers: int = 1
    name: str = ""

    @property
    def expert_elements_per_layer(self) -> int:
        """w1 [E, 2I, H] + w2 [E, H, I] element count for one layer."""
        return self.num_experts * (
            2 * self.intermediate * self.hidden + self.hidden * self.intermediate
        )

    @property
    def expert_elements(self) -> int:
        return self.expert_elements_per_layer * self.num_layers


#: LLaDA-MoE-7B-A1B, from test_llada/src/model.py.
LLADA_MOE_7B_A1B = MoEShape(
    num_experts=64, top_k=8, hidden=2048, intermediate=1024, num_layers=16,
    name="LLaDA-MoE-7B-A1B",
)


@dataclass(frozen=True)
class Machine:
    """Peak throughput and bandwidth of the target accelerator."""

    name: str
    bf16_tflops: float
    int8_tops: float
    bandwidth_gbps: float
    memory_gb: float

    def peak_flops(self, compute_dtype: str) -> float:
        return (self.int8_tops if compute_dtype == "int8" else self.bf16_tflops) * 1e12

    def balance(self, compute_dtype: str) -> float:
        """Peak flops per byte of bandwidth — the roofline ridge point."""
        return self.peak_flops(compute_dtype) / (self.bandwidth_gbps * 1e9)


#: The GPU the inference repo's measurements were actually taken on
#: (INFERENCE_REPO_CHANGES.md: "single NVIDIA A40-24Q (sm_86, 84 SMs, 24 GB,
#: ~696 GB/s)"). A40-24Q is a 24 GB vGPU profile of a 48 GB A40. Its balance
#: of ~215 flops/byte matches the figure that handoff quotes independently.
A40_24Q = Machine("A40-24Q", bf16_tflops=149.7, int8_tops=299.3, bandwidth_gbps=696.0, memory_gb=24.0)

#: Full-card A40, same silicon and bandwidth, all 48 GB.
A40 = Machine("A40", bf16_tflops=149.7, int8_tops=299.3, bandwidth_gbps=696.0, memory_gb=48.0)

#: Present in an older benchmark trace's deviceProperties. Kept for
#: comparison; it is *not* the machine the measured numbers came from.
RTX_A6000 = Machine("RTX A6000", bf16_tflops=154.8, int8_tops=309.7, bandwidth_gbps=768.0, memory_gb=48.0)
A100_80GB = Machine("A100-SXM 80GB", bf16_tflops=312.0, int8_tops=624.0, bandwidth_gbps=2039.0, memory_gb=80.0)
H100_SXM = Machine("H100-SXM", bf16_tflops=989.0, int8_tops=1979.0, bandwidth_gbps=3350.0, memory_gb=80.0)


@dataclass(frozen=True)
class Scheme:
    """A weight/activation precision pair and what it costs to move a weight."""

    name: str
    weight_bytes: float
    compute_dtype: str

    @property
    def is_weight_only(self) -> bool:
        return self.compute_dtype == "bf16" and self.weight_bytes < 2


#: ``W4A16``/``W4A8`` assume the packed weights are expanded before the MMA,
#: which is what a Triton dequant-in-kernel path does. Ampere INT4 tensor
#: cores exist but no mainstream kernel stack emits them, so no scheme here
#: claims native 4-bit math.
SCHEMES: Dict[str, Scheme] = {
    "BF16": Scheme("BF16", 2.0, "bf16"),
    "W8A16": Scheme("W8A16", 1.0, "bf16"),
    "W4A16": Scheme("W4A16", 0.5, "bf16"),
    "W8A8": Scheme("W8A8", 1.0, "int8"),
    "W4A8": Scheme("W4A8", 0.5, "int8"),
}


@dataclass(frozen=True)
class Workload:
    """One decoding configuration.

    ``suffix_tokens`` is the sequence length the MoE actually sees per step.
    In LLaDA's cached decoder that is ``x[:, block_start:]`` — the *whole
    remaining suffix*, not just the active block — so it shrinks as blocks
    complete. Use :func:`suffix_lengths_for_schedule` to enumerate it.
    """

    batch: int
    suffix_tokens: int
    label: str = ""

    @property
    def tokens_per_step(self) -> int:
        return self.batch * self.suffix_tokens


def suffix_lengths_for_schedule(gen_length: int, block_length: int) -> List[int]:
    """Suffix length the model forwards at each block of a blockwise decode."""
    if gen_length % block_length:
        raise ValueError(f"block_length {block_length} must divide gen_length {gen_length}")
    return [gen_length - i * block_length for i in range(gen_length // block_length)]


def ideal_tokens_per_expert(shape: MoEShape, workload: Workload) -> float:
    """Rows per expert GEMM under perfectly balanced routing (optimistic)."""
    return workload.tokens_per_step * shape.top_k / shape.num_experts


@dataclass
class ExpertTokenStats:
    """Measured routing load, from real ``topk_ids``."""

    tokens: int
    top_k: int
    num_experts: int
    counts: List[int] = field(default_factory=list)

    @property
    def active_experts(self) -> int:
        return sum(1 for c in self.counts if c > 0)

    @property
    def mean(self) -> float:
        return sum(self.counts) / len(self.counts) if self.counts else 0.0

    @property
    def imbalance(self) -> float:
        """Max load over mean load. 1.0 is perfect balance; the max sets step time."""
        return max(self.counts) / self.mean if self.counts and self.mean else 1.0

    def percentile(self, p: float) -> float:
        if not self.counts:
            return 0.0
        ordered = sorted(self.counts)
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
        return float(ordered[idx])

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "top_k": self.top_k,
            "num_experts": self.num_experts,
            "active_experts": self.active_experts,
            "min": min(self.counts) if self.counts else 0,
            "p50": self.percentile(50),
            "mean": round(self.mean, 2),
            "p90": self.percentile(90),
            "p99": self.percentile(99),
            "max": max(self.counts) if self.counts else 0,
            "imbalance_max_over_mean": round(self.imbalance, 3),
        }


def expert_token_stats(topk_ids: torch.Tensor, num_experts: int) -> ExpertTokenStats:
    """Per-expert row counts from real routing.

    ``topk_ids`` is ``[tokens, top_k]`` as handed to ``fused_moe``. Call this
    from the inference repo with captured routing to replace the ideal-balance
    estimate with the real distribution — the max, not the mean, is what the
    step waits on.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be [tokens, top_k], got {tuple(topk_ids.shape)}")
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    return ExpertTokenStats(
        tokens=topk_ids.shape[0],
        top_k=topk_ids.shape[1],
        num_experts=num_experts,
        counts=counts.tolist(),
    )


@dataclass
class GemmRegime:
    """Roofline verdict for one expert GEMM under one scheme."""

    scheme: str
    m_per_expert: float
    gemm_w1: tuple
    gemm_w2: tuple
    arithmetic_intensity: float
    machine_balance: float

    @property
    def is_memory_bound(self) -> bool:
        return self.arithmetic_intensity < self.machine_balance

    @property
    def bound(self) -> str:
        return "memory" if self.is_memory_bound else "compute"

    @property
    def crossover_m(self) -> float:
        """Rows per expert at which this scheme stops being bandwidth-bound.

        From ``AI = 2M / weight_bytes``, the ridge point ``AI == balance``
        gives ``M = balance * weight_bytes / 2``.
        """
        if not self.arithmetic_intensity or not self.m_per_expert:
            return float("inf")
        weight_bytes = 2 * self.m_per_expert / self.arithmetic_intensity
        return self.machine_balance * weight_bytes / 2

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "m_per_expert": round(self.m_per_expert, 2),
            "gemm_w1_MNK": list(self.gemm_w1),
            "gemm_w2_MNK": list(self.gemm_w2),
            "arithmetic_intensity_flops_per_byte": round(self.arithmetic_intensity, 2),
            "machine_balance_flops_per_byte": round(self.machine_balance, 1),
            "bound": self.bound,
            "crossover_m_per_expert": round(self.crossover_m, 1),
        }


def gemm_regime(
    shape: MoEShape,
    m_per_expert: float,
    scheme: Scheme,
    machine: Machine = A40_24Q,
) -> GemmRegime:
    """Roofline classification of one expert's GEMM pair.

    Arithmetic intensity for a weight-stationary GEMM with few rows is
    ``2*M*N*K / (N*K*weight_bytes) = 2*M / weight_bytes`` — activations are
    negligible while M is small, which is exactly the regime in question.
    """
    intensity = 2 * m_per_expert / scheme.weight_bytes
    return GemmRegime(
        scheme=scheme.name,
        m_per_expert=m_per_expert,
        gemm_w1=(round(m_per_expert), 2 * shape.intermediate, shape.hidden),
        gemm_w2=(round(m_per_expert), shape.hidden, shape.intermediate),
        arithmetic_intensity=intensity,
        machine_balance=machine.balance(scheme.compute_dtype),
    )


def crossover_m(scheme: Scheme, machine: Machine = A40_24Q) -> float:
    """Rows per expert where ``scheme`` stops being bandwidth-bound."""
    return machine.balance(scheme.compute_dtype) * scheme.weight_bytes / 2


@dataclass
class RegimeRow:
    workload: Workload
    m_per_expert: float
    regimes: Dict[str, GemmRegime]

    def to_dict(self) -> dict:
        return {
            "label": self.workload.label,
            "batch": self.workload.batch,
            "suffix_tokens": self.workload.suffix_tokens,
            "tokens_per_step": self.workload.tokens_per_step,
            "m_per_expert": round(self.m_per_expert, 2),
            "regimes": {k: v.to_dict() for k, v in self.regimes.items()},
        }


@dataclass
class RegimeReport:
    shape: MoEShape
    machine: Machine
    rows: List[RegimeRow] = field(default_factory=list)
    measured: Optional[ExpertTokenStats] = None

    def to_dict(self) -> dict:
        return {
            "model": self.shape.name,
            "moe": {
                "num_experts": self.shape.num_experts,
                "top_k": self.shape.top_k,
                "hidden": self.shape.hidden,
                "intermediate": self.shape.intermediate,
                "num_layers": self.shape.num_layers,
            },
            "machine": {
                "name": self.machine.name,
                "bf16_tflops": self.machine.bf16_tflops,
                "int8_tops": self.machine.int8_tops,
                "bandwidth_gbps": self.machine.bandwidth_gbps,
                "memory_gb": self.machine.memory_gb,
                "balance_bf16": round(self.machine.balance("bf16"), 1),
                "balance_int8": round(self.machine.balance("int8"), 1),
            },
            "crossovers_m_per_expert": {
                name: round(crossover_m(s, self.machine), 1) for name, s in SCHEMES.items()
            },
            "rows": [r.to_dict() for r in self.rows],
            "measured_routing": self.measured.to_dict() if self.measured else None,
            "routing_note": (
                "rows assume perfectly balanced routing (M*top_k/E), which is "
                "optimistic; measured_routing is None unless real topk_ids were supplied"
            ),
        }

    def to_table(self, schemes: Sequence[str] = ("BF16", "W8A16", "W4A16")) -> str:
        head = f"{'workload':<26}{'tokens/step':>12}{'M/expert':>10}  " + "".join(
            f"{s:>10}" for s in schemes
        )
        lines = [head, "-" * len(head)]
        for row in self.rows:
            cells = "".join(f"{row.regimes[s].bound:>10}" for s in schemes)
            lines.append(
                f"{row.workload.label:<26}{row.workload.tokens_per_step:>12}"
                f"{row.m_per_expert:>10.1f}  {cells}"
            )
        lines.append("")
        lines.append(f"machine: {self.machine.name}  "
                     f"balance bf16={self.machine.balance('bf16'):.0f} "
                     f"int8={self.machine.balance('int8'):.0f} flops/byte")
        lines.append("crossover M/expert (bandwidth-bound below this): " + ", ".join(
            f"{s}={crossover_m(SCHEMES[s], self.machine):.0f}" for s in schemes
        ))
        return "\n".join(lines)


def regime_sweep(
    shape: MoEShape = LLADA_MOE_7B_A1B,
    workloads: Optional[Sequence[Workload]] = None,
    machine: Machine = A40_24Q,
    measured: Optional[ExpertTokenStats] = None,
) -> RegimeReport:
    """Classify every workload against the roofline for every scheme."""
    if workloads is None:
        workloads = default_workloads()
    rows = []
    for workload in workloads:
        m = ideal_tokens_per_expert(shape, workload)
        rows.append(
            RegimeRow(
                workload=workload,
                m_per_expert=m,
                regimes={
                    name: gemm_regime(shape, m, scheme, machine)
                    for name, scheme in SCHEMES.items()
                },
            )
        )
    return RegimeReport(shape=shape, machine=machine, rows=rows, measured=measured)


def default_workloads(gen_length: int = 128, block_length: int = 32) -> List[Workload]:
    """Latency-oriented and throughput-oriented points from the real decoder.

    Batch sizes mirror what the inference repo actually benchmarks (its saved
    traces are batch 31-57), plus batch 1 for the latency case.
    """
    suffixes = suffix_lengths_for_schedule(gen_length, block_length)
    first, last = suffixes[0], suffixes[-1]
    out: List[Workload] = []
    for batch in (1, 4, 16, 32, 57):
        out.append(Workload(batch, first, f"batch={batch}, first block (L={first})"))
        out.append(Workload(batch, last, f"batch={batch}, last block (L={last})"))
    return out


def kernel_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    *,
    weight_bytes: float = 2.0,
    activation_bytes: int = 2,
    b_tiles: int = 2,
) -> int:
    """Shared memory one fused-MoE program stages, generalised over precision.

    Mirrors ``fused_moe_triton._shmem_bytes`` in the inference repo, which
    hardcodes 2 bytes per element for *both* operands::

        (bm * bk + b_tiles * bk * bn) * num_stages * 2

    That is correct only for BF16 weights. A low-bit kernel stages 1 byte per
    INT8 weight or 0.5 per packed INT4, so the same tile needs materially less
    shared memory — and using the BF16 formula would over-estimate the budget
    and silently reject configurations that would have fit.

    ``b_tiles=2`` is the default because the SiLU epilogue has each program
    hold the gate tile at ``offs_bn`` and the up tile at ``offs_bn + N``
    simultaneously. GEMM2 has no epilogue partner, so pass ``b_tiles=1``.
    """
    if min(block_m, block_n, block_k, num_stages, b_tiles) < 1:
        raise ValueError("tile dimensions, stages and b_tiles must all be >= 1")
    a_bytes = block_m * block_k * activation_bytes
    weight_tile_bytes = b_tiles * block_k * block_n * weight_bytes
    return int((a_bytes + weight_tile_bytes) * num_stages)


def shared_memory_headroom(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    scheme: Scheme,
    *,
    limit_bytes: int = 101_376 - 1024,
    b_tiles: int = 2,
) -> Dict[str, float]:
    """How a tile's shared-memory cost changes when the weights get smaller.

    ``limit_bytes`` defaults to the sm_86 opt-in limit the inference repo uses
    (``sharedMemPerBlockOptin`` 101,376 minus its 1 KB margin).
    """
    bf16 = kernel_shared_memory_bytes(
        block_m, block_n, block_k, num_stages, weight_bytes=2.0, b_tiles=b_tiles
    )
    low_bit = kernel_shared_memory_bytes(
        block_m, block_n, block_k, num_stages, weight_bytes=scheme.weight_bytes, b_tiles=b_tiles
    )
    return {
        "scheme": scheme.name,
        "bf16_bytes": bf16,
        "scheme_bytes": low_bit,
        "limit_bytes": limit_bytes,
        "bf16_fits": bf16 <= limit_bytes,
        "scheme_fits": low_bit <= limit_bytes,
        "freed_bytes": bf16 - low_bit,
    }
