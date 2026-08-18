"""Decision benchmark: is the expert GEMM bandwidth-bound enough to matter?

Run this **before** writing a fused low-bit MoE kernel. It answers the only
question that determines whether that kernel pays for itself: how many rows
does each expert's GEMM actually see, and does that put it below or above the
roofline crossover.

Analytic by default (needs nothing but the config). Pass ``--routing-file``
with a saved ``topk_ids`` tensor captured from a real run to replace the
ideal-balance assumption with the measured distribution — the slowest expert,
not the average one, sets the step time.

    python benchmarks/bench_moe_regime.py
    python benchmarks/bench_moe_regime.py --gen-length 256 --block-length 64
    python benchmarks/bench_moe_regime.py --routing-file topk_ids.pt
"""

from __future__ import annotations

import argparse
import json

import torch

from LLaDA_Quant.analysis import (
    A40,
    A40_24Q,
    A100_80GB,
    H100_SXM,
    LLADA_MOE_7B_A1B,
    RTX_A6000,
    SCHEMES,
    MoEShape,
    Workload,
    crossover_m,
    expert_token_stats,
    regime_sweep,
    suffix_lengths_for_schedule,
)

#: ``a40-24q`` is the default because it is the GPU the inference repo
#: actually measured on (see INFERENCE_REPO_CHANGES.md).
MACHINES = {
    "a40-24q": A40_24Q,
    "a40": A40,
    "a6000": RTX_A6000,
    "a100": A100_80GB,
    "h100": H100_SXM,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32, 57])
    parser.add_argument("--routing-file", type=str, default=None,
                        help="torch .pt file holding a [tokens, top_k] topk_ids tensor")
    parser.add_argument("--machine", choices=sorted(MACHINES), default="a40-24q",
                        help="target accelerator (default: the GPU the measurements came from)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    shape: MoEShape = LLADA_MOE_7B_A1B
    machine = MACHINES[args.machine]
    suffixes = suffix_lengths_for_schedule(args.gen_length, args.block_length)
    workloads = []
    for batch in args.batches:
        for suffix in (suffixes[0], suffixes[-1]):
            position = "first" if suffix == suffixes[0] else "last"
            workloads.append(Workload(batch, suffix, f"batch={batch}, {position} block (L={suffix})"))

    measured = None
    if args.routing_file:
        topk_ids = torch.load(args.routing_file, map_location="cpu")
        measured = expert_token_stats(topk_ids, shape.num_experts)

    report = regime_sweep(shape=shape, workloads=workloads, machine=machine, measured=measured)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print(report.to_table(("BF16", "W8A16", "W4A16", "W8A8")))
    print()
    for bits, label in ((2, "BF16"), (1, "INT8"), (0.5, "INT4")):
        gb = shape.expert_elements * bits / 1e9
        print(f"  expert weights {label:5s}: {gb:6.2f} GB  "
              f"({machine.memory_gb - gb:5.2f} GB of {machine.memory_gb:.0f} GB left for "
              f"activations, KV cache and batch)")
    if measured is not None:
        print("\nmeasured routing:")
        for key, value in measured.to_dict().items():
            print(f"  {key:>26}: {value}")
        print(f"  {'note':>26}: the max, not the mean, bounds the step")
    else:
        print("\nRows assume perfect routing balance. Pass --routing-file with real")
        print("topk_ids to replace that with the measured distribution.")

    print("\nHow to read this:")
    print("  memory-bound  -> weight-only quantization (W8A16/W4A16) buys latency,")
    print("                   ceiling ~= the weight-byte ratio (2x for INT8, 4x for INT4)")
    print("  compute-bound -> weight-only quantization buys NO latency; the only")
    print("                   benefit is capacity, which raises throughput by")
    print("                   allowing a larger batch")
    print("  crossover M/expert: " + ", ".join(
        f"{n}={crossover_m(s, machine):.0f}" for n, s in SCHEMES.items()
    ))


if __name__ == "__main__":
    main()
