"""Numerical benchmark: how much quantization changes the expert output.

Category B of the benchmark split. This is a **quantization-error** benchmark.
It is not, and must never be presented as, an INT8/INT4 speed benchmark: both
paths here run BF16 matmuls, because dequantize-then-matmul is the only
execution path that currently exists. Timing them against each other would
compare a computation with itself, which is exactly what the deleted
``bench_experts.py`` did.

Latency belongs in a Category C benchmark that does not exist yet, because
the kernel it would measure does not exist yet.

    python benchmarks/bench_numerical.py --num-experts 32 --hidden 1024 --intermediate 512
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from LLaDA_Quant.algorithms.symmetric import quantize_tensor, storage_bytes
from LLaDA_Quant.validation.metrics import summarize_metrics


def expert_forward(w1: torch.Tensor, w2: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """One dense pass over every expert (routing-free, for error only)."""
    intermediate = w2.shape[-1]
    out = torch.zeros(tokens.shape[0], w2.shape[1], dtype=torch.float32)
    for expert in range(w1.shape[0]):
        gate = tokens @ w1[expert, :intermediate].t().float()
        up = tokens @ w1[expert, intermediate:].t().float()
        out += (F.silu(gate) * up) @ w2[expert].t().float()
    return out / w1.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--heavy-tailed", action="store_true",
                        help="Student-t(3) weights instead of Gaussian, closer to real LLM tails")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    e, h, i = args.num_experts, args.hidden, args.intermediate
    if args.heavy_tailed:
        draw = lambda *shape: torch.distributions.StudentT(3.0).sample(shape) * 0.01
    else:
        draw = lambda *shape: torch.randn(*shape) * 0.02
    w1 = draw(e, 2 * i, h).to(torch.bfloat16)
    w2 = draw(e, h, i).to(torch.bfloat16)
    tokens = torch.randn(args.tokens, h, dtype=torch.float32)

    reference = expert_forward(w1, w2, tokens)
    bf16_bytes = (w1.numel() + w2.numel()) * 2

    rows = []
    for bits in (8, 4):
      for search in ("amax", "mse"):
        kw = dict(bits=bits, group_size=args.group_size, scale_search=search)
        q1 = quantize_tensor(w1, **kw)
        q2 = quantize_tensor(w2, **kw)
        out = expert_forward(q1.dequantize(torch.bfloat16), q2.dequantize(torch.bfloat16), tokens)
        weight_metrics = summarize_metrics(w1.float(), q1.dequantize(torch.float32))
        output_metrics = summarize_metrics(reference, out)
        measured = q1.storage_bytes() + q2.storage_bytes()
        w1f = w1.float()
        rows.append(
            {
                "bits": bits,
                "scale_search": search,
                "group_size": args.group_size,
                "packed": q1.packed,
                "weight_rel_l2": float(
                    (q1.dequantize(torch.float32) - w1f).norm() / w1f.norm()
                ),
                "weight_max_abs_error": weight_metrics["max_abs_error"],
                "weight_cosine": weight_metrics["cosine_similarity"],
                "output_rel_max_error": output_metrics["max_abs_error"]
                / float(reference.abs().max()),
                "output_cosine": output_metrics["cosine_similarity"],
                "storage_bytes_measured": measured,
                "storage_bytes_formula": storage_bytes(
                    w1.numel() + w2.numel(), bits, args.group_size
                ),
                "storage_ratio_vs_bf16": round(measured / bf16_bytes, 4),
            }
        )

    report = {
        "benchmark": "numerical",
        "measures": "quantization error of expert weights and expert output",
        "does_not_measure": (
            "latency; both paths execute BF16 matmuls because no packed kernel exists"
        ),
        "shape": {"num_experts": e, "hidden": h, "intermediate": i, "tokens": args.tokens},
        "bf16_weight_bytes": bf16_bytes,
        "rows": rows,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    header = (
        f"{'bits':>5}{'search':>8}{'packed':>8}{'w rel L2':>10}{'out cos':>10}"
        f"{'bytes':>12}{'vs BF16':>9}{'gain':>9}"
    )
    print(header)
    print("-" * len(header))
    baseline: dict[int, float] = {}
    for row in rows:
        if row["scale_search"] == "amax":
            baseline[row["bits"]] = row["weight_rel_l2"]
            gain = ""
        else:
            gain = f"{(1 - row['weight_rel_l2'] / baseline[row['bits']]) * 100:.1f}%"
        print(
            f"{row['bits']:>5}{row['scale_search']:>8}{str(row['packed']):>8}"
            f"{row['weight_rel_l2']:>10.4f}{row['output_cosine']:>10.5f}"
            f"{row['storage_bytes_measured']:>12}{row['storage_ratio_vs_bf16']:>9.3f}"
            f"{gain:>9}"
        )
    print("\nQuantization error only. No latency claim is made or implied:")
    print("every row executes the same BF16 matmul on dequantized weights.")
    print("'gain' is the weight-error reduction from MSE-optimal scale search,")
    print("which costs zero extra bytes -- the storage columns are identical.")


if __name__ == "__main__":
    main()
