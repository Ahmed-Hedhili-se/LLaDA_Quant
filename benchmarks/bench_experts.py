"""Benchmark: MoE expert weights memory + forward latency, BF16 vs INT8 (reference path).

Run with the GPU::

    python benchmarks/bench_experts.py --num-experts 64 --hidden 2048 --intermediate 1024

v0.1 measures the dequantize-then-matmul reference path (correct but slow);
v0.3 will replace it with the packed Triton kernel, same CLI, same report.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from LLaDA_Quant.algorithms.symmetric import quantize_tensor


def build_expert_weights(num_experts: int, hidden: int, intermediate: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    w1 = (torch.randn(num_experts, 2 * intermediate, hidden) * 0.02).to(dtype)
    w2 = (torch.randn(num_experts, hidden, intermediate) * 0.02).to(dtype)
    return w1, w2


def reference_forward(w1: torch.Tensor, w2: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    e, _, _ = w1.shape
    out = torch.zeros(tokens.shape[0], w2.shape[1], device=w1.device, dtype=w1.dtype)
    for expert in range(e):
        gate = tokens @ w1[expert, : w1.shape[1] // 2].t()
        up = tokens @ w1[expert, w1.shape[1] // 2 :].t()
        h = torch.nn.functional.silu(gate) * up
        out += h @ w2[expert].t()
    return out / e


def measure(fn, warmup: int, iters: int, *args) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = torch.bfloat16
    w1, w2 = build_expert_weights(args.num_experts, args.hidden, args.intermediate, dtype)
    q1 = quantize_tensor(w1, bits=8, group_size=128)
    q2 = quantize_tensor(w2, bits=8, group_size=128)
    w1_q = q1.dequantize(dtype=dtype)
    w2_q = q2.dequantize(dtype=dtype)

    tokens = torch.randn(args.tokens, args.hidden, dtype=dtype)
    t_bf16 = measure(reference_forward, args.warmup, args.iters, w1, w2, tokens)
    t_int8 = measure(reference_forward, args.warmup, args.iters, w1_q, w2_q, tokens)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    n_bytes_bf16 = (w1.numel() + w2.numel()) * 2
    n_bytes_int8 = (q1.q.numel() + q2.q.numel()) + (q1.scale.numel() + q2.scale.numel()) * 4
    report = {
        "expert_count": args.num_experts,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "latency_ms": {"bf16": round(t_bf16 * 1e3, 3), "int8_reference": round(t_int8 * 1e3, 3)},
        "weight_bytes": {"bf16": n_bytes_bf16, "int8": n_bytes_int8},
        "memory_saving_pct": round(100 * (1 - n_bytes_int8 / n_bytes_bf16), 1),
        "note": "reference (dequant+matmul) path; packed Triton kernel lands in v0.3",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()