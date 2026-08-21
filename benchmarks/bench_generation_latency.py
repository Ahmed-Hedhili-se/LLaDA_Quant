"""What PACKED INT4 costs in wall-clock time, against BF16, on the real model.

**This measures the dequantize-then-matmul path, which is the only quantized
execution that exists.** It is not a fused-kernel benchmark and must never be
quoted as one: no kernel consumes packed weights directly yet, so this is the
*cost* of quantization today, not a speedup.

Expect INT4 to be SLOWER. PACKED mode reconstructs the BF16 expert weights on
every access, so each expert GEMM reads int4, writes bf16, then reads it back --
roughly 2.5x the memory traffic of plain BF16. The win quantization currently
delivers is capacity (2.8x smaller model), and this benchmark is what that
capacity costs.

Two things are timed:

  forward   a single dense forward at several sequence lengths -- the unit the
            MoE kernel actually sees, and where the dequantize tax lands
  generate  a full denoising loop through the production commit rule, i.e. the
            tokens/sec a user would observe

    python benchmarks/bench_generation_latency.py \\
        --repo ~/test_llada --weight-dir ~/test_llada/weights \\
        --build-device cuda:0 --gen-length 128 --steps 128
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from LLaDA_Quant import compare_resident_memory, resident_memory
from LLaDA_Quant.trajectory import (
    LLADA_MASK_ID,
    fully_masked_state,
    load_llada_decoder,
    make_llada_advance_fn,
)

from bench_bf16_vs_int4 import (  # noqa: E402  (same directory)
    build_bf16_model,
    encode_prompt,
    int4_config,
    quantize_experts_streaming,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_forward(model, input_ids, warmup: int, iters: int) -> dict:
    """Median and spread of a single dense forward."""
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
        _sync()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            model(input_ids)
            _sync()
            samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": round(statistics.median(samples), 2),
        "min_ms": round(min(samples), 2),
        "max_ms": round(max(samples), 2),
    }


def time_generation(model, start_state, logits_fn, advance, max_steps: int) -> dict:
    """A full denoising loop through the production commit rule."""
    _sync()
    began = time.perf_counter()
    state, steps = start_state, 0
    with torch.no_grad():
        for _ in range(max_steps):
            nxt = advance(state, logits_fn(model, state))
            if nxt is None:
                break
            state, steps = nxt, steps + 1
            if state.num_masked == 0:
                break
    _sync()
    elapsed = time.perf_counter() - began
    committed = int(start_state.mask_positions.sum()) - state.num_masked
    return {
        "seconds": round(elapsed, 3),
        "steps": steps,
        "tokens": committed,
        "tokens_per_second": round(committed / elapsed, 2) if elapsed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-search", choices=("amax", "mse"), default="mse")
    parser.add_argument("--search-grid", type=int, default=8)
    parser.add_argument("--build-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("building BF16 ...", flush=True)
    bf16 = build_bf16_model(args.repo, args.weight_dir, args.build_device).to(args.device).eval()

    print("building + quantizing INT4 ...", flush=True)
    config = int4_config(args.group_size, args.scale_search, args.search_grid)
    int4 = build_bf16_model(args.repo, args.weight_dir, args.build_device)
    quantize_experts_streaming(int4, config, args.device, verbose=False)
    int4 = int4.to(args.device).eval()

    memory = compare_resident_memory(bf16, int4, label="INT4 PACKED")
    print("  " + memory.describe())

    models = {"BF16": bf16, "INT4-PACKED": int4}
    report: dict = {
        "benchmark": "generation_latency",
        "measures": "wall clock of dequantize-then-matmul INT4 vs BF16",
        "does_not_measure": (
            "a fused low-bit kernel; none exists. INT4 is expected to be SLOWER "
            "because PACKED mode reconstructs BF16 weights on every access"
        ),
        "memory": memory.to_dict(),
        "forward": {},
        "generation": {},
    }

    # --- forward latency --------------------------------------------------
    print(f"\n{'seq':>6}" + "".join(f"{n:>16}" for n in models) + f"{'INT4 / BF16':>14}")
    print("-" * (6 + 16 * len(models) + 14))
    for length in args.seq_lengths:
        ids = torch.randint(0, 100, (1, length), device=args.device)
        row = {name: time_forward(m, ids, args.warmup, args.iters) for name, m in models.items()}
        ratio = row["INT4-PACKED"]["median_ms"] / max(1e-9, row["BF16"]["median_ms"])
        report["forward"][length] = {**row, "int4_over_bf16": round(ratio, 3)}
        cells = "".join(f"{row[n]['median_ms']:>13.2f} ms" for n in models)
        print(f"{length:>6}{cells}{ratio:>13.2f}x")

    # --- full generation --------------------------------------------------
    prompt_ids = encode_prompt(args.repo, args.weight_dir, args.prompt, args.chat_template)
    decoder = load_llada_decoder(args.repo)
    advance = make_llada_advance_fn(decoder, steps=args.steps, temperature=0.0)
    start = fully_masked_state(prompt_ids, args.gen_length, LLADA_MASK_ID)

    def logits_fn(model, state):
        out = model(state.input_ids.to(args.device))
        return (out[0] if isinstance(out, tuple) else out).float()

    print(f"\n{'model':>14}{'seconds':>10}{'tokens':>9}{'tok/s':>10}")
    print("-" * 43)
    for name, model in models.items():
        result = time_generation(model, start, logits_fn, advance, args.steps)
        report["generation"][name] = result
        print(f"{name:>14}{result['seconds']:>10.2f}{result['tokens']:>9}"
              f"{result['tokens_per_second']:>10.2f}")

    bf16_tps = report["generation"]["BF16"]["tokens_per_second"]
    int4_tps = report["generation"]["INT4-PACKED"]["tokens_per_second"]
    if bf16_tps:
        change = int4_tps / bf16_tps
        report["generation"]["int4_over_bf16_throughput"] = round(change, 3)
        verdict = "FASTER" if change > 1 else "SLOWER"
        print(f"\nINT4 throughput is {change:.2f}x BF16 -- {verdict}.")

    print("\nThis is dequantize-then-matmul, the only quantized execution that")
    print("exists today. A slowdown here is the expected, honest result: the")
    print("current win is capacity ("
          f"{memory.ratio:.3f}x resident), not speed. Latency would come from a")
    print("fused kernel reading packed weights directly, which is not built.")

    if args.json:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
