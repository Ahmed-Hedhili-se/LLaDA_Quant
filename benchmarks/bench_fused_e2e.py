"""End-to-end generation with the fused W8A16 MoE kernel, on the real model.

Everything else in this repo measures the kernel or one forward. This measures
what a user would feel: ``generate_cached`` on the real checkpoint, three arms
on one machine in one process.

    BF16            the speed target
    PACKED          quantized, dequantize-per-access -- what shipped before
    PACKED + fused  quantized, packed straight into the kernel

The three arms share the loaded weights, the decoder, the prompt and the seed,
so the only variable is how the expert GEMM gets its weights.

Usage::

    python benchmarks/bench_fused_e2e.py --repo ~/test_llada \\
        --weight-dir ~/test_llada/weights
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path


def _timed_generate(model, generate_cached, prompt_ids, gen_length, steps, block_length, runs):
    import torch

    latencies = []
    out = None
    for _ in range(runs + 1):  # first pass is warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = generate_cached(model, prompt_ids.clone(), gen_length=gen_length,
                                  steps=steps, block_length=block_length,
                                  temperature=0.0)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    latencies = latencies[1:]
    return sum(latencies) / len(latencies), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="inference repository root")
    ap.add_argument("--weight-dir", required=True)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    repo = str(Path(args.repo).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from transformers import AutoTokenizer

    import src.server as server
    from model_update.distributed import init_distributed
    from model_update.generate import generate_cached

    from LLaDA_Quant.api import quantize_model
    from LLaDA_Quant.config import QuantConfig
    from LLaDA_Quant.runtime import fused_block

    init_distributed()
    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    prompt = "The chemical symbol for gold is Au and for silver is"
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

    def _load():
        # server.load_model builds the UNFUSED model, loads the HF checkpoint
        # into it, then converts to fused expert blocks via
        # load_state_dict_from_unfused. Calling load_weights_tp on an
        # already-fused model instead silently leaves w1/w2 at their
        # torch.empty init -- the shapes still work and the benchmark still
        # times, it just measures random weights.
        server.load_model(args.weight_dir, device, "fast_dense")
        return server.MODEL

    def _mib():
        return torch.cuda.memory_allocated() / 2 ** 20

    rows = []

    print("Loading BF16 ...", flush=True)
    model = _load()
    bf16_mem = _mib()
    bf16_t, bf16_out = _timed_generate(model, generate_cached, prompt_ids,
                                       args.gen_length, args.steps,
                                       args.block_length, args.runs)
    rows.append(("BF16", bf16_t, bf16_mem, bf16_out))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading + quantizing INT{args.bits} (PACKED) ...", flush=True)
    model = _load()
    quantize_model(model, QuantConfig(bits=args.bits, group_size=args.group_size,
                                      targets=("expert",), execution_mode="packed",
                                      expect_expert_blocks=model.cfg.NL))
    gc.collect()
    torch.cuda.empty_cache()
    packed_mem = _mib()
    packed_t, packed_out = _timed_generate(model, generate_cached, prompt_ids,
                                           args.gen_length, args.steps,
                                           args.block_length, args.runs)
    rows.append(("PACKED (dequant/access)", packed_t, packed_mem, packed_out))

    switched = fused_block.install(model, strict=True)
    print(f"Switched {len(switched)} expert blocks to the fused kernel.", flush=True)
    fused_t, fused_out = _timed_generate(model, generate_cached, prompt_ids,
                                         args.gen_length, args.steps,
                                         args.block_length, args.runs)
    rows.append(("PACKED + fused W8A16", fused_t, _mib(), fused_out))

    print()
    print("=" * 78)
    print(f"  generate_cached  gen={args.gen_length} steps={args.steps} "
          f"block={args.block_length}  mean of {args.runs}")
    print("=" * 78)
    print(f"  {'arm':<26} {'time':>9} {'tok/s':>9} {'vs BF16':>9} {'resident':>11}")
    for name, t, mem, _ in rows:
        print(f"  {name:<26} {t:8.2f}s {args.gen_length / t:8.2f} "
              f"{bf16_t / t:8.2f}x {mem:9.0f} MiB")

    print()
    for name, _, _, out in rows:
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"  {name:<26} {text[:60]!r}")

    same = torch.equal(rows[1][3], rows[2][3])
    print()
    print(f"  PACKED vs PACKED+fused produce identical tokens: {same}")
    if not same:
        n = int((rows[1][3] != rows[2][3]).sum())
        print(f"    {n}/{rows[1][3].numel()} differ -- expected to be small; the two "
              "paths round at different points, they are not bit-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
