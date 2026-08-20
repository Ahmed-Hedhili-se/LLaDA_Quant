"""BF16 vs INT4-MSE on the real LLaDA-MoE checkpoint: routing, commits, output.

The experiment:

    same prompt, same masked input, same deterministic decoding, same seed
    -> router overlap, token commits, final output

Requires a GPU with Triton and the real weights, and the inference repository
importable. **It never modifies that repository** — it imports the model, the
weight loader and the production decoder, exactly as
``trajectory/llada.py`` documents.

Three runs, and all three are needed to read any of them:

  1. Mode A, shared state   — BF16 vs INT4 fed byte-identical masked inputs.
     Isolates error injected per forward: logit divergence, per-layer router
     overlap, router margins, tie fraction. Cannot show amplification.
  2. Mode B, free running   — each model denoises itself through the *real*
     commit rule. Shows amplification: which positions each commits, in which
     order, and how far the final sequences drift apart.
  3. Noise floor, BF16 vs BF16 — the same two runs with the reference against
     itself. At temperature 0 this should be exactly clean; if it is not, the
     harness is non-deterministic and runs 1 and 2 mean nothing.

Determinism: ``temperature=0.0`` makes ``add_gumbel_noise`` a no-op (it returns
its input unchanged), so the whole trajectory is a deterministic function of
the logits. Nothing here samples.

Usage::

    python benchmarks/bench_bf16_vs_int4.py \\
        --repo ../test_llada --weight-dir ../test_llada/weights \\
        --prompt "Natalia sold clips to 48 friends in April..." \\
        --gen-length 32 --steps 32 --out traces/

    # sanity-check the harness first, without quantizing anything:
    python benchmarks/bench_bf16_vs_int4.py ... --noise-floor-only
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

from LLaDA_Quant import (
    ExecutionMode,
    QuantConfig,
    compare_resident_memory,
    quantize_model,
    resident_memory,
)
from LLaDA_Quant.analysis import LLADA_MOE_7B_A1B, expert_token_stats
from LLaDA_Quant.trajectory import (
    LLADA_MASK_ID,
    TrajectoryReport,
    attach_router_capture,
    capture_free_running,
    capture_shared,
    fully_masked_state,
    gates_fn_for,
    load_llada_decoder,
    make_llada_advance_fn,
    make_masked_states,
    replay_free_running,
    replay_shared,
    router_fn_for,
    verify_replay,
)


# --------------------------------------------------------------------------
# Model construction (mirrors compare_models.py::load_ours, read-only)
# --------------------------------------------------------------------------


def _timer(label: str):
    """Print a stage banner now and its duration when the returned fn is called."""
    import time

    print(f"  {label} ...", flush=True)
    start = time.perf_counter()

    def done() -> None:
        print(f"  {label}: {time.perf_counter() - start:.1f}s", flush=True)

    return done


def build_bf16_model(repo: str, weight_dir: str):
    """Build the fused-MoE BF16 model on CPU, the way the inference repo does.

    Deliberately stays on the host: both models have to be resident at once for
    the comparison, and 2 x 12.88 GB of BF16 experts does not fit on a 24 GB
    card. Quantize first, move second.
    """
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model_update.model import LLaDAMoEKV, TritonFusedMoEBlock
    from src.model import load_weights

    # Construct directly in BF16. The unfused model is 3072 expert Linears
    # (64 experts x 3 projections x 16 layers); at the default fp32 that is
    # ~25.6 GB of randomly initialised host memory, plus another 12.8 GB for
    # the .to(bfloat16) copy. On a box with less RAM than that it swaps, and
    # construction takes tens of minutes instead of a couple.
    step = _timer("allocating unfused model (BF16)")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = LLaDAMoEKV(use_fused_moe=False)
    finally:
        torch.set_default_dtype(previous_dtype)
    model = model.to(torch.bfloat16).eval()
    step()

    step = _timer("loading weights")
    load_weights(model, weight_dir, verbose=False)
    step()

    # Fuse AFTER loading: load_state_dict_from_unfused writes w1[i] in place,
    # which is precisely what PACKED mode cannot accept.
    step = _timer(f"fusing {len(model.layers)} MoE blocks")
    for index, layer in enumerate(model.layers):
        fused = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16)
        fused.load_state_dict_from_unfused(layer.mlp)
        layer.mlp = fused
        print(f"    layer {index + 1}/{len(model.layers)} fused", flush=True)
    step()
    return model.eval()


def int4_config(group_size: int, search: str) -> QuantConfig:
    return QuantConfig(
        bits=4,
        group_size=group_size,
        targets=("expert",),
        execution_mode=ExecutionMode.PACKED.value,
        scale_search=search,
        expect_expert_blocks=LLADA_MOE_7B_A1B.num_layers,
    )


def build_model_pair(repo: str, weight_dir: str, args):
    """BF16 and INT4 models, both on ``args.device``, without a 26 GB VRAM peak.

    Order matters. Quantization happens on the host, so the INT4 copy is 3.22 GB
    before either model is moved; the pair then needs ~16.1 GB of VRAM instead of
    ~25.8 GB. On a 24 GB card the naive order (move, then deep-copy) OOMs.

    ``--rebuild-for-int4`` moves the reference to the GPU *before* building the
    second model, so only one model is ever in host RAM: peak ~13.7 GB instead
    of ~27.4 GB. It costs a second weight load (~90 s). Worth it on a box that
    swaps -- swapping turns a two-minute build into half an hour.
    """
    config = int4_config(args.group_size, args.scale_search)

    print("building BF16 model on CPU ...")
    bf16 = build_bf16_model(repo, weight_dir)

    if args.rebuild_for_int4:
        # Park the reference on the GPU first; the host then holds one model
        # at a time. Deep-copying instead would keep both here at once.
        print(f"moving BF16 to {args.device} to free host RAM ...")
        bf16 = bf16.to(args.device).eval()
        print("rebuilding a second copy for INT4 (--rebuild-for-int4) ...")
        int4 = build_bf16_model(repo, weight_dir)
    else:
        print("deep-copying for INT4 (host RAM peak ~27 GB) ...")
        int4 = copy.deepcopy(bf16)

    step = _timer(f"quantizing on CPU (group_size={args.group_size}, "
                  f"scale_search={args.scale_search})")
    result = quantize_model(int4, config)
    step()

    # Byte accounting only; it does not care which device either model is on.
    memory = compare_resident_memory(bf16, int4, label="INT4 PACKED")

    print(f"moving INT4 to {args.device} ...")
    int4 = int4.to(args.device).eval()
    if not args.rebuild_for_int4:
        print(f"moving BF16 to {args.device} ...")
        bf16 = bf16.to(args.device).eval()
    return bf16, int4, result, config, memory


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------


def make_logits_fn(device: str):
    """Dense forward, no KV cache — one deterministic function of input_ids."""

    def logits_fn(model, state):
        ids = state.input_ids.to(device)
        out = model(ids)
        return (out[0] if isinstance(out, tuple) else out).float()

    return logits_fn


def encode_prompt(repo: str, weight_dir: str, prompt: str) -> torch.Tensor:
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)
    return torch.tensor([tok(prompt)["input_ids"]], dtype=torch.long)


def decode_tokens(repo: str, weight_dir: str, ids: torch.Tensor) -> str:
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)
    return tok.decode(ids.reshape(-1).tolist(), skip_special_tokens=True)


# --------------------------------------------------------------------------
# The three runs
# --------------------------------------------------------------------------


def run_pair(label, reference, quantized, prompt_ids, args, decoder, logits_fn):
    """Mode A + Mode B for one model pair, with router capture on both."""
    ref_cap = attach_router_capture(reference)
    qnt_cap = attach_router_capture(quantized)
    router_fn = router_fn_for(ref_cap, qnt_cap)
    gates_fn = gates_fn_for(ref_cap, qnt_cap)
    try:
        # --- Mode A: identical masked inputs at several denoising ratios ----
        completion = torch.full(
            (1, args.gen_length), LLADA_MASK_ID, dtype=torch.long
        )
        if args.reference_completion:
            completion = torch.tensor(
                [json.loads(args.reference_completion)], dtype=torch.long
            )
        states = make_masked_states(
            prompt_ids,
            completion,
            LLADA_MASK_ID,
            ratios=tuple(args.ratios),
            generator=torch.Generator().manual_seed(args.seed),
        )
        shared = capture_shared(
            reference,
            quantized,
            states,
            logits_fn,
            router_fn,
            gates_fn,
            unmask_k=max(1, args.gen_length // args.steps),
            labels=(f"{label}-reference", f"{label}-quantized"),
        )
        mode_a = replay_shared(shared.reference, shared.quantized)

        # --- Mode B: each model drives itself through the real commit rule --
        advance = make_llada_advance_fn(
            decoder, steps=args.steps, mask_token_id=LLADA_MASK_ID, temperature=0.0
        )
        start = fully_masked_state(prompt_ids, args.gen_length, LLADA_MASK_ID)
        free = capture_free_running(
            reference,
            quantized,
            start,
            logits_fn,
            advance,
            max_steps=args.steps,
            seed=args.seed,
            labels=(f"{label}-reference", f"{label}-quantized"),
        )
        mode_b = replay_free_running(free.reference, free.quantized)
    finally:
        ref_cap.remove()
        qnt_cap.remove()

    if args.out:
        shared.save(args.out, prefix=f"{label}-modeA")
        free.save(args.out, prefix=f"{label}-modeB")
    return shared, free, mode_a, mode_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="path to the inference repository")
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--prompt", default="Question: What is 17 times 24?\nAnswer:")
    parser.add_argument("--gen-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-search", choices=("amax", "mse"), default="mse")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--reference-completion", default=None,
                        help="JSON list of token ids to reveal in Mode A "
                             "(default: fully masked at every ratio)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="directory for traces + report")
    parser.add_argument("--noise-floor-only", action="store_true",
                        help="run only BF16 vs BF16, to validate determinism first")
    parser.add_argument("--rebuild-for-int4", action="store_true",
                        help="hold one model in host RAM at a time (~13.7 GB peak "
                             "instead of ~27.4 GB) at the cost of a second weight load")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    if args.noise_floor_only:
        print("building BF16 model on CPU ...")
        bf16 = build_bf16_model(args.repo, args.weight_dir)
        bf16_mem = resident_memory(bf16)
        print(f"  resident: {bf16_mem.total / 2**30:.2f} GiB")
        print(f"moving to {args.device} ...")
        bf16 = bf16.to(args.device).eval()
        int4 = result = config = memory = None
    else:
        bf16, int4, result, config, memory = build_model_pair(
            args.repo, args.weight_dir, args
        )
        bf16_mem = resident_memory(bf16)
        print("  " + result.summary().replace("\n", "\n  "))
        print("  " + memory.describe())

    decoder = load_llada_decoder(args.repo)
    print(f"  decoder: {decoder.describe()}")
    logits_fn = make_logits_fn(args.device)
    prompt_ids = encode_prompt(args.repo, args.weight_dir, args.prompt)
    print(f"  prompt: {prompt_ids.shape[1]} tokens, generating {args.gen_length}")

    # --- 3. noise floor first: BF16 against itself ------------------------
    print("\n[floor] BF16 vs BF16 (must be exactly clean at temperature 0) ...")
    _, floor_free, floor_a, floor_b = run_pair(
        "floor", bf16, bf16, prompt_ids, args, decoder, logits_fn
    )
    clean = (
        floor_a.summary["min_top1_agreement"] == 1.0
        and floor_b.summary["final_token_agreement"] == 1.0
    )
    print(f"  Mode A min top-1 agreement : {floor_a.summary['min_top1_agreement']:.6f}")
    print(f"  Mode B final agreement     : {floor_b.summary['final_token_agreement']:.6f}")
    print(f"  deterministic: {clean}")
    if not clean:
        print("  WARNING: the floor is not clean. Any INT4 number below is unreadable")
        print("  until this is fixed -- it is measuring harness noise, not quantization.")

    report: dict = {
        "experiment": "bf16_vs_int4",
        "deterministic_floor": clean,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "gen_length": args.gen_length,
        "steps": args.steps,
        "seed": args.seed,
        "temperature": 0.0,
        "bf16_resident_bytes": bf16_mem.total,
        "noise_floor": {"mode_a": floor_a.summary, "mode_b": floor_b.summary},
    }

    if args.noise_floor_only:
        print("\n--noise-floor-only: stopping before quantization.")
        _emit(report, args)
        return

    # --- routing statistics, free: one forward already happened -----------
    cap = attach_router_capture(bf16)
    try:
        with torch.no_grad():
            logits_fn(bf16, fully_masked_state(prompt_ids, args.gen_length, LLADA_MASK_ID))
        routing = {
            name: expert_token_stats(ids, LLADA_MOE_7B_A1B.num_experts).to_dict()
            for name, ids in cap.topk_ids.items()
        }
    finally:
        cap.remove()
    report["routing_balance"] = routing
    imbalances = [v["imbalance_max_over_mean"] for v in routing.values()]
    if imbalances:
        print(f"\n[routing] measured imbalance (max/mean) across {len(imbalances)} layers: "
              f"min={min(imbalances):.2f} max={max(imbalances):.2f}")

    # --- 1 + 2. INT4-MSE vs BF16 -----------------------------------------
    print("\n[int4] Mode A + Mode B against BF16 ...")
    _, int4_free, mode_a, mode_b = run_pair(
        "int4", bf16, int4, prompt_ids, args, decoder, logits_fn
    )

    trajectory = TrajectoryReport(
        mode_a=mode_a, mode_b=mode_b,
        noise_floor_a=floor_a, noise_floor_b=floor_b,
        label=f"INT4-{args.scale_search} g{args.group_size}",
    )
    print("\n" + trajectory.to_table())

    problems = verify_replay(mode_a)
    if problems:
        print("\nreplay/capture disagreement:")
        for line in problems:
            print("  " + line)

    print("\n[final output]")
    for name, ids in (("BF16", int4_free.reference), ("INT4", int4_free.quantized)):
        text = decode_tokens(args.repo, args.weight_dir, torch.tensor(ids.final_tokens))
        print(f"  {name}: {text!r}")

    report.update({
        "quantization": {
            "config": config.to_dict(),
            "targets": len(result.targets),
            "weight_ratio": result.weight_ratio,
            "resident": memory.to_dict(),
        },
        "trajectory": trajectory.to_dict(),
        "final_tokens": {
            "bf16": int4_free.reference.final_tokens,
            "int4": int4_free.quantized.final_tokens,
        },
    })
    _emit(report, args)


def _emit(report: dict, args) -> None:
    if not args.out:
        return
    path = os.path.join(args.out, "bf16_vs_int4.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
