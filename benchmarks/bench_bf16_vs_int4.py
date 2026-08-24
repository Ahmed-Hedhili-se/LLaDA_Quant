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
from LLaDA_Quant.llada_repo import build_bf16_model, quantize_experts_streaming
from LLaDA_Quant.llada_repo import timer as _timer
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
# Model construction -- see LLaDA_Quant.llada_repo for the build/quantize
# helpers, which tools/quantize_checkpoint.py shares with this benchmark.
# --------------------------------------------------------------------------


def int4_config(group_size: int, search: str, search_grid: int = 24) -> QuantConfig:
    return QuantConfig(
        bits=4,
        group_size=group_size,
        targets=("expert",),
        execution_mode=ExecutionMode.PACKED.value,
        scale_search=search,
        search_grid=search_grid,
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
    config = int4_config(args.group_size, args.scale_search, args.search_grid)
    where = args.quantize_device or args.device

    if args.rebuild_for_int4:
        # Quantize FIRST, while the GPU is still empty. The scale search needs
        # several GB of transient VRAM per block; running it with the 13.7 GB
        # reference already resident is what OOMs. Afterwards the INT4 model is
        # only ~3.2 GB, so the reference fits comfortably alongside it.
        print(f"building the model to quantize, on {args.build_device} ...")
        int4 = build_bf16_model(repo, weight_dir, args.build_device)

        step = _timer(f"quantizing on {where} (group_size={args.group_size}, "
                      f"scale_search={args.scale_search}, grid={args.search_grid})")
        result = (
            quantize_model(int4, config)
            if where == "cpu"
            else quantize_experts_streaming(int4, config, where)
        )
        step()
        int4 = int4.to(args.device).eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"building the BF16 reference on {args.build_device} ...")
        bf16 = build_bf16_model(repo, weight_dir, args.build_device)
        memory = compare_resident_memory(bf16, int4, label="INT4 PACKED")
        print(f"moving BF16 reference to {args.device} ...")
        bf16 = bf16.to(args.device).eval()
        return bf16, int4, result, config, memory

    print(f"building BF16 model on {args.build_device} ...")
    bf16 = build_bf16_model(repo, weight_dir, args.build_device)
    print("deep-copying for INT4 (host RAM peak ~27 GB) ...")
    int4 = copy.deepcopy(bf16)

    step = _timer(f"quantizing on {where} (group_size={args.group_size}, "
                  f"scale_search={args.scale_search}, grid={args.search_grid})")
    result = (
        quantize_model(int4, config)
        if where == "cpu"
        else quantize_experts_streaming(int4, config, where)
    )
    step()
    memory = compare_resident_memory(bf16, int4, label="INT4 PACKED")
    print(f"moving both models to {args.device} ...")
    int4 = int4.to(args.device).eval()
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


_TOKENIZER = None


def get_tokenizer(repo: str, weight_dir: str):
    """Cached tokenizer. Loading it per call re-parses a 7.7 MB tokenizer.json."""
    global _TOKENIZER
    if _TOKENIZER is None:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(weight_dir, trust_remote_code=True)
    return _TOKENIZER


def encode_prompt(
    repo: str, weight_dir: str, prompt: str, chat_template: bool = False
) -> torch.Tensor:
    """Tokenize the prompt, optionally through the model's chat template.

    LLaDA-MoE-7B-A1B-**Instruct** was tuned on chat-formatted input. Feeding it
    a bare ``Question:/Answer:`` string is out of distribution, and it shows:
    the model emits one short span and then padding, so both BF16 and INT4
    produce the same degenerate output and the comparison measures nothing.
    """
    tok = get_tokenizer(repo, weight_dir)
    if chat_template:
        if getattr(tok, "chat_template", None) is None:
            raise ValueError(
                "--chat-template was requested but this tokenizer defines none; "
                "drop the flag or supply the formatted prompt yourself"
            )
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return torch.tensor([tok(prompt)["input_ids"]], dtype=torch.long)


def decode_tokens(
    repo: str, weight_dir: str, ids: torch.Tensor, skip_special: bool = True
) -> str:
    tok = get_tokenizer(repo, weight_dir)
    return tok.decode(ids.reshape(-1).tolist(), skip_special_tokens=skip_special)


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
    parser.add_argument("--search-grid", type=int, default=24,
                        help="clipping ratios tried by the MSE search; 8 captures "
                             "almost all of the gain at a third of the cost")
    parser.add_argument("--chat-template", action="store_true",
                        help="wrap the prompt in the model chat template. The "
                             "checkpoint is an Instruct model, so a bare prompt is "
                             "out of distribution and yields degenerate output")
    parser.add_argument("--build-device", default="cpu",
                        help="where to construct the model: 'cpu' is safe anywhere; "
                             "'cuda:0' is seconds instead of ~100s but needs ~14 GB "
                             "of VRAM during construction (use on a 40 GB+ card)")
    parser.add_argument("--quantize-device", default=None,
                        help="where to run quantization (default: --device). "
                             "'cpu' is tens of minutes on 6.4B weights")
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
        print(f"building BF16 model on {args.build_device} ...")
        bf16 = build_bf16_model(args.repo, args.weight_dir, args.build_device)
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
    prompt_ids = encode_prompt(
        args.repo, args.weight_dir, args.prompt, chat_template=args.chat_template
    )
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
    # Report how much of the generation is real content. When a prompt is out
    # of distribution the model emits a short span and then pads, and both
    # models "agree" on output that says nothing about quantization.
    for name, ids in (("BF16", int4_free.reference), ("INT4", int4_free.quantized)):
        tokens = ids.final_tokens
        kept = decode_tokens(args.repo, args.weight_dir, torch.tensor(tokens), True)
        raw = decode_tokens(args.repo, args.weight_dir, torch.tensor(tokens), False)
        print(f"  {name}: {len(tokens)} committed, "
              f"{len(kept.split())} word(s) after dropping special tokens")
        if len(kept.strip()) < 10:
            print(f"    WARNING: near-empty output. Raw: {raw[:160]!r}")
            print("    A degenerate generation makes the BF16/INT4 comparison "
                  "meaningless; try --chat-template.")
    for name, ids in (("BF16", int4_free.reference), ("INT4", int4_free.quantized)):
        text = decode_tokens(args.repo, args.weight_dir, torch.tensor(ids.final_tokens))
        print(f"    {name} text: {text!r}")

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
