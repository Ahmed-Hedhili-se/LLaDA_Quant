"""Serve the inference repo's own server with quantized expert weights.

The correctness harnesses in ``eval/correctness/`` are HTTP clients: they take
``--base-url`` and talk to a running server. So evaluating a quantized model
means *serving* one, not reimplementing the eval loop.

This replicates ``src/server.py::main()`` exactly -- same ``load_model``, same
batch collector/executor threads, same ``uvicorn.run(app)`` -- and inserts one
step between loading and serving: quantize ``server.MODEL`` in place. The
inference repository is imported, never modified.

``--no-quantize`` serves the unmodified BF16 model through this same launcher.
Use it for the baseline: then the only difference between the two runs is the
quantization itself, not launcher-versus-start.sh.

**Use ``--execution-mode reference`` for accuracy work.** It is numerically
identical to ``packed`` (a test asserts the reconstructed weights are equal)
but keeps the dequantized BF16 resident, so it runs at BF16 speed instead of
paying the ~250 ms per-forward dequantization tax. GSM8K at 1024 tokens is
~12,800 forwards; that difference is minutes versus an hour. It costs ~1.5x
memory, which a 48 GB card absorbs easily. Switch to ``packed`` only when you
are measuring memory rather than accuracy.

    # baseline
    python benchmarks/serve_quantized.py --repo ~/test_llada \\
        --weight-dir ~/test_llada/weights --no-quantize --port 8000

    # INT4, same path
    python benchmarks/serve_quantized.py --repo ~/test_llada \\
        --weight-dir ~/test_llada/weights --bits 4 --scale-search mse \\
        --execution-mode reference --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import torch

from LLaDA_Quant import QuantConfig, compare_resident_memory, quantize_model, resident_memory
from LLaDA_Quant.analysis import LLADA_MOE_7B_A1B


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="inference repository root")
    parser.add_argument("--weight-dir", default=None, help="defaults to <repo>/weights")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", default="fast_dense",
                        choices=["ours", "ours_kv", "fast_dense", "hf"])
    parser.add_argument("--no-quantize", action="store_true",
                        help="serve BF16 unmodified; the baseline arm")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-search", default="mse", choices=["amax", "mse"])
    parser.add_argument("--search-grid", type=int, default=8)
    parser.add_argument("--execution-mode", default="reference",
                        choices=["reference", "packed"],
                        help="reference: same numerics at BF16 speed, ~1.5x memory "
                             "(use for accuracy). packed: real memory reduction, "
                             "~7x slower per forward (use for memory)")
    parser.add_argument("--quantize-device", default=None,
                        help="where to run the scale search (default: the model's device)")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    weight_dir = args.weight_dir or os.path.join(repo, "weights")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # Do NOT pre-set MASTER_ADDR here. init_distributed() fills in the whole
    # single-process rendezvous (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE)
    # but only when MASTER_ADDR is absent:
    #
    #     if world_size == 1 and "MASTER_ADDR" not in os.environ:
    #
    # Setting just MASTER_ADDR defeats that guard and leaves RANK unset, which
    # fails inside torch's env:// rendezvous. Under torchrun all four are
    # already present and this is a no-op either way.
    import src.server as server
    from model_update.distributed import get_tp_rank, get_tp_size, init_distributed

    init_distributed()
    device = args.device or (
        f"cuda:{get_tp_rank()}" if torch.cuda.is_available() else "cpu"
    )
    if get_tp_size() > 1:
        device = f"cuda:{get_tp_rank()}"

    server.load_model(weight_dir, device, args.backend)

    if args.no_quantize:
        print(f"\nserving BF16 unmodified "
              f"({resident_memory(server.MODEL).total / 2**30:.2f} GiB resident)\n")
    else:
        if args.backend != "fast_dense":
            raise SystemExit(
                f"--backend {args.backend} does not build fused expert blocks, so "
                "there is nothing for the expert adapter to match. Use fast_dense."
            )
        config = QuantConfig(
            bits=args.bits,
            group_size=args.group_size,
            targets=("expert",),
            execution_mode=args.execution_mode,
            scale_search=args.scale_search,
            search_grid=args.search_grid,
            expect_expert_blocks=LLADA_MOE_7B_A1B.num_layers,
        )
        import copy

        before = copy.deepcopy(resident_memory(server.MODEL))
        print(f"\nquantizing served model: {config.to_json()}")
        result = quantize_model(server.MODEL, config)
        after = resident_memory(server.MODEL)
        print(result.summary())
        print(f"resident: {before.total / 2**30:.2f} GiB -> {after.total / 2**30:.2f} GiB "
              f"({after.total / before.total:.3f}x)")
        if config.mode.value == "reference":
            print("REFERENCE mode: identical numerics to packed, at BF16 speed, "
                  "using more memory. Correct choice for accuracy evaluation.\n")
        else:
            print("PACKED mode: real memory reduction, ~7x slower per forward.\n")

    # Expose what is actually being served. Without this there is no way to
    # tell a quantized server from a BF16 one over HTTP, and forgetting to
    # restart between arms silently produces two identical result files that
    # look like a clean "quantization changed nothing" finding.
    @server.app.get("/v1/quantization")
    def quantization_status() -> dict:
        return dict(served)

    print("=" * 62)
    print(f"  SERVING: {served['label']}")
    print(f"  verify with: curl -s http://localhost:{args.port}/v1/quantization")
    print("=" * 62 + "\n")

    if get_tp_size() > 1 and get_tp_rank() != 0:
        print(f"Rank {get_tp_rank()} waiting for generation tasks...")
        server.worker_loop()
        return

    if args.backend == "fast_dense" and get_tp_size() == 1:
        print(f"Starting batch collector + executor "
              f"(max_size={server.BATCH_MAX_SIZE}, wait={server.BATCH_WAIT_S}s)...")
        threading.Thread(target=server._batch_collector, daemon=True).start()
        threading.Thread(target=server._batch_executor, daemon=True).start()

    import uvicorn

    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
