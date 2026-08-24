"""Quantize the weights once, offline, into a standalone artifact.

Everything else in this repository quantizes at *startup*: build the BF16
model, run the scale search over 6.44 B expert weights, then serve. This does
that work once and writes the result to disk, so later runs load packed
integers instead of re-deriving them.

    python tools/quantize_checkpoint.py \\
        --repo ~/test_llada --weight-dir ~/test_llada/weights \\
        --out ~/llada-moe-int8-g128 \\
        --bits 8 --group-size 128 --scale-search mse \\
        --build-device cuda:0 --quantize-device cuda:0

**What the artifact changes and what it does not.** It changes the bytes on
disk, the provenance (the manifest pins bits, group size, search and the source
hash), and the startup cost of the scale search. It does **not** change
accuracy or inference speed by a single digit: the scale search is
deterministic, so an artifact produced here holds exactly the tensors a
startup-time run would have produced. There is a test for that determinism, and
``--verify`` re-reads the file and compares it against memory tensor by tensor.

**Residency is not baked in.** The saved bytes are the same for either
execution mode -- the BF16 experts are re-derivable and never stored -- so
``load_quantized_weights(..., execution_mode=...)`` picks PACKED or REFERENCE at
load time. One artifact serves both the deployment run and the accuracy run.

**One honest caveat about load time.** Loading this artifact into a served
model still builds the BF16 model first and then overwrites the experts,
because model construction belongs to the inference repository and this project
does not modify it. So the artifact saves the *scale search*, not the BF16
weight read. Skipping that too would need a meta-device build path on the
inference side.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from LLaDA_Quant import (
    ExecutionMode,
    QuantConfig,
    QuantizationManifest,
    quantize_model,
    resident_memory,
    save_quantized_checkpoint,
)
from LLaDA_Quant.analysis import LLADA_MOE_7B_A1B
from LLaDA_Quant.formats.safetensors import (
    checkpoint_size_bytes,
    collect_quantized_state,
    derivable_tensor_names,
    find_weights_file,
    load_quantized_checkpoint,
)
from LLaDA_Quant.llada_repo import build_bf16_model, quantize_experts_streaming, timer

GIB = 2 ** 30


def directory_bytes(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(path)
        for name in files
    )


def verify_artifact(directory: str, model, manifest: QuantizationManifest) -> None:
    """Re-read the artifact and prove it matches memory, tensor by tensor.

    A checkpoint writer that silently drops or truncates a tensor produces a
    model that loads and runs and is quietly wrong. Comparing against the live
    state dict is cheap next to the hours the artifact will be used for.
    """
    step = timer("verifying the artifact against memory")
    state, restored = load_quantized_checkpoint(directory)
    expected = collect_quantized_state(model, manifest)

    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise SystemExit(f"artifact mismatch: missing={missing[:8]}, extra={extra[:8]}")

    derivable = derivable_tensor_names(manifest)
    stored_derivable = sorted(derivable & set(state))
    if stored_derivable:
        raise SystemExit(
            "artifact stores re-derivable BF16 expert weights, which is exactly the "
            f"bug the format exists to prevent: {stored_derivable[:8]}"
        )

    for name, want in expected.items():
        got = state[name]
        if got.dtype != want.dtype or got.shape != want.shape:
            raise SystemExit(
                f"{name}: stored {got.dtype} {tuple(got.shape)}, "
                f"memory {want.dtype} {tuple(want.shape)}"
            )
        if not torch.equal(got, want.cpu()):
            raise SystemExit(f"{name}: stored bytes differ from memory")

    if restored.config != manifest.config:
        raise SystemExit(
            f"manifest did not round-trip: {restored.config} != {manifest.config}"
        )
    step()
    print(f"    {len(expected)} tensors verified bit-exact, "
          f"{len(derivable)} re-derivable tensors correctly absent")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", required=True, help="inference repository root")
    parser.add_argument("--weight-dir", default=None, help="defaults to <repo>/weights")
    parser.add_argument("--out", required=True, help="directory to write the artifact to")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-search", default="mse", choices=["amax", "mse"])
    parser.add_argument("--search-grid", type=int, default=24)
    parser.add_argument("--build-device", default="cpu",
                        help="where to construct the BF16 model (cuda:0 is ~80x faster "
                             "but needs ~14 GB of VRAM)")
    parser.add_argument("--quantize-device", default=None,
                        help="where to run the scale search; defaults to --build-device")
    parser.add_argument("--overwrite", action="store_true",
                        help="write into --out even if it already holds a checkpoint")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the read-back check (not recommended)")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    weight_dir = args.weight_dir or os.path.join(repo, "weights")
    out = os.path.abspath(os.path.expanduser(args.out))

    if os.path.isdir(out) and not args.overwrite:
        try:
            existing = find_weights_file(out)
        except FileNotFoundError:
            existing = None
        if existing:
            raise SystemExit(
                f"{out} already holds {os.path.basename(existing)}. Pass --overwrite "
                "to replace it, or choose another --out. Two weight files in one "
                "directory make the checkpoint ambiguous and unloadable."
            )

    config = QuantConfig(
        bits=args.bits,
        group_size=args.group_size,
        targets=("expert",),
        # PACKED is what the artifact is for. It is not locked in: the loader
        # takes execution_mode= and the bytes are identical either way.
        execution_mode=ExecutionMode.PACKED.value,
        scale_search=args.scale_search,
        search_grid=args.search_grid,
        expect_expert_blocks=LLADA_MOE_7B_A1B.num_layers,
    )
    print(f"config: {config.to_json()}\n")

    model = build_bf16_model(repo, weight_dir, args.build_device)
    before = resident_memory(model).total

    where = args.quantize_device or args.build_device
    step = timer(f"quantizing on {where}")
    result = (
        quantize_model(model, config)
        if where == "cpu"
        else quantize_experts_streaming(model, config, where)
    )
    step()
    after = resident_memory(model).total
    print(result.summary())
    print(f"resident: {before / GIB:.2f} GiB -> {after / GIB:.2f} GiB "
          f"({after / before:.3f}x)\n")

    manifest = QuantizationManifest(
        source_checkpoint=weight_dir, config=config, targets=result.targets
    )
    step = timer(f"writing the artifact to {out}")
    path = save_quantized_checkpoint(model, manifest, out)
    step()

    if not args.no_verify:
        verify_artifact(out, model, manifest)

    source_bytes = directory_bytes(weight_dir)
    artifact_bytes = checkpoint_size_bytes(out)
    print("\n" + "=" * 62)
    print(f"  wrote {path}")
    print(f"  source checkpoint : {source_bytes / GIB:8.2f} GiB")
    print(f"  quantized artifact: {artifact_bytes / GIB:8.2f} GiB "
          f"({artifact_bytes / source_bytes:.3f}x)")
    print("=" * 62)
    print(json.dumps({
        "artifact": out,
        "bits": args.bits,
        "group_size": args.group_size,
        "scale_search": args.scale_search,
        "source_bytes": source_bytes,
        "artifact_bytes": artifact_bytes,
        "ratio": round(artifact_bytes / source_bytes, 4),
    }, indent=2))
    print("\nServe it with:")
    print("  python benchmarks/serve_quantized.py \\")
    print(f"      --repo {repo} --weight-dir {weight_dir} \\")
    print(f"      --quantized-checkpoint {out} --execution-mode packed --fused")


if __name__ == "__main__":
    main()
