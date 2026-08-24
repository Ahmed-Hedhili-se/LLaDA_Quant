"""Prove a saved artifact equals a startup-time quantization, byte for byte.

``tools/quantize_checkpoint.py`` claims the scale search is deterministic, so
the tensors it writes are exactly the ones a server would derive at startup.
That claim fails **silently** if it is wrong: a slightly different set of
scales still loads, still runs, and still produces plausible text. Nothing
downstream would report a problem, and every accuracy number measured against
the artifact would quietly describe a different model than the one the config
names.

So it gets checked. This rebuilds the model from the source weights in a fresh
process, requantizes it from scratch using the config recorded in the
artifact's own manifest, and compares every tensor against the file.

    python tools/check_determinism.py ~/llada-moe-int8-g128 \\
        --repo ~/test_llada --weight-dir ~/test_llada/weights

Exits non-zero on any mismatch, so it is usable as a gate.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from LLaDA_Quant.formats.safetensors import load_quantized_checkpoint
from LLaDA_Quant.llada_repo import build_bf16_model, quantize_experts_streaming

PACKED_LEAVES = ("_qw1", "_qw2", "_sw1", "_sw2")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifact", help="directory written by quantize_checkpoint.py")
    parser.add_argument("--repo", required=True, help="inference repository root")
    parser.add_argument("--weight-dir", default=None, help="defaults to <repo>/weights")
    parser.add_argument("--build-device", default="cuda:0")
    parser.add_argument("--quantize-device", default=None,
                        help="defaults to --build-device")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    weight_dir = args.weight_dir or os.path.join(repo, "weights")
    artifact = os.path.abspath(os.path.expanduser(args.artifact))

    state, manifest = load_quantized_checkpoint(artifact)
    config = manifest.config
    if config is None:
        raise SystemExit(
            f"{artifact} carries no quantization config, so there is nothing to "
            "reproduce. It was not written by tools/quantize_checkpoint.py."
        )
    print(f"artifact config: {config.to_json()}\n")

    # A fresh build matters: reusing an already-quantized model in this process
    # would compare the artifact against the tensors that produced it, which is
    # the save path's check, not this one.
    model = build_bf16_model(repo, weight_dir, args.build_device)
    quantize_experts_streaming(
        model, config, args.quantize_device or args.build_device, verbose=False
    )

    live = model.state_dict()
    checked = 0
    mismatched = []
    for name, stored in state.items():
        if name not in live:
            continue
        got = live[name].cpu()
        checked += 1
        if got.dtype != stored.dtype or got.shape != stored.shape:
            mismatched.append(
                f"{name}: {got.dtype} {tuple(got.shape)} vs "
                f"{stored.dtype} {tuple(stored.shape)}"
            )
        elif not torch.equal(got, stored):
            delta = (got.float() - stored.float()).abs().max().item()
            mismatched.append(f"{name}: same shape, values differ (max |delta| {delta:g})")

    packed = [n for n in state if n.rpartition(".")[2] in PACKED_LEAVES]
    print()
    print(f"compared      : {checked} tensors ({len(packed)} packed expert buffers)")
    print(f"mismatched    : {len(mismatched)}")
    for line in mismatched[:20]:
        print(f"  {line}")
    if len(mismatched) > 20:
        print(f"  ... and {len(mismatched) - 20} more")
    verdict = "IDENTICAL - offline == startup" if not mismatched else "DIFFERS"
    print(f"VERDICT       : {verdict}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
