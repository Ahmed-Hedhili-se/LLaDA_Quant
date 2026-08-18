"""Storage benchmark: resident memory and checkpoint size, measured.

Category A of the benchmark split. This measures **capacity only** and makes
no latency claim whatsoever — see ``bench_numerical.py`` for quantization
error and ``bench_moe_regime.py`` for whether a faster kernel is even worth
building.

Every number here comes from walking the live module tree or calling
``os.path.getsize``. Nothing is derived from the theoretical size of a packed
tensor, which is precisely how the previous benchmark reported a 47% saving
on a path that grew the model by 52%.

    python benchmarks/bench_storage.py --num-experts 64 --hidden 2048 --intermediate 1024
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import torch.nn as nn

from LLaDA_Quant import (
    ExecutionMode,
    QuantConfig,
    QuantizationManifest,
    quantize_and_measure,
    resident_memory,
    save_quantized_checkpoint,
)
from LLaDA_Quant.formats.safetensors import checkpoint_size_bytes


class FusedExpertBlock(nn.Module):
    """Stand-in with the exact fused layout the LLaDA adapter targets."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int, dtype: torch.dtype):
        super().__init__()
        torch.manual_seed(0)
        self.w1 = nn.Parameter((torch.randn(num_experts, 2 * intermediate, hidden) * 0.02).to(dtype))
        self.w2 = nn.Parameter((torch.randn(num_experts, hidden, intermediate) * 0.02).to(dtype))


def bf16_checkpoint_bytes(model: nn.Module, directory: str) -> int:
    from safetensors.torch import save_file

    path = os.path.join(directory, "bf16.safetensors")
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, path)
    return os.path.getsize(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    baseline = nn.Sequential(
        *[
            FusedExpertBlock(args.num_experts, args.hidden, args.intermediate, torch.bfloat16)
            for _ in range(args.layers)
        ]
    )
    base_resident = resident_memory(baseline)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        bf16_ckpt = bf16_checkpoint_bytes(baseline, tmp)
        for bits in (8, 4):
            for mode in (ExecutionMode.PACKED, ExecutionMode.REFERENCE):
                config = QuantConfig(
                    bits=bits,
                    group_size=128,
                    targets=("expert",),
                    execution_mode=mode.value,
                    expect_expert_blocks=args.layers,
                )
                model, result, comparison = quantize_and_measure(baseline, config)
                ckpt_dir = os.path.join(tmp, f"int{bits}-{mode.value}")
                save_quantized_checkpoint(
                    model,
                    QuantizationManifest(
                        source_checkpoint="synthetic", config=config, targets=result.targets
                    ),
                    ckpt_dir,
                )
                ckpt_bytes = checkpoint_size_bytes(ckpt_dir)
                rows.append(
                    {
                        "bits": bits,
                        "execution_mode": mode.value,
                        "packed": result.targets[0].packed,
                        "resident_bytes": comparison.quantized.total,
                        "resident_ratio_vs_bf16": round(comparison.ratio, 4),
                        "resident_is_saving": comparison.is_saving,
                        "checkpoint_bytes": ckpt_bytes,
                        "checkpoint_ratio_vs_bf16": round(ckpt_bytes / bf16_ckpt, 4),
                        "converted_weight_ratio": round(result.weight_ratio, 4),
                        "resident_by_dtype": comparison.quantized.by_dtype,
                    }
                )

    report = {
        "benchmark": "storage",
        "measures": "resident tensor bytes and on-disk artifact bytes",
        "does_not_measure": "latency or throughput of any kind",
        "shape": {
            "num_experts": args.num_experts,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "layers": args.layers,
        },
        "baseline_bf16": {
            "resident_bytes": base_resident.total,
            "checkpoint_bytes": bf16_ckpt,
        },
        "rows": rows,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"BF16 baseline: resident {base_resident.total / 2**20:8.2f} MiB   "
          f"checkpoint {bf16_ckpt / 2**20:8.2f} MiB")
    header = (
        f"{'bits':>5}{'mode':>12}{'packed':>8}{'resident MiB':>14}{'vs BF16':>10}"
        f"{'ckpt MiB':>11}{'vs BF16':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        flag = "" if row["resident_is_saving"] else "  <- LARGER than BF16"
        print(
            f"{row['bits']:>5}{row['execution_mode']:>12}{str(row['packed']):>8}"
            f"{row['resident_bytes'] / 2**20:>14.2f}{row['resident_ratio_vs_bf16']:>10.3f}"
            f"{row['checkpoint_bytes'] / 2**20:>11.2f}{row['checkpoint_ratio_vs_bf16']:>10.3f}{flag}"
        )
    print("\nResident bytes are measured from live tensors, not from packed-tensor arithmetic.")
    print("REFERENCE mode keeps BF16 weights alongside the packed ones on purpose;")
    print("it is a validation mode and is expected to be larger than the baseline.")


if __name__ == "__main__":
    main()
